use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use smc_thrift::get_smc2_client;
use tokio_retry::Retry;
use tokio_retry::strategy::FixedInterval;
use tokio_retry::strategy::jitter;

// Timeout on reading SMC tier.
const SMC_READ_TIMEOUT: Duration = Duration::from_secs(60);
// Minimum time to wait between smc read retries.
const SMC_MIN_RETRY_INTERVAL: Duration = Duration::from_secs(1);
// Maximum time to wait between smc read retries.
const SMC_MAX_RETRY_INTERVAL: Duration = Duration::from_secs(2);

pub fn canonicalize_hostname(hostname: &str) -> String {
    if !hostname.ends_with(".facebook.com") {
        format!("{}.facebook.com", hostname)
    } else {
        hostname.to_string()
    }
}

#[derive(Clone)]
pub struct SMCClient {
    client: Arc<dyn smc2_clients::Smc2 + Send + Sync>,
    tier: String,
}

impl SMCClient {
    pub fn new(fb: fbinit::FacebookInit, tier: String) -> Result<Self, anyhow::Error> {
        match get_smc2_client(fb, None) {
            Ok(client) => Ok(Self { client, tier }),
            Err(err) => Err(anyhow::anyhow!("failed to create SMC client: {}", err)),
        }
    }

    /// Retrieve the system address from SMC tier.
    /// Return the hostname and port of the system.
    pub async fn get_system_address(&self) -> Result<(String, u16), anyhow::Error> {
        tracing::info!(
            "Reading SMC tier {}, this will block until address is available",
            self.tier
        );
        let strategy = FixedInterval::new(SMC_MIN_RETRY_INTERVAL)
            .map(|d| d + jitter(SMC_MAX_RETRY_INTERVAL - SMC_MIN_RETRY_INTERVAL));
        // Filter only active services.
        let filter = BTreeMap::from([("enabled".to_string(), vec![])]);
        let services = Retry::spawn(strategy, || async {
            match tokio::time::timeout(
                SMC_READ_TIMEOUT,
                self.client.getFilteredTierByName(&self.tier, &filter),
            )
            .await
            {
                Ok(inner @ Ok(_)) => {
                    let tier = inner?;
                    if tier.services.is_empty() {
                        Err(anyhow::anyhow!("no system found in SMC tier {}", self.tier).into())
                    } else {
                        Ok(tier)
                    }
                }
                Ok(Err(err)) => {
                    tracing::error!("Reading SMC tier {} failed: {}", self.tier, err);
                    Err(err)
                }
                Err(_) => {
                    tracing::error!(
                        "Reading SMC tier {}: timeout after {:?}",
                        self.tier,
                        SMC_READ_TIMEOUT
                    );
                    Err(anyhow::anyhow!("timeout after {:?}", SMC_READ_TIMEOUT).into())
                }
            }
        })
        .await?
        .services;

        // MAST smcBridge will publish a service for each taskgroup task. In situation where
        // we co-locate the system with workers, we may get more than 1 service.
        // In that case we will just pick the first one.
        // Note that this will be consistent as the same hostname we obtain here will be
        // used to determine if we are a system host.
        if services.len() > 1 {
            return Err(anyhow::anyhow!(
                "more than one system ({}) found in SMC tier {}",
                services.len(),
                self.tier
            ));
        }

        let service = &services[0];
        let hostname = service.hostname.clone();
        let port = service.port as u16;
        tracing::info!(
            "Found system {}:{} in SMC tier {}",
            hostname,
            port,
            self.tier
        );
        Ok((hostname, port))
    }
}

#[cfg(test)]
mod tests {
    use std::net::ToSocketAddrs;

    use fbinit::FacebookInit;
    use smc2_clients::errors::CreateTierError;
    use uuid::Uuid;

    use super::*;

    /// Map a hostname-port pair to a socket address.
    fn to_socket_addrs(host: &str, port: u16) -> Result<std::net::SocketAddr, anyhow::Error> {
        let mut addrs = (host, port).to_socket_addrs()?;
        match addrs.next() {
            Some(first_addr) => {
                if addrs.next().is_some() {
                    Err(anyhow::anyhow!(
                        "more than one addresses found for {}: {}, {:?}",
                        host,
                        first_addr,
                        addrs
                    ))?
                } else {
                    Ok(first_addr)
                }
            }
            None => Err(anyhow::anyhow!("no addresses found for {}", host)),
        }
    }

    impl SMCClient {
        /// Publish the system address to SMC tier with system's hostname and its port.
        async fn publish_system_address_for_unit_tests(
            &self,
            hostname: &str,
            port: u16,
        ) -> Result<(), anyhow::Error> {
            tracing::info!(
                "creating SMC tier {} with system address {}:{}",
                self.tier,
                hostname,
                port
            );
            let system_addr = to_socket_addrs(hostname, port)?;
            let tier = &smc::Tier {
                name: self.tier.clone(),
                state: i32::from(smc::TierState::ENABLED_TIER)
                    | i32::from(smc::TierState::DEV_TIER),
                services: vec![smc::Service {
                    hostname: hostname.to_string(),
                    port: port as i32,
                    key: Some("system".to_string()),
                    state: i32::from(smc::ServiceState::ENABLED) as i16,
                    ipaddr: system_addr.ip().to_string(),
                    ..Default::default()
                }],
                props: BTreeMap::from([(
                    "smc_oncall".to_string(),
                    smc::Property {
                        value: "monarch".to_string(),
                        data_type: smc::PropertyDataType::STRING,
                        ..Default::default()
                    },
                )]),
                ..Default::default()
            };

            match self.client.createTier(tier).await {
                Ok(()) => {
                    tracing::info!("created SMC tier {}", self.tier);
                    Ok(())
                }
                Err(CreateTierError::ex(e)) if e.message.contains("already exists") => {
                    tracing::warn!("SMC tier {} already exists; update tier", self.tier);
                    // There is a chance that during a full MAST recovery, the workers can read the old system address.
                    // The workers will not be able to send messages. The corresponding workers will fail.
                    // When they come back again. They will read the new system address and connect to the new system.
                    self.client.updateTier(tier).await?;
                    Ok(())
                }
                Err(e) => {
                    tracing::error!("failed to create SMC tier {}", e);
                    Err(e)
                }
            }?;
            Ok(())
        }
    }

    // SMC tier updates with the same parent tier can fail due to race conditions.
    // This is particularly serious for the stress tests where we can send requests in parallel to the same parent tier.
    async fn retry_operation<F, Fut>(
        max_retries: u32,
        delay_ms: u64,
        operation: F,
    ) -> Result<(), anyhow::Error>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<(), anyhow::Error>>,
    {
        let mut retries = 0;
        loop {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    retries += 1;
                    if retries >= max_retries {
                        anyhow::bail!("Failed after {} retries: {}", max_retries, e);
                    }
                    #[allow(clippy::disallowed_methods)]
                    tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms)).await;
                }
            }
        }
    }

    #[test]
    fn test_fqdn() {
        assert_eq!(
            "twshared19437.05.nha2.facebook.com",
            canonicalize_hostname("twshared19437.05.nha2"),
        );
        assert_eq!(
            "twshared19437.05.nha2.facebook.com",
            canonicalize_hostname("twshared19437.05.nha2.facebook.com"),
        );
    }

    #[fbinit::test]
    async fn test_smc_client(fb: FacebookInit) {
        let tier = format!("monarch.unittest.{}", Uuid::new_v4());
        let smc_client = SMCClient::new(fb, tier).unwrap();

        const MAX_RETRIES: u32 = 20;
        const RETRY_DELAY_MS: u64 = 100;

        let test = async {
            // NB: use anyhow::ensure!() over assert_*!() since this block
            //     should return an Err rather than panic
            //     otherwise the deleteTier cleanup won't run down below

            let control_host = "twshared19437.05.nha2.facebook.com";
            let control_port = 12345;

            // normal publish
            retry_operation(MAX_RETRIES, RETRY_DELAY_MS, || {
                smc_client.publish_system_address_for_unit_tests(control_host, control_port)
            })
            .await?;

            let (test_host, test_port) = smc_client.get_system_address().await.unwrap();
            anyhow::ensure!(control_host == test_host);
            anyhow::ensure!(control_port == test_port);

            // publish again
            retry_operation(MAX_RETRIES, RETRY_DELAY_MS, || {
                smc_client.publish_system_address_for_unit_tests(control_host, control_port)
            })
            .await?;

            let (test_host, test_port) = smc_client.get_system_address().await.unwrap();
            anyhow::ensure!(control_host == test_host);
            anyhow::ensure!(control_port == test_port);

            // publish with different port
            let control_port = 54321;
            retry_operation(MAX_RETRIES, RETRY_DELAY_MS, || {
                smc_client.publish_system_address_for_unit_tests(control_host, control_port)
            })
            .await?;

            let (test_host, test_port) = smc_client.get_system_address().await?;
            anyhow::ensure!(control_host == test_host);
            anyhow::ensure!(control_port == test_port);

            Ok(())
        }
        .await;

        // clean up - always runs since the test block will return a Result rather than panic'ing
        retry_operation(MAX_RETRIES, RETRY_DELAY_MS, || async {
            smc_client
                .client
                .deleteTier(&smc_client.tier)
                .await
                .map_err(Into::into)
        })
        .await
        .expect("Error deleting smc tier during test teardown");

        // assert test block
        test.unwrap();
    }
}
