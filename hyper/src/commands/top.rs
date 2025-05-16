use std::collections::HashMap;
use std::env;
use std::sync::Arc;

use anyhow::Context;
use anyhow::Result;
use employee::EmployeeClient;
use employee::Unixname;
use hyperactor_multiprocess::system_actor::SYSTEM_ACTOR_REF;
use hyperactor_multiprocess::system_actor::SystemMessageClient;
use hyperactor_multiprocess::system_actor::SystemSnapshotFilter;
use rfe_thrift::SQLQueryResult;
use rfe_thrift_srclients::RockfortExpress;
use rfe_thrift_srclients::make_RockfortExpress_srclient;
use tui::top::ActorInfo;
use tui::top::App;
use utils::system_address::SystemAddr;

#[derive(clap::Args, Debug)]
pub struct TopCommand {
    /// The address of the system. Can be a channel address or a MAST job name.
    system_addr: SystemAddr,
}

impl TopCommand {
    pub async fn run(self) -> anyhow::Result<()> {
        color_eyre::install().unwrap();
        let mut terminal = ratatui::init();
        let mut system = hyperactor_multiprocess::System::new(self.system_addr.into());
        let client = system.attach().await.expect("failed to attach to system");

        let snapshot = SYSTEM_ACTOR_REF
            .snapshot(&client, SystemSnapshotFilter::all())
            .await
            .expect("failed to snapshot system");
        let execution_id = snapshot.execution_id;

        let user = env::var("USER").context("Failed to get USER environment variable")?;
        let fb = fbinit::expect_init();
        let client = make_RockfortExpress_srclient!(fb)?;
        let employee_client = EmployeeClient::new(fb)?;
        let fbid = employee_client
            .get_fbid_from_unixname(Unixname(user.clone()))
            .await?
            .ok_or_else(|| anyhow::anyhow!("Couldn't get FBID for user {}", user))?
            .0 as i64;

        let mut app = App::new(execution_id.clone());
        let fetch_actors = async || {
            fetch_actors_from_scuba(client.clone(), user.as_str(), execution_id.as_str(), fbid)
                .await
        };
        let result = app.run(&mut terminal, fetch_actors).await;

        ratatui::restore();
        result
    }
}

async fn fetch_actors_from_scuba(
    client: Arc<dyn RockfortExpress + Send + Sync>,
    user: &str,
    execution_id: &str,
    fbid: i64,
) -> Result<Vec<ActorInfo>> {
    let ts = max_ts_sec(client.clone(), user, fbid, execution_id).await?;

    let query = format!(
        "
        SELECT
            dest_actor_id, SUM(sum)
        FROM monarch_metrics
        WHERE
            execution_id = '{}'
            AND dest_actor_id is not NULL
            AND time={}
        GROUP BY dest_actor_id
        ",
        execution_id, ts,
    );
    let result = client
        .querySQL(
            &rfe_thrift::QueryCommon {
                user_name: Some(user.to_string()),
                user_id: Some(fbid),
                ..Default::default()
            },
            &query,
        )
        .await?;

    let new_actors = scuba_result_to_message_count(result)?;

    let mut actors = new_actors
        .into_iter()
        .map(|(actor_id, message_count)| ActorInfo {
            actor_id,
            message_count,
        })
        .collect::<Vec<_>>();

    actors.sort_by(|a, b| b.message_count.cmp(&a.message_count));

    Ok(actors)
}

/// find out the max time of a scuba log
async fn max_ts_sec(
    client: Arc<dyn RockfortExpress + Send + Sync>,
    user: &str,
    user_id: i64,
    execution_id: &str,
) -> Result<u64> {
    let query = format!(
        "SELECT MAX(time) FROM monarch_metrics WHERE execution_id='{}'",
        execution_id
    );
    let result = client
        .querySQL(
            &rfe_thrift::QueryCommon {
                user_name: Some(user.to_string()),
                user_id: Some(user_id),
                ..Default::default()
            },
            &query,
        )
        .await?;
    let ts_sec = result
        .value
        .first()
        .ok_or(anyhow::anyhow!(
            "failed query max ts with SQL `{}`, result.value is empty",
            &query
        ))?
        .first()
        .ok_or(anyhow::anyhow!(
            "failed query max ts with SQL `{}`, 0 rows returned",
            &query
        ))?
        .parse::<u64>()?;
    Ok(ts_sec)
}

fn scuba_result_to_message_count(result: SQLQueryResult) -> Result<HashMap<String, usize>> {
    let mut message_count = HashMap::new();
    for entry in result.value.iter() {
        let actor_id;
        let count;
        if let Some(actor_id_str) = entry.first() {
            // TODO: There are actor IDs like "world[0].world[0].ping[0][0]" which cannot be parsed
            actor_id = actor_id_str;
        } else {
            tracing::error!("failed to get actor_id from scuba result entry {:?}", entry);
            continue;
        }
        if let Some(count_str) = entry.last() {
            count = count_str.parse();
        } else {
            tracing::error!("failed to get counter from scuba result entry {:?}", entry);
            continue;
        }
        if let Ok(count) = count {
            message_count.insert(actor_id.to_string(), count);
        } else {
            tracing::error!(
                "failed to parse counter and actor_id from scuba result entry {:?}",
                entry
            );
            continue;
        }
    }
    Ok(message_count)
}
