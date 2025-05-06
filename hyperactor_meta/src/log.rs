use std::collections::VecDeque;
use std::fmt::Debug;
use std::ops::Range;

use async_trait::async_trait;
use cpentity_client::CPEntityClient;
use cpentity_client::EntityVersionPair;
use cpentity_common::CPEntityId;
use cpentity_service::CPEntityGetRequest;
use cpentity_service::CPEntityRemoveRequest;
use cpentity_service::CPEntityRequestContext;
use cpentity_service::PutStatus;
use fbinit::FacebookInit;
use futures::StreamExt;
use futures::pin_mut;
use futures::stream;
use futures::stream::BoxStream;
use hyperactor::RemoteMessage;
use hyperactor::mailbox::log::MessageLog;
use hyperactor::mailbox::log::MessageLogError;
use hyperactor::mailbox::log::SeqId;
use log::MessageData;
use standard::StandardProtocol;

// CPEntity related parameters.
pub static CPENTITY_TENANT: &str = "cpentity";
pub static CPENTITY_ENVIRONMENT: &str = "shared_test_env_1";
pub static CPENTITY_TIER: &str = "cpentity_service.multitenantcluster.bbbf.test";

/// [`CPEntityLog`] is an CPEntity-based implementation of [`MessageLog`]. CPEntity
/// (https://fburl.com/cpentity) is backed by Delos. It supports durable, consistent,
/// and scalable control plane entity CRUD operations.
///
/// Two major issues with the current implementation: latency and richness of the Rust
/// client. The current CPEntity Rust client only supports single-entity CRUD operations.
/// Each CRUD operation takes a few hundred milliseconds.
pub struct CPEntityLog<M: RemoteMessage> {
    client: CPEntityClient,
    group_id: String,        // CPEntity group to uniquely identify a CPEntity shard
    queue: VecDeque<M>,      // messages to be persisted
    log_range: Range<SeqId>, // range of sequence ids that have been persisted
}

const CLIENT_NAME: &str = "hyperactor";
const APPEND_MESSAGE_LIMIT: usize = 10;

impl<M: RemoteMessage> CPEntityLog<M> {
    /// Create a new [`CPEntityLog`] instance. `group_id` is to uniquely identify a
    /// CPEntity shard. `tenant`, `environment`, and `cpentity_service_tier` are necessary
    /// info to create a CPEntity client (https://www.internalfb.com/intern/wiki/CPEntity/Quickstart/).
    /// `log_range` indicates the starting (inclusive) and ending (exclusive) sequence
    /// ids that are available to read.
    ///
    /// We have a test tier till Nov 2024. Use the following fields to access the tier.
    /// ```ignore
    /// tenant: String::from("cpentity"),
    /// environment: String::from("shared_test_env_1"),
    /// cpentity_service_tier: Some(String::from("cpentity_service.multitenantcluster.bbbf.test")),
    /// ```
    /// Only monarch oncall has access to this tier.
    pub fn new(
        fb: FacebookInit,
        group_id: String,
        tenant: String,
        environment: String,
        cpentity_service_tier: Option<String>,
        log_range: Range<SeqId>,
    ) -> Result<Self, MessageLogError> {
        let client = CPEntityClient::new(
            fb,
            String::from(CLIENT_NAME),
            tenant,
            environment,
            cpentity_service_tier,
            StandardProtocol::Compact,
        )
        .map_err(MessageLogError::Other)?;

        Self::with_client(group_id, client, log_range)
    }

    fn with_client(
        group_id: String,
        client: CPEntityClient,
        log_range: Range<SeqId>,
    ) -> Result<Self, MessageLogError> {
        Ok(Self {
            client,
            group_id,
            queue: VecDeque::with_capacity(APPEND_MESSAGE_LIMIT),
            log_range,
        })
    }

    async fn flush_single_message(&self, message: &M) -> Result<(), MessageLogError> {
        let seq_id = self.log_range.end;
        let data = bincode::serialize(&message)
            .map_err(|err| MessageLogError::Append(seq_id, err.into()))?;
        let message_data = MessageData {
            data,
            ..Default::default()
        };

        let entity_id = CPEntityId {
            groupId: self.group_id.to_owned(),
            localId: seq_id.to_string(),
            ..Default::default()
        };
        let request_context = CPEntityRequestContext {
            requestId: seq_id.to_string(),
            ..Default::default()
        };

        // There are cases where we have flushed a sequence id but log_range is not updated before crash.
        // Allow UPDATED status to handle such already flushed cases.
        let request = self.client.make_cpentity_put_request(
            &entity_id,
            &message_data,
            &request_context,
            cpentity_service::PutMethod::UPSERT,
            None,
        );

        let response = self
            .client
            .put_entity(&request)
            .await
            .map_err(|err| MessageLogError::Flush(seq_id, seq_id, err.into()))?;

        if response.status != PutStatus::INSERTED && response.status != PutStatus::UPDATED {
            return Err(MessageLogError::Flush(
                seq_id,
                seq_id,
                anyhow::anyhow!(
                    "failed to flush message at sequence id {} with status {}",
                    seq_id,
                    response.status
                ),
            ));
        }
        Ok(())
    }
}

impl<M: RemoteMessage> Debug for CPEntityLog<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CPEntityLog")
            .field("group_id", &self.group_id)
            .field("queue", &self.queue)
            .field("log_range", &self.log_range)
            .finish()
    }
}

impl<M: RemoteMessage> Drop for CPEntityLog<M> {
    fn drop(&mut self) {
        if !self.queue.is_empty() {
            tracing::error!(
                "{}: found {} unflushed messages",
                self.group_id,
                self.queue.len()
            );
        }
    }
}

#[async_trait]
impl<M: RemoteMessage> MessageLog<M> for CPEntityLog<M> {
    type Stream<'a> = BoxStream<'a, Result<(SeqId, M), MessageLogError>>;

    /// Append a put request with its serialized message to the queue.
    async fn append(&mut self, message: M) -> Result<(), MessageLogError> {
        if self.queue.len() >= APPEND_MESSAGE_LIMIT {
            self.flush().await?;
        }
        self.queue.push_back(message);

        Ok(())
    }

    /// Flush all the put requests to CPEntity in the queue.
    async fn flush(&mut self) -> Result<SeqId, MessageLogError> {
        while !self.queue.is_empty() {
            let message = self.queue.front().unwrap();

            self.flush_single_message(message).await?;

            self.queue.pop_front();
            self.log_range.end += 1;
        }
        Ok(self.log_range.end)
    }

    async fn append_and_flush(&mut self, message: &M) -> Result<SeqId, MessageLogError> {
        self.flush().await?;
        self.flush_single_message(message).await?;
        self.log_range.end += 1;
        Ok(self.log_range.end)
    }

    /// Remove persisted CPEntity messages permanently up to (non-inclusive) `new_start`.
    async fn trim(&mut self, new_start: SeqId) -> Result<(), MessageLogError> {
        for seq_id in self.log_range.start..new_start {
            let entity_id = CPEntityId {
                groupId: self.group_id.to_owned(),
                localId: seq_id.to_string(),
                ..Default::default()
            };
            let request_context = CPEntityRequestContext {
                requestId: seq_id.to_string(),
                ..Default::default()
            };
            let remove_request = CPEntityRemoveRequest {
                entityId: entity_id,
                requestContext: request_context,
                ..Default::default()
            };

            match self.client.remove_entity(&remove_request).await {
                Ok(_) => {
                    // No need to check the response. If the entity does not exist, just ignore it.
                }
                Err(err) => {
                    // No need to worry about errors. CPEntity has retention to clean up messages.
                    tracing::error!("error trimming message at sequence id {}: {}", seq_id, err);
                }
            }
            self.log_range.start = seq_id;
        }
        Ok(())
    }

    /// Read all messages from a given sequence id from CPEntity.
    async fn read(&self, from: SeqId) -> Result<Self::Stream<'_>, MessageLogError> {
        if !self.log_range.contains(&from) {
            return Err(MessageLogError::Read(
                from,
                anyhow::anyhow!(
                    "the current range {:?} does not contain the given sequence id",
                    self.log_range
                ),
            ));
        }

        Ok(stream::try_unfold(from, move |seq_it_to_read| async move {
            if seq_it_to_read >= self.log_range.end {
                return Ok(None);
            }
            let entity_id = CPEntityId {
                groupId: self.group_id.to_owned(),
                localId: seq_it_to_read.to_string(),
                ..Default::default()
            };
            let request_context = CPEntityRequestContext {
                requestId: seq_it_to_read.to_string(),
                ..Default::default()
            };
            let get_request = CPEntityGetRequest {
                entityId: entity_id,
                requestContext: request_context,
                ..Default::default()
            };

            // TODO: CPEntity client does support streaming API but only in C++. The Rust API is to be implemented.
            match self.client.get_entity::<MessageData>(&get_request).await {
                Ok(Some(EntityVersionPair { entity, .. })) => {
                    match bincode::deserialize::<M>(&entity.data) {
                        Ok(message) => Ok(Some(((seq_it_to_read, message), seq_it_to_read + 1))),
                        Err(err) => Err(MessageLogError::Read(seq_it_to_read, err.into())),
                    }
                }
                Ok(None) => Err(MessageLogError::Read(
                    seq_it_to_read,
                    anyhow::anyhow!("found no message"),
                )),
                Err(err) => Err(MessageLogError::Read(seq_it_to_read, err.into())),
            }
        })
        .boxed())
    }

    /// Read exactly one message from CPEntity.
    async fn read_one(&self, seq_id: SeqId) -> Result<M, MessageLogError> {
        let it = self.read(seq_id).await?;

        pin_mut!(it);
        match it.next().await {
            Some(Ok((result_seq_id, message))) => {
                if result_seq_id != seq_id {
                    return Err(MessageLogError::Read(
                        seq_id,
                        anyhow::anyhow!("failed to find message with sequence {}", seq_id),
                    ));
                }
                return Ok(message);
            }
            Some(Err(err)) => {
                return Err(err);
            }
            None => {
                return Err(MessageLogError::Read(
                    seq_id,
                    anyhow::anyhow!("failed to find message with sequence {}", seq_id),
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use std::assert_matches::assert_matches;

    use cpentity_service::CPEntityClientContext;
    use futures::StreamExt;
    use hyperactor::mailbox::log::MessageLog;
    use hyperactor::mailbox::log::SeqId;
    use once_cell::sync::Lazy;
    use test_utils::InMemoryCPEntityService;

    use super::*;

    const IS_MOCK_CPENTITY: bool = true;
    static IN_MEMORY_CPENTITY_SERVICE: Lazy<InMemoryCPEntityService> =
        Lazy::new(InMemoryCPEntityService::new);

    fn get_client() -> CPEntityClient {
        if IS_MOCK_CPENTITY {
            CPEntityClient::builder()
                .cpentity_service_client(IN_MEMORY_CPENTITY_SERVICE.get_service()
                    as cpentity_service_srclients::CPEntityServiceClient)
                .client_context(CPEntityClientContext::default())
                .cpentity_service_tier("cpentity.test".to_owned())
                .protocol(StandardProtocol::Compact)
                .build()
        } else {
            // The test tier will expire after Nov 2024. It is better not to run this in CI.
            CPEntityClient::new(
                // SAFETY: Called once during module initialization
                unsafe { fbinit::perform_init() },
                "test".to_owned(),
                String::from(CPENTITY_TENANT),
                String::from(CPENTITY_ENVIRONMENT),
                Some(String::from(CPENTITY_TIER)),
                StandardProtocol::Compact,
            )
            .unwrap()
        }
    }

    #[fbinit::test]
    async fn test_cpentity_log() {
        let messages = (APPEND_MESSAGE_LIMIT + 2) as u64;

        // Trim with a new log; make sure we do not have residual messages in CPEntity
        CPEntityLog::<u64>::with_client("test".to_owned(), get_client(), 0..messages)
            .unwrap()
            .trim(messages)
            .await
            .unwrap();

        let mut log =
            CPEntityLog::<u64>::with_client("test".to_owned(), get_client(), 0..0).unwrap();

        // Write some data
        for i in 0..messages {
            log.append(42u64 + i).await.unwrap();
        }

        // We should flush at `APPEND_MESSAGE_LIMIT` messages
        let mut it = log.read(0).await.unwrap();
        for i in 0..messages - 2 {
            let (seq, message): (SeqId, u64) = it.next().await.unwrap().unwrap();
            assert_eq!(seq, i);
            assert_eq!(message, 42u64 + i);
        }
        assert!(it.next().await.is_none());
        drop(it);

        // Now flush the last 2 messages
        log.flush().await.unwrap();

        // Normal read
        let mut it = log.read(0).await.unwrap();
        for i in 0..messages {
            let (seq, message): (SeqId, u64) = it.next().await.unwrap().unwrap();
            assert_eq!(seq, i);
            assert_eq!(message, 42u64 + i);
        }
        assert!(it.next().await.is_none());
        drop(it);

        // Read one
        assert_eq!(log.read_one(3).await.unwrap(), 45u64);

        // Read trimmed messages
        log.trim(2).await.unwrap();
        let mut it = log.read(1).await.unwrap();
        assert_eq!(it.next().await.unwrap().unwrap_err().to_string(), "read: 1");
        drop(it);

        // Read from the last untrimmed message
        let mut it = log.read(2).await.unwrap();
        let (seq, message): (SeqId, u64) = it.next().await.unwrap().unwrap();
        assert_eq!(seq, 2);
        assert_eq!(message, 44u64);
        drop(it);

        // Create a new log and test we can read the same messages
        let log2 =
            CPEntityLog::<u64>::with_client("test".to_owned(), get_client(), 2..messages).unwrap();
        let mut it = log2.read(2).await.unwrap();
        for i in 2..messages {
            let (seq, message): (SeqId, u64) = it.next().await.unwrap().unwrap();
            assert_eq!(seq, i);
            assert_eq!(message, 42u64 + i);
        }
        assert!(it.next().await.is_none());
        drop(it);

        // Clean up state
        log.trim(messages).await.unwrap();
    }

    #[fbinit::test]
    async fn test_cpentity_log_crash_flush() {
        // Write some data with the first log
        let mut log =
            CPEntityLog::<u64>::with_client("test_crash_flush".to_owned(), get_client(), 0..0)
                .unwrap();
        log.append_and_flush(&5u64).await.unwrap();

        // Suppose the first log is crashed and we again start with [0, 0) writing some different data
        let mut log =
            CPEntityLog::<u64>::with_client("test_crash_flush".to_owned(), get_client(), 0..0)
                .unwrap();
        log.append_and_flush(&6u64).await.unwrap();
        assert_eq!(
            log.read(0).await.unwrap().next().await.unwrap().unwrap(),
            (0, 6u64)
        );
    }

    #[fbinit::test]
    async fn test_cpentity_log_ignore_invalid_trim() {
        // Starting with a wide range without any data flushed
        let mut log = CPEntityLog::<u64>::with_client(
            "test_ignore_invalid_trim".to_owned(),
            get_client(),
            0..10,
        )
        .unwrap();

        // Trim outside range
        log.trim(20).await.unwrap();

        // Trim inside range
        log.trim(5).await.unwrap();
    }

    #[fbinit::test]
    async fn test_cpentity_log_handle_invalid_read() {
        // Starting with a wide range without any data flushed
        let log = CPEntityLog::<u64>::with_client(
            "test_handle_invalid_read".to_owned(),
            get_client(),
            0..10,
        )
        .unwrap();

        // Read outside range
        assert_matches!(
            log.read(20).await.err().unwrap(),
            MessageLogError::Read(20, _,)
        );

        // Read inside range
        assert_matches!(
            log.read(5)
                .await
                .unwrap()
                .next()
                .await
                .unwrap()
                .err()
                .unwrap(),
            MessageLogError::Read(5, _,)
        );
    }
}
