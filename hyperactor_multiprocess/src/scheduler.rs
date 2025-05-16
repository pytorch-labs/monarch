use async_trait::async_trait;

/// TODO: add missing doc
#[async_trait]
pub trait Scheduler {
    /// TODO: add missing doc
    type GangHandle;
    /// TODO: add missing doc
    async fn schedule_gang(&self, size: u64) -> Result<Self::GangHandle, anyhow::Error>;
}

/// TODO: add missing doc
pub struct UnimplementedScheduler;

#[async_trait]
impl Scheduler for UnimplementedScheduler {
    type GangHandle = !;

    async fn schedule_gang(&self, _size: u64) -> Result<Self::GangHandle, anyhow::Error> {
        unimplemented!()
    }
}
