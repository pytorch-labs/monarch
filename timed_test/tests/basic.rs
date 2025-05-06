use timed_test::async_timed_test;

#[async_timed_test(timeout_secs = 5)]
async fn good() {
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
}

#[async_timed_test(timeout_secs = 1)]
#[should_panic]
async fn bad() {
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
}
