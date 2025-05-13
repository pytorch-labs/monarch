/// This is an "empty shell" bootstrap process,
/// simply invoking [`hyperactor_mesh::bootstrap_or_die`].
#[tokio::main]
async fn main() {
    hyperactor_mesh::bootstrap_or_die().await;
}
