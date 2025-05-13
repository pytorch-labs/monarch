//! A binary to launch the simulated Monarch controller along with necessary environment.
use std::process::ExitCode;

use anyhow::Result;
use clap::Parser;
use hyperactor::channel::ChannelAddr;
use monarch_simulator_lib::bootstrap::boostrap;

#[derive(Debug, Parser)]
struct Args {
    #[arg(short, long)]
    system_addr: ChannelAddr,
}

const TITLE: &str = r#"
******************************************************
*                                                    *
* ____ ___ __  __ _   _ _        _  _____ ___  ____  *
*/ ___|_ _|  \/  | | | | |      / \|_   _/ _ \|  _ \ *
*\___ \| || |\/| | | | | |     / _ \ | || | | | |_) |*
* ___) | || |  | | |_| | |___ / ___ \| || |_| |  _ < *
*|____/___|_|  |_|\___/|_____/_/   \_\_| \___/|_| \_\*
*                                                    *
******************************************************
"#;

#[tokio::main]
async fn main() -> Result<ExitCode> {
    eprintln!("{}", TITLE);
    hyperactor::initialize();
    let args = Args::parse();

    let system_addr = args.system_addr.clone();
    tracing::info!("starting Monarch simulation");

    let operational_listener_handle = boostrap(system_addr).await?;

    operational_listener_handle
        .await
        .expect("simulator exited unexpectedly");

    Ok(ExitCode::SUCCESS)
}
