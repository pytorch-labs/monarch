mod commands;

use clap::Parser;
use clap::Subcommand;

use crate::commands::demo::DemoCommand;
use crate::commands::procs::ProcsCommand;
use crate::commands::serve::ServeCommand;
use crate::commands::show::ShowCommand;
#[cfg(fbcode_build)]
use crate::commands::top::TopCommand;

#[derive(Parser)]
#[command()]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Spawns and serve a system actor.
    Serve(ServeCommand),
    #[command(subcommand)]
    /// Demo some basic concepts of multiprocess hyperactor. Before using these
    /// commands, use `serve` to start a system actor, and get the system
    /// address from the output.
    Demo(DemoCommand),
    #[clap(about = r#"Show the state of a reference. For example:
    - System: show <system_address>
    - World:  show <system_address> world
    - Gang:   show <system_address> world.gang
    - Proc:   show <system_address> world[2]
    - Actor:  show <system_address> world[3].actor[1]"#)]
    Show(ShowCommand),
    #[clap(about = "Show details about processes running in worlds.")]
    #[command(subcommand)]
    Procs(ProcsCommand),
    #[cfg(fbcode_build)]
    #[clap(about = "Show a dynamic real-time view of the system")]
    Top(TopCommand),
}

#[cfg(fbcode_build)]
#[fbinit::main]
async fn main(_: fbinit::FacebookInit) -> Result<(), anyhow::Error> {
    run().await
}

#[cfg(not(fbcode_build))]
#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    run().await
}

async fn run() -> Result<(), anyhow::Error> {
    let args = Cli::parse();
    hyperactor::initialize();

    match args.command {
        Command::Serve(command) => Ok(command.run().await?),
        Command::Demo(command) => Ok(command.run().await?),
        Command::Show(command) => Ok(command.run().await?),
        Command::Procs(command) => Ok(command.run().await?),
        Command::Top(command) => Ok(command.run().await?),
    }
}
