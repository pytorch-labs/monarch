mod args;
pub mod commands;

use clap::Parser;
use commands::Command;

#[derive(Parser, Debug)]
#[command(about = "Command line tool for monarch")]
pub struct Cli {
    #[command(subcommand)]
    pub subcmd: Command,
}
