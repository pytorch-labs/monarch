use anyhow::Result;
use fbinit::FacebookInit;
use monarch_tools::Cli;
use monarch_tools::commands::Runnable;

#[cli::main(
    "monarch",
    fastcli_fallback(
        path = "py_monarch",
        resource_path = "monarch/tools/py_monarch",
        metadata_symbol_target = "cli-fallback-metadata",
    )
)]
fn main(_fb: FacebookInit, cli: Cli) -> Result<cli::ExitCode> {
    match cli.subcmd.run() {
        Ok(()) => Ok(cli::ExitCode::SUCCESS),
        Err(e) => {
            eprintln!("Error running subcommand: {:?} ", e);
            Ok(cli::ExitCode::FAILURE)
        }
    }
}
