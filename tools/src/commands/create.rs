use anyhow::Error;
use anyhow::anyhow;
use clap::Parser;

use crate::commands::Runnable;

#[derive(Parser, Debug)]
pub struct Cmd {
    /// Host type to use for the worker hosts (must be a TorchX named resource)
    #[arg(short = 't', long)]
    host_type: String,
    /// Number of worker host
    #[arg(short = 'n', long, default_value_t = 1)]
    num_hosts: u32,
    // Scheduler to submit to
    #[arg(short = 's', long)]
    scheduler: String,
}

impl Runnable for Cmd {
    fn run(&self) -> Result<(), Error> {
        Err(anyhow!("TODO implement"))
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use crate::Cli;
    use crate::commands::Runnable;

    #[test]
    fn test_run() {
        let cli = Cli::parse_from([
            "monarch",
            "create",
            "-t",
            "gtt_any",
            "-n",
            "4",
            "-s",
            "mast_conda",
        ]);

        let subcmd = cli.subcmd;
        let result = subcmd.run();
        assert_eq!(result.unwrap_err().to_string(), "TODO implement");
    }
}
