use anyhow::anyhow;
use clap::Parser;
use clap::value_parser;

use crate::args::ServerHandle;
use crate::commands::Error;
use crate::commands::Runnable;

#[derive(Parser, Debug)]
pub struct Cmd {
    #[arg(index=1, value_parser=value_parser!(ServerHandle))]
    pub server: ServerHandle,
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
        let cli = Cli::parse_from(["monarch", "stop", "k8s://ns/jobid"]);

        let subcmd = cli.subcmd;
        let result = subcmd.run();
        assert_eq!(result.unwrap_err().to_string(), "TODO implement");
    }
}
