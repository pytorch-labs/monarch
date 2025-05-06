use std::collections::HashMap;
use std::io;
use std::io::Write;

use clap::Subcommand;
use console::style;
#[cfg(not(fbcode_build))]
use hyper::utils::system_address::SystemAddr;
use hyperactor::ActorRef;
use hyperactor::reference::Reference;
use hyperactor_multiprocess::proc_actor::ProcActor;
use hyperactor_multiprocess::proc_actor::ProcMessageClient;
use hyperactor_multiprocess::proc_actor::PySpyConfig;
use hyperactor_multiprocess::proc_actor::StackTrace;
use hyperactor_multiprocess::pyspy::PySpyStackTrace;
use hyperactor_multiprocess::system_actor::SYSTEM_ACTOR_REF;
use hyperactor_multiprocess::system_actor::SystemMessageClient;
use hyperactor_multiprocess::system_actor::SystemSnapshot;
use hyperactor_multiprocess::system_actor::SystemSnapshotFilter;
use tabwriter::TabWriter;
#[cfg(fbcode_build)]
use utils::system_address::SystemAddr;

#[derive(Subcommand)]
pub enum ProcsCommand {
    /// Show current state of procs in the system.
    Show(ShowCommand),
    /// Perform a py-spy dump and return stack traces.
    PySpy(PySpyCommand),
}

#[derive(clap::Args, Debug)]
pub struct ShowCommand {
    /// The address of the system. Can be a channel address or a MAST job name.
    system_addr: SystemAddr,

    /// Labels to match on the world, key=value,...
    #[arg(long)]
    world_labels: Option<Vec<String>>,

    /// Labels to match on procs, key=value,...
    #[arg(long)]
    proc_labels: Option<Vec<String>>,
}

#[derive(clap::Args, Debug)]
pub struct PySpyCommand {
    /// The address of the system. Can be a channel address or a MAST job name.
    system_addr: SystemAddr,

    /// Also produce native stack frames. Requires non_blocking to be false.
    #[arg(long, default_value_t = false)]
    native: bool,
    /// Also produce native stack frames for threads created by native code.
    /// Implies native. Requires non_blocking to be false.
    #[arg(long, default_value_t = false)]
    native_all: bool,
    /// Do not suspend process while dumping.
    #[arg(long, default_value_t = false)]
    non_blocking: bool,
    /// Include thread activity information
    #[arg(long, default_value_t = true)]
    include_activity: bool,

    /// The proc reference to dump.
    proc_ref: Reference,
}

impl ProcsCommand {
    pub async fn run(self) -> anyhow::Result<()> {
        match self {
            ProcsCommand::Show(args) => args.run().await,
            ProcsCommand::PySpy(args) => args.run().await,
        }
    }
}

impl ShowCommand {
    pub async fn run(self) -> anyhow::Result<()> {
        let mut system = hyperactor_multiprocess::System::new(self.system_addr.clone().into());
        let client = system.attach().await.expect("failed to attach to system");

        let mut filter = SystemSnapshotFilter::all();
        if let Some(ref world_labels) = self.world_labels {
            let labels = Self::parse_labels(world_labels)?;
            filter.world_labels = labels;
        }
        if let Some(ref proc_labels) = self.proc_labels {
            let labels = Self::parse_labels(proc_labels)?;
            filter.proc_labels = labels;
        }

        let snapshot = SYSTEM_ACTOR_REF
            .snapshot(&client, filter)
            .await
            .expect("failed to snapshot system");
        self.print_snapshot(snapshot)?;

        Ok(())
    }

    fn print_snapshot(&self, snapshot: SystemSnapshot) -> anyhow::Result<()> {
        let mut tw = TabWriter::new(io::stdout());

        let mut worlds = snapshot.worlds.keys().collect::<Vec<_>>();
        worlds.sort();
        for world in worlds {
            let world_snapshot = &snapshot.worlds[world];
            write!(tw, "world: {}\n", world.name())?;
            let mut procs: Vec<_> = world_snapshot.procs.iter().collect();
            procs.sort_by_key(|e| e.0);
            for (proc_id, proc_info) in procs {
                write!(tw, "\t{}", proc_id)?;
                if !proc_info.labels.is_empty() {
                    write!(tw, ":\n")?;
                    write!(tw, "\t\tlabels:\n")?;
                    for (name, value) in &proc_info.labels {
                        write!(tw, "\t\t\t{}={}\n", name, value)?;
                    }
                } else {
                    write!(tw, "\n")?;
                }
            }
        }
        tw.flush()?;
        Ok(())
    }

    fn parse_labels(labels: &[String]) -> anyhow::Result<HashMap<String, String>> {
        let mut result = HashMap::new();
        for label in labels {
            let labels = label.split(',').collect::<Vec<_>>();
            for label in labels {
                let parts: Vec<&str> = label.splitn(2, '=').collect();
                if parts.len() != 2 {
                    anyhow::bail!("invalid label format: {}", label);
                }
                result.insert(parts[0].trim().to_string(), parts[1].trim().to_string());
            }
        }
        Ok(result)
    }
}

impl PySpyCommand {
    pub async fn run(self) -> anyhow::Result<()> {
        let pyspy_config = if self.native || self.native_all {
            if self.non_blocking {
                anyhow::bail!("native requires non_blocking to be false");
            }
            PySpyConfig::Blocking {
                native: Some(self.native),
                native_all: Some(self.native_all),
            }
        } else {
            PySpyConfig::NonBlocking
        };
        let mut system = hyperactor_multiprocess::System::new(self.system_addr.clone().into());
        let client = system.attach().await.expect("failed to attach to system");

        match self.proc_ref {
            Reference::Proc(proc_id) => {
                // TODO: we should really look up the proc actor from the system
                let proc_ref: ActorRef<ProcActor> = ActorRef::attest(proc_id.actor_id("proc", 0));
                let traces = proc_ref.py_spy_dump(&client, pyspy_config).await?;
                Self::print_traces(&traces, self.include_activity);
                Ok(())
            }
            _ => {
                anyhow::bail!("unsupported reference type");
            }
        }
    }

    fn print_traces(traces: &StackTrace, include_activity: bool) {
        println!(
            "Process {}: {}",
            style(traces.trace.pid).bold().yellow(),
            traces.trace.command_line
        );
        if let Some(ref traces) = traces.trace.stack_traces {
            for trace in traces.iter().rev() {
                Self::print_trace(trace, include_activity);
                println!();
            }
        } else if let Some(ref err) = traces.trace.error {
            println!("{} {}", style("Error:").red(), err);
        } else {
            println!("{} No stack traces found", style("Error:").red());
        }
    }

    fn print_trace(trace: &PySpyStackTrace, include_activity: bool) {
        let thread_id = match trace.os_thread_id {
            Some(tid) => format!("{}", tid),
            None => format!("{:#X}", trace.thread_id),
        };
        let status_str = match (trace.owns_gil, trace.active) {
            (_, false) => "idle",
            (true, true) => "active+gil",
            (false, true) => "active",
        };
        let status = if include_activity {
            format!(" ({})", status_str)
        } else if trace.owns_gil {
            " (gil)".to_owned()
        } else {
            "".to_owned()
        };

        match trace.thread_name.as_ref() {
            Some(name) => {
                println!(
                    "Thread {}{}: \"{}\"",
                    style(thread_id).bold().yellow(),
                    status,
                    name
                );
            }
            None => {
                println!("Thread {}{}", style(thread_id).bold().yellow(), status);
            }
        };

        for frame in &trace.frames {
            let filename = match &frame.short_filename {
                Some(f) => f,
                None => &frame.filename,
            };
            if frame.line != 0 {
                println!(
                    "    {} ({}:{})",
                    style(&frame.name).green(),
                    style(&filename).cyan(),
                    style(frame.line).dim()
                );
            } else {
                println!(
                    "    {} ({})",
                    style(&frame.name).green(),
                    style(&filename).cyan()
                );
            }

            if let Some(locals) = &frame.locals {
                let mut shown_args = false;
                let mut shown_locals = false;
                for local in locals {
                    if local.arg && !shown_args {
                        println!("        {}", style("Arguments:").dim());
                        shown_args = true;
                    } else if !local.arg && !shown_locals {
                        println!("        {}", style("Locals:").dim());
                        shown_locals = true;
                    }

                    let repr = local.repr.as_deref().unwrap_or("?");
                    println!("            {}: {}", local.name, repr);
                }
            }
        }
    }
}
