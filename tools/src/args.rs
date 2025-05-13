use std::fmt::Display;
use std::fmt::Formatter;
use std::str::FromStr;

use anyhow::anyhow;
use regex::Regex;

const SERVER_HANDLE_REGEX: &str = r"(?<scheduler>.+)://(?<namespace>.*)/(?<id>.+)";

#[derive(Clone, Debug)]
pub struct ServerHandle {
    pub scheduler: String,
    pub namespace: Option<String>,
    pub id: String,
}

/// Uniquely identifies a running Monarch server (backend).
///
/// A server handle is of the form `scheduler://namespace/id``.
/// Where the `scheduler` identifies where it is running,
/// the `namespace` is interpreted by the scheduler to identify the
/// partition (e.g. region, entitlement, etc), and the `id`
/// is usually the job id/name.
///
/// The `namespace` can be empty (in which case it defaults to `None`)
/// if the scheduler does not support such concept.
impl ServerHandle {
    pub fn new(scheduler: &str, namespace: Option<&str>, id: &str) -> Self {
        Self {
            scheduler: scheduler.to_owned(),
            namespace: namespace.map(str::to_owned),
            id: id.to_owned(),
        }
    }

    /// Parses a server handle from its URI (str) form to the `ServerHandle` struct.
    ///
    /// The URI format is:
    ///   1. `<scheduler://<namespace>/<id>`
    ///   2. -- or -- `<scheduler:///<id>` (no namespace)
    pub fn from_uri(uri: &str) -> anyhow::Result<Self> {
        let re = Regex::new(SERVER_HANDLE_REGEX).unwrap();
        let capture = re
            .captures(uri)
            .ok_or_else(|| anyhow!("{uri} does not match the regex: {SERVER_HANDLE_REGEX}"))?;
        let scheduler = String::from(&capture["scheduler"]);
        let ns = &capture["namespace"];

        let namespace = if ns.is_empty() {
            None
        } else {
            Some(String::from(ns))
        };
        let id = String::from(&capture["id"]);
        Ok(ServerHandle {
            scheduler,
            namespace,
            id,
        })
    }
}

impl FromStr for ServerHandle {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Self> {
        ServerHandle::from_uri(s)
    }
}

impl Display for ServerHandle {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let ns = self.namespace.to_owned().unwrap_or("".into());
        write!(f, "{}://{}/{}", self.scheduler, ns, self.id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_handle_with_namespace() {
        let handle = ServerHandle::from_uri("k8s://foo/bar").unwrap();
        assert_eq!(handle.scheduler, "k8s");
        assert_eq!(handle.namespace.unwrap(), "foo");
        assert_eq!(handle.id, "bar");
    }

    #[test]
    fn test_server_handle_no_namespace() {
        let handle = ServerHandle::from_uri("k8s:///test_job_name").unwrap();
        assert_eq!(handle.scheduler, "k8s");
        assert_eq!(handle.namespace, None);
        assert_eq!(handle.id, "test_job_name");
    }

    #[test]
    fn test_server_handle_underscore_in_scheduler() {
        let handle = ServerHandle::from_uri("mast_conda:///bar").unwrap();
        assert_eq!(handle.scheduler, "mast_conda");
        assert_eq!(handle.namespace, None);
        assert_eq!(handle.id, "bar");
    }

    #[test]
    fn test_server_handle_id_as_namespace() {
        // id in place of namespace
        let uri = "k8s://jobid";
        assert!(ServerHandle::from_uri(uri).is_err());
    }

    #[test]
    fn test_server_handle_missing_id() {
        // missing id
        let uri = "k8s://namespace/";
        assert!(ServerHandle::from_uri(uri).is_err());
    }

    #[test]
    fn test_server_handle_invalid_uri() {
        let uri = "k8s//foo/bar";
        assert!(ServerHandle::from_uri(uri).is_err());
    }

    #[test]
    fn test_server_handle_to_string() {
        let uri = "k8s://foo/bar";
        let handle = ServerHandle::from_str(uri).unwrap();
        assert_eq!(uri, handle.to_string());
    }
}
