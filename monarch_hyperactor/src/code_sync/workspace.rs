/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use serde::Deserialize;
use serde::Serialize;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum WorkspaceLocation {
    //// Workspace directory specified by the given path.
    Constant(PathBuf),

    /// Workspace directory specified by dereferencing the value of the environment variable
    /// and appending the relative path to it.
    ///
    /// Example: `WorkspaceLocation::FromEnvVar{ env:"WORKSPACE_DIR", relpath: PathBuf::from("github/torchtitan) }`
    /// points to `$WORKSPACE_DIR/github/torchtitan`.
    FromEnvVar {
        env: String,
        relpath: PathBuf,
    },
}

impl WorkspaceLocation {
    pub fn resolve(&self) -> Result<PathBuf> {
        Ok(match self {
            WorkspaceLocation::Constant(p) => p.clone(),
            WorkspaceLocation::FromEnvVar { env, relpath } => PathBuf::from(
                std::env::var_os(env)
                    .with_context(|| format!("workspace env var not set: {}", env))?,
            )
            .join(relpath),
        })
    }
}
#[cfg(test)]
mod tests {
    use std::env;

    use tempfile::tempdir;

    use super::*;

    #[test]
    fn test_constant_workspace_location_constant() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_path_buf();
        let loc = WorkspaceLocation::Constant(path.clone());
        let resolved = loc.resolve().unwrap();
        assert_eq!(resolved, path);
    }

    #[test]
    fn test_from_env_var_workspace_location() {
        let tmpdir = tempdir().unwrap();

        // SAFETY: ok for single threaded test case
        unsafe { env::set_var("WORKSPACE_DIR", tmpdir.path()) }

        assert_eq!(
            tmpdir.path().join("github/torchtitan"),
            WorkspaceLocation::FromEnvVar {
                env: "WORKSPACE_DIR".to_string(),
                relpath: PathBuf::from("github/torchtitan")
            }
            .resolve()
            .unwrap(),
        );

        assert_eq!(
            tmpdir.path().to_path_buf(),
            WorkspaceLocation::FromEnvVar {
                env: "WORKSPACE_DIR".to_string(),
                relpath: PathBuf::new()
            }
            .resolve()
            .unwrap(),
        );

        // SAFETY: ok for single threaded test case
        unsafe { env::remove_var("WORKSPACE_DIR") }
    }

    #[test]
    fn test_from_env_var_missing_env() {
        let loc = WorkspaceLocation::FromEnvVar {
            env: "__NON_EXISTENT__".to_string(),
            relpath: PathBuf::from("foo"),
        };
        let err = loc.resolve().unwrap_err();
        assert!(format!("{:?}", err).contains("__NON_EXISTENT__"));
    }
}
