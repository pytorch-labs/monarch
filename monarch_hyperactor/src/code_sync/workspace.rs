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
    Constant(PathBuf),
    FromEnvVar(String),
}

impl WorkspaceLocation {
    pub fn resolve(&self) -> Result<PathBuf> {
        Ok(match self {
            WorkspaceLocation::Constant(p) => p.clone(),
            WorkspaceLocation::FromEnvVar(v) => PathBuf::from(
                std::env::var_os(v).with_context(|| format!("workspace env var not set: {}", v))?,
            ),
        })
    }
}
