/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use chrono::DateTime;
use chrono::Utc;
use digest::Digest;
use digest::Output;
use rattler_conda_types::PrefixRecord;
use rattler_conda_types::prefix_record::PathsEntry;
use serde::Deserialize;
use serde::Serialize;
use serde_json;
use sha2::Sha256;
use tokio::fs;
use walkdir::WalkDir;

use crate::hash_utils;
use crate::pack_meta::History;
use crate::pack_meta::Offsets;

/// Fingerprint of the conda-meta directory, used by `CondaFingerprint` below.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CondaMetaFingerprint {
    // TODO(agallagher): It might be worth storing more information of installed
    // packages, so that we could print better error messages when we detect two
    // envs are not equivalent.
    hash: Output<Sha256>,
}

impl CondaMetaFingerprint {
    async fn from_env(path: &Path) -> Result<Self> {
        let mut hasher = Sha256::new();
        hash_utils::hash_directory_tree(&path.join("conda-meta"), &mut hasher).await?;
        Ok(Self {
            hash: hasher.finalize(),
        })
    }
}

/// Fingerprint of the pack-meta directory, used by `CondaFingerprint` below.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PackMetaFingerprint {
    offsets: Output<Sha256>,
    pub history: History,
}

impl PackMetaFingerprint {
    async fn from_env(path: &Path) -> Result<Self> {
        let pack_meta = path.join("pack-meta");

        // Read the fulle history.jsonl file.
        let contents = fs::read_to_string(pack_meta.join("history.jsonl")).await?;
        let history = History::from_contents(&contents)?;

        // Read entire offsets.jsonl file, but avoid hashing the offsets, which can change.
        let mut hasher = Sha256::new();
        let contents = fs::read_to_string(pack_meta.join("offsets.jsonl")).await?;
        let offsets = Offsets::from_contents(&contents)?;
        for ent in offsets.entries {
            let contents = bincode::serialize(&(ent.path, ent.mode, ent.offsets.len()))?;
            hasher.update(contents.len().to_le_bytes());
            hasher.update(&contents);
        }
        let offsets = hasher.finalize();

        Ok(Self { history, offsets })
    }
}

/// A fingerprint of a conda environment, used to detect if two envs are similar enough to
/// facilitate mtime-based conda syncing.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CondaFingerprint {
    pub conda_meta: CondaMetaFingerprint,
    pub pack_meta: PackMetaFingerprint,
}

impl CondaFingerprint {
    pub async fn from_env(path: &Path) -> Result<Self> {
        Ok(Self {
            conda_meta: CondaMetaFingerprint::from_env(path).await?,
            pack_meta: PackMetaFingerprint::from_env(path).await?,
        })
    }

    /// Create a comparator to compare the mtimes of files from two "equivalent" conda envs.
    /// In particular, thie comparator will be aware of spuriuos mtime changes that occurs from
    /// prefix replacement (via `meta-pack`), and will filter them out.
    pub fn mtime_comparator(
        a: &Self,
        b: &Self,
    ) -> Result<Box<dyn Fn(&SystemTime, &SystemTime) -> std::cmp::Ordering + Send + Sync>> {
        let (a_prefix, a_base) = a.pack_meta.history.first()?;
        let (b_prefix, b_base) = b.pack_meta.history.first()?;
        ensure!(a_prefix == b_prefix);

        // NOTE(agallagher): There appears to be some mtime drift on some files after fbpkg creation,
        // so acccount for that here.
        let slop = Duration::from_secs(5 * 60);

        // We load the timestamp from the first history entry, and use this to see if any
        // files have been updated since the env was created.
        let a_base = UNIX_EPOCH + Duration::from_secs(a_base) + slop;
        let b_base = UNIX_EPOCH + Duration::from_secs(b_base) + slop;

        // We also load the last prefix update window for each, as any mtimes from this window
        // should be ignored.
        let a_window = a
            .pack_meta
            .history
            .prefix_and_last_update_window()?
            .1
            .map(|(s, e)| {
                (
                    UNIX_EPOCH + Duration::from_secs(s),
                    UNIX_EPOCH + Duration::from_secs(e + 1),
                )
            });
        let b_window = b
            .pack_meta
            .history
            .prefix_and_last_update_window()?
            .1
            .map(|(s, e)| {
                (
                    UNIX_EPOCH + Duration::from_secs(s),
                    UNIX_EPOCH + Duration::from_secs(e + 1),
                )
            });

        Ok(Box::new(move |a: &SystemTime, b: &SystemTime| {
            match (
                *a > a_base && a_window.is_none_or(|(s, e)| *a < s || *a > e),
                *b > b_base && b_window.is_none_or(|(s, e)| *b < s || *b > e),
            ) {
                (true, false) => std::cmp::Ordering::Greater,
                (false, true) => std::cmp::Ordering::Less,
                (false, false) => std::cmp::Ordering::Equal,
                (true, true) => a.cmp(b),
            }
        }))
    }
}
