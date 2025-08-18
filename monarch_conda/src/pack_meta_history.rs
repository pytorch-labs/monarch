/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use rattler_conda_types::package::FileMode;
use serde::Deserialize;
use serde::Serialize;
use tokio::fs;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Offset {
    pub start: usize,
    pub len: usize,
    pub contents: Option<Vec<(usize, usize)>>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub struct OffsetRecord {
    pub path: PathBuf,
    pub mode: FileMode,
    pub offsets: Vec<Offset>,
}

impl OffsetRecord {
    fn to_str(&self) -> Result<String> {
        Ok(serde_json::to_string(&(
            &self.path,
            self.mode,
            &self
                .offsets
                .iter()
                .map(|o| {
                    (
                        o.start,
                        o.len,
                        o.contents.as_ref().map(|c| {
                            c.iter()
                                .map(|(a, b)| (a, b, None::<()>))
                                .collect::<Vec<_>>()
                        }),
                    )
                })
                .collect::<Vec<_>>(),
        ))?)
    }

    fn from_str(str: &str) -> Result<Self> {
        let (path, mode, offsets): (_, _, Vec<(usize, usize, Option<Vec<(usize, usize, ())>>)>) =
            serde_json::from_str(str).with_context(|| format!("parsing: {}", str))?;
        Ok(OffsetRecord {
            path,
            mode,
            offsets: offsets
                .into_iter()
                .map(|(start, len, contents)| Offset {
                    start,
                    len,
                    contents: contents.map(|c| c.into_iter().map(|(a, b, _)| (a, b)).collect()),
                })
                .collect(),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Offsets {
    pub entries: Vec<OffsetRecord>,
}

impl Offsets {
    pub fn from_contents(s: &str) -> Result<Self> {
        let mut entries = Vec::new();
        for line in s.lines() {
            entries.push(OffsetRecord::from_str(line)?);
        }
        Ok(Offsets { entries })
    }

    pub async fn from_env(env: &Path) -> Result<Self> {
        let path = env.join("pack-meta").join("offsets.jsonl");
        let s = fs::read_to_string(&path).await?;
        Self::from_contents(&s)
    }

    pub fn to_str(&self) -> Result<String> {
        let mut str = String::new();
        for entry in &self.entries {
            str += &entry.to_str()?;
            str += "\n";
        }
        Ok(str)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub struct HistoryRecord {
    pub timestamp: u64, // timestamps are truncated into seconds
    pub prefix: PathBuf,
    pub finished: bool,
}

impl HistoryRecord {
    fn to_str(&self) -> Result<String> {
        Ok(serde_json::to_string(&(
            self.timestamp,
            &self.prefix,
            self.finished,
        ))?)
    }

    fn from_str(line: &str) -> Result<Self> {
        let (timestamp, prefix, finished) = serde_json::from_str(line)?;
        Ok(HistoryRecord {
            timestamp,
            prefix,
            finished,
        })
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub struct History {
    pub entries: Vec<HistoryRecord>,
}

impl History {
    pub fn from_contents(s: &str) -> Result<Self> {
        let mut entries = Vec::new();
        for line in s.lines() {
            entries.push(HistoryRecord::from_str(line)?);
        }
        Ok(History { entries })
    }

    pub async fn from_env(env: &Path) -> Result<Self> {
        let path = env.join("pack-meta").join("history.jsonl");
        let s = fs::read_to_string(&path).await?;
        Self::from_contents(&s)
    }

    pub fn to_str(&self) -> Result<String> {
        let mut str = String::new();
        for entry in &self.entries {
            str += &entry.to_str()?;
            str += "\n";
        }
        Ok(str)
    }

    pub fn first(&self) -> Result<(&Path, u64)> {
        let first = self.entries.first().context("missing history")?;
        ensure!(first.finished);
        Ok((&first.prefix, first.timestamp))
    }

    pub fn last_prefix_update(&self) -> Result<Option<(&Path, u64, u64)>> {
        let last = self.entries.last().context("missing history")?;
        ensure!(last.finished);
        Ok(if let [.., record, _] = &self.entries[..] {
            ensure!(!record.finished);
            ensure!(record.prefix == last.prefix);
            Some((&record.prefix, record.timestamp, last.timestamp))
        } else {
            None
        })
    }

    pub fn last_prefix(&self) -> Result<&Path> {
        if let Some((prefix, _, _)) = self.last_prefix_update()? {
            return Ok(prefix);
        }
        let (prefix, _) = self.first()?;
        Ok(prefix)
    }

    pub fn prefix_and_last_update_window(&self) -> Result<(&Path, Option<(u64, u64)>)> {
        let src_first = self.first()?;
        Ok(if let Some((prefix, s, e)) = self.last_prefix_update()? {
            (prefix, Some((s, e)))
        } else {
            (src_first.0, None)
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn test_offset_record_parsing() {
        let json = r#"["lib/pkgconfig/pthread-stubs.pc", "text", [[7, 67, null]]]"#;
        let record: OffsetRecord = serde_json::from_str(json).unwrap();

        assert_eq!(record.path, PathBuf::from("lib/pkgconfig/pthread-stubs.pc"));
        assert_eq!(record.mode, FileMode::Text);
        assert_eq!(record.offsets.len(), 1);
        assert_eq!(record.offsets[0].start, 7);
        assert_eq!(record.offsets[0].len, 67);
        assert_eq!(record.offsets[0].contents, None);
    }
}
