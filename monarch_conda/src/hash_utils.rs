/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;

use anyhow::Result;
use anyhow::bail;
use digest::Digest;
use tokio::fs;
use walkdir::WalkDir;

/// Compute a hash of a directory tree using the provided hasher.
///
/// This function traverses the directory tree deterministically (sorted by file name)
/// and includes both file paths and file contents in the hash computation.
///
/// # Arguments
/// * `dir` - The directory to hash
/// * `hasher` - A hasher implementing the Digest trait (e.g., Sha256::new())
///
/// # Returns
/// () - The hasher is updated with the directory tree data
pub async fn hash_directory_tree<D: Digest>(dir: &Path, hasher: &mut D) -> Result<()> {
    // Iterate entries with deterministic ordering
    for entry in WalkDir::new(dir).sort_by_file_name().into_iter() {
        let entry = entry?;
        let path = entry.path();
        let relative_path = path.strip_prefix(dir)?;

        // Hash the relative path (normalized to use forward slashes)
        let path_str = relative_path.to_string_lossy().replace('\\', "/");
        hasher.update(path_str.as_bytes());
        hasher.update(b"\0"); // null separator

        if entry.file_type().is_file() {
            // Hash file type marker, size, and contents
            hasher.update(b"FILE:");
            let contents = fs::read(path).await?;
            hasher.update(contents.len().to_le_bytes());
            hasher.update(&contents);
        } else if entry.file_type().is_dir() {
            // For directories, hash a type marker
            hasher.update(b"DIR:");
        } else if entry.file_type().is_symlink() {
            // For symlinks, hash type marker, target size, and target
            hasher.update(b"SYMLINK:");
            let target = fs::read_link(path).await?;
            let target_string = target.to_string_lossy().into_owned();
            let target_bytes = target_string.as_bytes();
            hasher.update(target_bytes.len().to_le_bytes());
            hasher.update(target_bytes);
        } else {
            // Unexpected file type
            bail!("Unexpected file type for path: {}", path.display());
        }

        hasher.update(b"\n"); // entry separator
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use sha2::Sha256;
    use tempfile::TempDir;
    use tokio::fs;

    use super::*;

    #[tokio::test]
    async fn test_hash_directory_tree() -> Result<()> {
        // Create a temporary directory with some test files
        let temp_dir = TempDir::new()?;
        let dir_path = temp_dir.path();

        // Create test files
        fs::write(dir_path.join("file1.txt"), "Hello, world!").await?;
        fs::write(dir_path.join("file2.txt"), "Another file").await?;
        fs::create_dir(dir_path.join("subdir")).await?;
        fs::write(dir_path.join("subdir").join("file3.txt"), "Nested file").await?;

        // Hash the directory
        let mut hasher1 = Sha256::new();
        let mut hasher2 = Sha256::new();
        hash_directory_tree(dir_path, &mut hasher1).await?;
        hash_directory_tree(dir_path, &mut hasher2).await?;

        let hash1 = hasher1.finalize();
        let hash2 = hasher2.finalize();

        // Should be deterministic
        assert_eq!(hash1, hash2);
        assert_eq!(hash1.len(), 32); // SHA256 raw bytes length

        Ok(())
    }

    #[tokio::test]
    async fn test_no_hash_collision_between_file_and_dir() -> Result<()> {
        // Test that a file containing "DIR:" and an empty directory don't collide
        let temp_dir1 = TempDir::new()?;
        let temp_dir2 = TempDir::new()?;

        // Create a file with content that could collide with directory marker
        fs::write(temp_dir1.path().join("test"), "DIR:").await?;

        // Create an empty directory with the same name
        fs::create_dir(temp_dir2.path().join("test")).await?;

        // Hash both scenarios
        let mut hasher_file = Sha256::new();
        let mut hasher_dir = Sha256::new();
        hash_directory_tree(temp_dir1.path(), &mut hasher_file).await?;
        hash_directory_tree(temp_dir2.path(), &mut hasher_dir).await?;

        let hash_file = hasher_file.finalize();
        let hash_dir = hasher_dir.finalize();

        // Should be different due to type prefixes
        assert_ne!(hash_file, hash_dir);

        Ok(())
    }

    #[tokio::test]
    async fn test_no_structural_marker_collision() -> Result<()> {
        // Test that files containing our structural markers don't cause collisions
        let temp_dir1 = TempDir::new()?;
        let temp_dir2 = TempDir::new()?;

        // Create a file that could potentially collide without size prefixes:
        // Path: "test1", Content: "foo\n"
        // Without size prefixes: test1\0FILE:foo\n\n
        fs::write(temp_dir1.path().join("test1"), "foo\n").await?;

        // Create a file with path that includes our structural markers:
        // Path: "test1\nFILE:", Content: "foo\n"
        // Without size prefixes: test1\nFILE:\0FILE:foo\n\n
        // This could potentially collide with the above
        fs::write(temp_dir2.path().join("test1\nFILE:"), "foo\n").await?;

        // Hash both scenarios
        let mut hasher1 = Sha256::new();
        let mut hasher2 = Sha256::new();
        hash_directory_tree(temp_dir1.path(), &mut hasher1).await?;
        hash_directory_tree(temp_dir2.path(), &mut hasher2).await?;

        let hash1 = hasher1.finalize();
        let hash2 = hasher2.finalize();

        // Should be different - size prefixes prevent structural marker confusion
        assert_ne!(hash1, hash2);

        Ok(())
    }
}
