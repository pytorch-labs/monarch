/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Configuration for Hyperactor Mesh.
//!
//! This module provides a centralized way to manage configuration settings for Hyperactor Mesh.
//! It uses the attrs system for type-safe, flexible configuration management that supports
//! environment variables, YAML files, and temporary modifications for tests.

use std::env;
use std::time::Duration;

use hyperactor::attrs::Attrs;
use hyperactor::attrs::declare_attrs;

// Declare configuration keys using the attrs system with defaults
declare_attrs! {
    /// Heartbeat interval for remote allocator
    pub attr REMOTE_ALLOCATOR_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);
}

/// Load configuration from environment variables
pub fn from_env() -> Attrs {
    let mut config = Attrs::new();

    // Load remote allocator heartbeat interval
    if let Ok(val) = env::var("HYPERACTOR_MESH_REMOTE_ALLOCATOR_HEARTBEAT_INTERVAL_SECS") {
        if let Ok(parsed) = val.parse::<u64>() {
            config[REMOTE_ALLOCATOR_HEARTBEAT_INTERVAL] = Duration::from_secs(parsed);
        }
    }

    config
}

/// Merge with another configuration, with the other taking precedence
pub fn merge(config: &mut Attrs, other: &Attrs) {
    if other.contains_key(REMOTE_ALLOCATOR_HEARTBEAT_INTERVAL) {
        config[REMOTE_ALLOCATOR_HEARTBEAT_INTERVAL] = other[REMOTE_ALLOCATOR_HEARTBEAT_INTERVAL];
    }
}

/// Global configuration functions
pub mod global {
    use std::sync::Arc;
    use std::sync::LazyLock;
    use std::sync::RwLock;

    use hyperactor::attrs::Key;
    use hyperactor::config::global::ConfigLock;

    use super::*;

    /// Global configuration instance, initialized from environment variables.
    static CONFIG: LazyLock<Arc<RwLock<Attrs>>> =
        LazyLock::new(|| Arc::new(RwLock::new(from_env())));

    /// Get a key from the global configuration.
    pub fn get<
        T: Send
            + Sync
            + Copy
            + serde::Serialize
            + serde::de::DeserializeOwned
            + hyperactor::data::Named
            + 'static,
    >(
        key: Key<T>,
    ) -> T {
        *CONFIG.read().unwrap().get(key).unwrap()
    }

    /// Reset the global configuration to defaults (for testing only)
    pub fn reset_to_defaults() {
        let mut config = CONFIG.write().unwrap();
        *config = Attrs::new();
    }

    /// Acquire the global configuration lock for testing.
    pub fn lock() -> ConfigLock {
        hyperactor::config::global::new(CONFIG.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Attrs::new();
        assert_eq!(
            config[REMOTE_ALLOCATOR_HEARTBEAT_INTERVAL],
            Duration::from_secs(5)
        );
    }

    #[test]
    fn test_from_env() {
        // Set environment variables
        // SAFETY: TODO: Audit that the environment access only happens in single-threaded code.
        unsafe {
            std::env::set_var(
                "HYPERACTOR_MESH_REMOTE_ALLOCATOR_HEARTBEAT_INTERVAL_SECS",
                "30",
            )
        };

        let config = from_env();

        assert_eq!(
            config[REMOTE_ALLOCATOR_HEARTBEAT_INTERVAL],
            Duration::from_secs(30)
        );

        // Clean up
        // SAFETY: TODO: Audit that the environment access only happens in single-threaded code.
        unsafe { std::env::remove_var("HYPERACTOR_MESH_REMOTE_ALLOCATOR_HEARTBEAT_INTERVAL_SECS") };
    }

    #[test]
    fn test_merge() {
        let mut config1 = Attrs::new();
        let mut config2 = Attrs::new();
        config2[REMOTE_ALLOCATOR_HEARTBEAT_INTERVAL] = Duration::from_secs(30);

        merge(&mut config1, &config2);

        assert_eq!(
            config1[REMOTE_ALLOCATOR_HEARTBEAT_INTERVAL],
            Duration::from_secs(30)
        );
    }

    #[test]
    fn test_global_config() {
        let config = global::lock();

        // Reset global config to defaults to avoid interference from other tests
        global::reset_to_defaults();

        assert_eq!(
            global::get(REMOTE_ALLOCATOR_HEARTBEAT_INTERVAL),
            Duration::from_secs(5)
        );
        {
            let _guard =
                config.override_key(REMOTE_ALLOCATOR_HEARTBEAT_INTERVAL, Duration::from_secs(30));
            assert_eq!(
                global::get(REMOTE_ALLOCATOR_HEARTBEAT_INTERVAL),
                Duration::from_secs(30)
            );
        }
        assert_eq!(
            global::get(REMOTE_ALLOCATOR_HEARTBEAT_INTERVAL),
            Duration::from_secs(5)
        );
    }

    #[test]
    fn test_defaults() {
        // Test that empty config now returns defaults via get_or_default
        let config = Attrs::new();

        // Verify that the config is empty (no values explicitly set)
        assert!(config.is_empty());

        // But getters should still return the defaults from the keys
        assert_eq!(
            config[REMOTE_ALLOCATOR_HEARTBEAT_INTERVAL],
            Duration::from_secs(5)
        );

        // Verify the keys have defaults
        assert!(REMOTE_ALLOCATOR_HEARTBEAT_INTERVAL.has_default());

        // Verify we can get defaults directly from keys
        assert_eq!(
            REMOTE_ALLOCATOR_HEARTBEAT_INTERVAL.default(),
            Some(&Duration::from_secs(5))
        );
    }
}
