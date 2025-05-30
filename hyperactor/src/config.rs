/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Configuration for Hyperactor.
//!
//! This module provides a centralized way to manage configuration settings for Hyperactor.
//! It abstracts away environment variables and allows configuration to be sourced from
//! different places (environment variables, YAML files, command line flags).
//! It also supports temporary modifications for tests.

use std::env;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::RwLock;
use std::time::Duration;

use serde::Deserialize;
use serde::Serialize;

/// Configuration for Hyperactor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Maximum frame length for codec
    pub codec_max_frame_length: usize,

    /// Message delivery timeout
    pub message_delivery_timeout: Duration,

    /// Message acknowledgment interval
    pub message_ack_time_interval: Duration,

    /// Number of messages after which to send an acknowledgment
    pub message_ack_every_n_messages: u64,

    /// Flag indicating if this is a managed subprocess
    pub is_managed_subprocess: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            codec_max_frame_length: 8 * 1024 * 1024, // 8 MB
            message_delivery_timeout: Duration::from_secs(30),
            message_ack_time_interval: Duration::from_millis(500),
            message_ack_every_n_messages: 1000,
            is_managed_subprocess: false,
        }
    }
}

impl Config {
    /// Load configuration from environment variables
    pub fn from_env() -> Self {
        let mut config = Self::default();

        // Load codec max frame length
        if let Ok(val) = env::var("HYPERACTOR_CODEC_MAX_FRAME_LENGTH") {
            if let Ok(parsed) = val.parse::<usize>() {
                config.codec_max_frame_length = parsed;
            }
        }

        // Load message delivery timeout
        if let Ok(val) = env::var("HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT_SECS") {
            if let Ok(parsed) = val.parse::<u64>() {
                config.message_delivery_timeout = Duration::from_secs(parsed);
            }
        }

        // Load message ack time interval
        if let Ok(val) = env::var("HYPERACTOR_MESSAGE_ACK_TIME_INTERVAL_MS") {
            if let Ok(parsed) = val.parse::<u64>() {
                config.message_ack_time_interval = Duration::from_millis(parsed);
            }
        }

        // Load message ack every n messages
        if let Ok(val) = env::var("HYPERACTOR_MESSAGE_ACK_EVERY_N_MESSAGES") {
            if let Ok(parsed) = val.parse::<u64>() {
                config.message_ack_every_n_messages = parsed;
            }
        }

        // Check if this is a managed subprocess
        config.is_managed_subprocess = env::var("HYPERACTOR_MANAGED_SUBPROCESS").is_ok();

        config
    }

    /// Load configuration from a YAML file
    pub fn from_yaml<P: AsRef<Path>>(path: P) -> Result<Self, anyhow::Error> {
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;

        let config = serde_yaml::from_str(&contents)?;
        Ok(config)
    }

    /// Merge with another configuration, with the other taking precedence
    pub fn merge(&mut self, other: &Self) {
        self.codec_max_frame_length = other.codec_max_frame_length;
        self.message_delivery_timeout = other.message_delivery_timeout;
        self.message_ack_time_interval = other.message_ack_time_interval;
        self.message_ack_every_n_messages = other.message_ack_every_n_messages;

        self.is_managed_subprocess = other.is_managed_subprocess;
    }

    /// Save configuration to a YAML file
    pub fn to_yaml<P: AsRef<Path>>(&self, path: P) -> Result<(), anyhow::Error> {
        let yaml = serde_yaml::to_string(self)?;
        std::fs::write(path, yaml)?;
        Ok(())
    }
}

/// Global configuration functions
pub mod global {
    use super::*;

    /// Global configuration instance, initialized from environment variables.
    static CONFIG: LazyLock<Arc<RwLock<Config>>> =
        LazyLock::new(|| Arc::new(RwLock::new(Config::from_env())));

    /// Initialize the global configuration from environment variables
    pub fn init_from_env() {
        let config = Config::from_env();
        let mut global_config = CONFIG.write().unwrap();
        *global_config = config;
    }

    /// Initialize the global configuration from a YAML file
    pub fn init_from_yaml<P: AsRef<Path>>(path: P) -> Result<(), anyhow::Error> {
        let config = Config::from_yaml(path)?;
        let mut global_config = CONFIG.write().unwrap();
        *global_config = config;
        Ok(())
    }

    /// Get a reference to the global configuration
    pub fn get() -> Arc<RwLock<Config>> {
        CONFIG.clone()
    }

    /// Get the codec max frame length
    pub fn codec_max_frame_length() -> usize {
        CONFIG.read().unwrap().codec_max_frame_length
    }

    /// Get the message delivery timeout
    pub fn message_delivery_timeout() -> Duration {
        CONFIG.read().unwrap().message_delivery_timeout
    }

    /// Get the message acknowledgment time interval
    pub fn message_ack_time_interval() -> Duration {
        CONFIG.read().unwrap().message_ack_time_interval
    }

    /// Get the number of messages after which to send an acknowledgment
    pub fn message_ack_every_n_messages() -> u64 {
        CONFIG.read().unwrap().message_ack_every_n_messages
    }

    /// A guard that restores the original configuration when dropped
    pub struct ConfigGuard {
        original_config: Config,
    }

    impl Drop for ConfigGuard {
        fn drop(&mut self) {
            let mut config = CONFIG.write().unwrap();
            *config = self.original_config.clone();
        }
    }

    /// Temporarily modify the configuration for testing
    ///
    /// Returns a guard that will restore the original configuration when dropped
    pub fn set_temp_config(temp_config: Config) -> ConfigGuard {
        let original_config = CONFIG.read().unwrap().clone();
        {
            let mut config = CONFIG.write().unwrap();
            *config = temp_config;
        }

        ConfigGuard { original_config }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.codec_max_frame_length, 8 * 1024 * 1024);
        assert_eq!(config.message_delivery_timeout, Duration::from_secs(30));
        assert_eq!(config.message_ack_time_interval, Duration::from_millis(500));
        assert_eq!(config.message_ack_every_n_messages, 1000);
    }

    #[test]
    fn test_from_env() {
        // Set environment variables
        std::env::set_var("HYPERACTOR_CODEC_MAX_FRAME_LENGTH", "1024");
        std::env::set_var("HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT_SECS", "60");

        let config = Config::from_env();

        assert_eq!(config.codec_max_frame_length, 1024);
        assert_eq!(config.message_delivery_timeout, Duration::from_secs(60));
        assert_eq!(config.message_ack_time_interval, Duration::from_millis(500)); // Default value

        // Clean up
        std::env::remove_var("HYPERACTOR_CODEC_MAX_FRAME_LENGTH");
        std::env::remove_var("HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT_SECS");
    }

    #[test]
    fn test_merge() {
        let mut config1 = Config::default();
        let config2 = Config {
            codec_max_frame_length: 1024,
            message_delivery_timeout: Duration::from_secs(60),
            ..Config::default()
        };

        config1.merge(&config2);

        assert_eq!(config1.codec_max_frame_length, 1024);
        assert_eq!(config1.message_delivery_timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_global_config() {
        assert_eq!(global::codec_max_frame_length(), 8 * 1024 * 1024);

        // Temporarily modify the configuration using the legacy function
        let temp_config = Config {
            codec_max_frame_length: 1024,
            ..Config::default()
        };
        {
            let _guard = global::set_temp_config(temp_config.clone());
            assert_eq!(global::codec_max_frame_length(), 1024);
        }

        // Check that the original configuration is restored
        assert_eq!(global::codec_max_frame_length(), 8 * 1024 * 1024);

        // Temporarily modify the configuration using the RAII pattern
        {
            let _guard = global::set_temp_config(temp_config);
            assert_eq!(global::codec_max_frame_length(), 1024);

            // The configuration will be automatically restored when _guard goes out of scope
        }

        // Check that the original configuration is restored
        assert_eq!(global::codec_max_frame_length(), 8 * 1024 * 1024);
    }
}
