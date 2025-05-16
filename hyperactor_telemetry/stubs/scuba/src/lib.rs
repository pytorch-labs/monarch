//! Stub implementation of scuba for OSS builds
//!
//! This is a minimal implementation that provides the necessary API surface
//! for code that depends on scuba, but doesn't actually do anything.

use std::collections::HashMap;

/// A stub for the ScubaValue
#[derive(Debug, Clone)]
pub enum ScubaValue {
    Int(i64),
    Double(f64),
    String(String),
    Bool(bool),
    Null,
}

impl From<i64> for ScubaValue {
    fn from(v: i64) -> Self {
        ScubaValue::Int(v)
    }
}

impl From<f64> for ScubaValue {
    fn from(v: f64) -> Self {
        ScubaValue::Double(v)
    }
}

impl From<String> for ScubaValue {
    fn from(v: String) -> Self {
        ScubaValue::String(v)
    }
}

impl From<&str> for ScubaValue {
    fn from(v: &str) -> Self {
        ScubaValue::String(v.to_string())
    }
}

impl From<bool> for ScubaValue {
    fn from(v: bool) -> Self {
        ScubaValue::Bool(v)
    }
}

/// A stub for the ScubaLogger
#[derive(Debug, Clone)]
pub struct ScubaLogger {
    _sample_rate: f64,
    _data: HashMap<String, ScubaValue>,
}

impl ScubaLogger {
    /// Create a new ScubaLogger (stub implementation)
    pub fn new() -> Self {
        ScubaLogger {
            _sample_rate: 1.0,
            _data: HashMap::new(),
        }
    }

    /// Create a new ScubaLogger with a dataset (stub implementation)
    pub fn new_with_dataset(_dataset: &str) -> Self {
        Self::new()
    }

    /// Add a key-value pair to the logger (stub implementation)
    pub fn add<K, V>(&mut self, key: K, value: V) -> &mut Self
    where
        K: Into<String>,
        V: Into<ScubaValue>,
    {
        self._data.insert(key.into(), value.into());
        self
    }

    /// Log the data (stub implementation)
    pub fn log(&self) {
        // Do nothing in the stub implementation
    }

    /// Clone the logger (stub implementation)
    pub fn clone_logger(&self) -> Self {
        self.clone()
    }
}

/// A stub for the ScubaSampleBuilder
#[derive(Debug, Clone)]
pub struct ScubaSampleBuilder {
    _logger: ScubaLogger,
}

impl ScubaSampleBuilder {
    /// Create a new ScubaSampleBuilder (stub implementation)
    pub fn new(_dataset: &str) -> Self {
        ScubaSampleBuilder {
            _logger: ScubaLogger::new(),
        }
    }

    /// Add a key-value pair to the builder (stub implementation)
    pub fn add<K, V>(&mut self, key: K, value: V) -> &mut Self
    where
        K: Into<String>,
        V: Into<ScubaValue>,
    {
        self._logger.add(key.into(), value.into());
        self
    }

    /// Log the data (stub implementation)
    pub fn log(&self) {
        // Do nothing in the stub implementation
    }
}
