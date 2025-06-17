/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Attribute dictionary for type-safe, heterogeneous key-value storage with serde support.
//!
//! This module provides `Attrs`, a type-safe dictionary that can store heterogeneous values
//! and serialize/deserialize them using serde. All stored values must implement
//! `Serialize + DeserializeOwned` to ensure the entire dictionary can be serialized.
//!
//! Keys are automatically registered at compile time using the `declare_attrs!` macro and the
//! inventory crate, eliminating the need for manual registry management.
//!
//! # Basic Usage
//!
//! ```
//! use std::time::Duration;
//!
//! use hyperactor::attrs::Attrs;
//! use hyperactor::attrs::declare_attrs;
//!
//! // Declare keys with their associated types
//! declare_attrs! {
//!    /// Request timeout
//!    attr TIMEOUT: Duration;
//!
//!   /// Maximum retry count
//!   attr MAX_RETRIES: u32 = 3;  // with default value
//! }
//!
//! let mut attrs = Attrs::new();
//! attrs.set(TIMEOUT, Duration::from_secs(30));
//!
//! assert_eq!(attrs.get(TIMEOUT), Some(&Duration::from_secs(30)));
//! assert_eq!(attrs.get(MAX_RETRIES), Some(&3));
//! ```
//!
//! # Serialization
//!
//! `Attrs` can be serialized to and deserialized automatically:
//!
//! ```
//! use std::time::Duration;
//!
//! use hyperactor::attrs::Attrs;
//! use hyperactor::attrs::declare_attrs;
//!
//! declare_attrs! {
//!   /// Request timeout
//!   pub attr TIMEOUT: Duration;
//! }
//!
//! let mut attrs = Attrs::new();
//! attrs.set(TIMEOUT, Duration::from_secs(30));
//!
//! // Serialize to JSON
//! let json = serde_json::to_string(&attrs).unwrap();
//!
//! // Deserialize from JSON (no manual registry needed!)
//! let deserialized: Attrs = serde_json::from_str(&json).unwrap();
//!
//! assert_eq!(deserialized.get(TIMEOUT), Some(&Duration::from_secs(30)));
//! ```

use std::any::Any;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::Index;
use std::ops::IndexMut;

use erased_serde::Deserializer as ErasedDeserializer;
use erased_serde::Serialize as ErasedSerialize;
use serde::Deserialize;
use serde::Deserializer;
use serde::Serialize;
use serde::Serializer;
use serde::de::DeserializeOwned;
use serde::de::MapAccess;
use serde::de::Visitor;
use serde::ser::SerializeMap;

use crate::data::Named;

// Information about an attribute key, used for automatic registration.
// This needs to be public to be accessible from other crates, but it is
// not part of the public API.
#[doc(hidden)]
pub struct AttrKeyInfo {
    /// Name of the key
    pub name: &'static str,
    /// Function to get the type hash of the associated value type
    pub typehash: fn() -> u64,
    /// Deserializer function that deserializes directly from any deserializer
    pub deserialize_erased:
        fn(&mut dyn ErasedDeserializer) -> Result<Box<dyn SerializableValue>, erased_serde::Error>,
}

inventory::collect!(AttrKeyInfo);

/// A typed key for the attribute dictionary.
///
/// Each key is associated with a specific type T and has a unique name.
/// Keys are typically created using the `declare_attrs!` macro which ensures they have
/// static lifetime and automatically registers them for serialization.
pub struct Key<T: 'static> {
    name: &'static str,
    default_value: Option<&'static T>,
    _phantom: PhantomData<T>,
}

impl<T: Named + 'static> Key<T> {
    /// Creates a new key with the given name.
    pub const fn new(name: &'static str) -> Self {
        Self {
            name,
            default_value: None,
            _phantom: PhantomData,
        }
    }

    /// Creates a new key with the given name and default value reference.
    pub const fn with_default(name: &'static str, default_value: &'static T) -> Self {
        Self {
            name,
            default_value: Some(default_value),
            _phantom: PhantomData,
        }
    }

    /// Returns the name of this key.
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Returns a reference to the default value for this key, if one exists.
    pub fn default(&self) -> Option<&'static T> {
        self.default_value
    }

    /// Returns whether this key has a default value.
    pub fn has_default(&self) -> bool {
        self.default_value.is_some()
    }

    /// Returns the type hash of the associated value type.
    pub fn typehash(&self) -> u64 {
        T::typehash()
    }
}

impl<T: 'static> Clone for Key<T> {
    fn clone(&self) -> Self {
        // Use Copy.
        *self
    }
}

impl<T: 'static> Copy for Key<T> {}

// Enable attr[key] syntax.
impl<T: Send + Sync + Serialize + DeserializeOwned + Named + 'static> Index<Key<T>> for Attrs {
    type Output = T;

    fn index(&self, key: Key<T>) -> &Self::Output {
        self.get(key).unwrap()
    }
}

// TODO: separately type keys with defaults, so that we can statically enforce that indexmut is only
// called on keys with defaults.
impl<T: Send + Sync + Serialize + DeserializeOwned + Named + Clone + 'static> IndexMut<Key<T>>
    for Attrs
{
    fn index_mut(&mut self, key: Key<T>) -> &mut Self::Output {
        self.get_mut(key).unwrap()
    }
}

// Internal trait for type-erased serialization
#[doc(hidden)]
pub trait SerializableValue: Send + Sync {
    /// Get a reference to this value as Any for downcasting
    fn as_any(&self) -> &dyn Any;
    /// Get a mutable reference to this value as Any for downcasting
    fn as_any_mut(&mut self) -> &mut dyn Any;
    /// Get a reference to this value as an erased serializable trait object
    fn as_erased_serialize(&self) -> &dyn ErasedSerialize;
    /// Clone the underlying value, retaining dyn compatibility.
    fn cloned(&self) -> Box<dyn SerializableValue>;
}

impl<T: Serialize + Send + Sync + Clone + 'static> SerializableValue for T {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn as_erased_serialize(&self) -> &dyn ErasedSerialize {
        self
    }

    fn cloned(&self) -> Box<dyn SerializableValue> {
        Box::new(self.clone())
    }
}

/// A heterogeneous, strongly-typed attribute dictionary with serialization support.
///
/// This dictionary stores key-value pairs where:
/// - Keys are type-safe and must be predefined with their associated types
/// - Values must implement `Send + Sync + Serialize + DeserializeOwned + Named + 'static`
/// - The entire dictionary can be serialized to/from JSON automatically
///
/// # Type Safety
///
/// The dictionary enforces type safety at compile time. You cannot retrieve a value
/// with the wrong type, and the compiler will catch such errors.
///
/// # Serialization
///
/// The dictionary can be serialized using serde. During serialization, each value
/// is serialized with its key name. During deserialization, the automatically registered
/// key information is used to determine the correct type for each value.
pub struct Attrs {
    values: HashMap<&'static str, Box<dyn SerializableValue>>,
}

impl Attrs {
    /// Create a new empty attribute dictionary.
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    /// Set a value for the given key.
    pub fn set<T: Send + Sync + Serialize + DeserializeOwned + Named + Clone + 'static>(
        &mut self,
        key: Key<T>,
        value: T,
    ) {
        self.values.insert(key.name, Box::new(value));
    }

    fn maybe_set_from_default<
        T: Send + Sync + Serialize + DeserializeOwned + Named + Clone + 'static,
    >(
        &mut self,
        key: Key<T>,
    ) {
        if self.contains_key(key) {
            return;
        }
        let Some(default) = key.default() else { return };
        self.set(key, default.clone());
    }

    /// Get a value for the given key, returning None if not present. If the key has a default value,
    /// that is returned instead.
    pub fn get<T: Send + Sync + Serialize + DeserializeOwned + Named + 'static>(
        &self,
        key: Key<T>,
    ) -> Option<&T> {
        self.values
            .get(key.name)
            .and_then(|value| value.as_any().downcast_ref::<T>())
            .or_else(|| key.default())
    }

    /// Get a mutable reference to a value for the given key. If the key has a default value, it is
    /// first set, and then returned as a mutable reference.
    pub fn get_mut<T: Send + Sync + Serialize + DeserializeOwned + Named + Clone + 'static>(
        &mut self,
        key: Key<T>,
    ) -> Option<&mut T> {
        self.maybe_set_from_default(key);
        self.values
            .get_mut(key.name)
            .and_then(|value| value.as_any_mut().downcast_mut::<T>())
    }

    /// Remove a value for the given key, returning it if present.
    pub fn remove<T: Send + Sync + Serialize + DeserializeOwned + Named + 'static>(
        &mut self,
        key: Key<T>,
    ) -> bool {
        // TODO: return value (this is tricky because of the type erasure)
        self.values.remove(key.name).is_some()
    }

    /// Checks if the given key exists in the dictionary.
    pub fn contains_key<T: Send + Sync + Serialize + DeserializeOwned + Named + 'static>(
        &self,
        key: Key<T>,
    ) -> bool {
        self.values.contains_key(key.name)
    }

    /// Returns the number of key-value pairs in the dictionary.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns true if the dictionary is empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Clear all key-value pairs from the dictionary.
    pub fn clear(&mut self) {
        self.values.clear();
    }

    // Internal methods for config guard support
    /// Take a value by key name, returning the boxed value if present
    pub(crate) fn take_value<T: 'static>(
        &mut self,
        key: Key<T>,
    ) -> Option<Box<dyn SerializableValue>> {
        self.values.remove(key.name)
    }

    /// Restore a value by key name
    pub(crate) fn restore_value<T: 'static>(
        &mut self,
        key: Key<T>,
        value: Box<dyn SerializableValue>,
    ) {
        self.values.insert(key.name, value);
    }

    /// Remove a value by key name
    pub(crate) fn remove_value<T: 'static>(&mut self, key: Key<T>) -> bool {
        self.values.remove(key.name).is_some()
    }
}

impl Clone for Attrs {
    fn clone(&self) -> Self {
        let mut values = HashMap::new();
        for (key, value) in &self.values {
            values.insert(*key, value.cloned());
        }
        Self { values }
    }
}

impl std::fmt::Debug for Attrs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Create a map of key names to their JSON representation for debugging
        let mut debug_map = std::collections::BTreeMap::new();
        for (key, value) in &self.values {
            match serde_json::to_string(value.as_erased_serialize()) {
                Ok(json) => {
                    debug_map.insert(*key, json);
                }
                Err(_) => {
                    debug_map.insert(*key, "<serialization error>".to_string());
                }
            }
        }

        f.debug_struct("Attrs").field("values", &debug_map).finish()
    }
}

impl Default for Attrs {
    fn default() -> Self {
        Self::new()
    }
}

impl Serialize for Attrs {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(self.values.len()))?;

        for (key_name, value) in &self.values {
            map.serialize_entry(key_name, value.as_erased_serialize())?;
        }

        map.end()
    }
}

struct AttrsVisitor;

impl<'de> Visitor<'de> for AttrsVisitor {
    type Value = Attrs;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a map of attribute keys to their serialized values")
    }

    fn visit_map<M>(self, mut access: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        static KEYS_BY_NAME: std::sync::LazyLock<HashMap<&'static str, &'static AttrKeyInfo>> =
            std::sync::LazyLock::new(|| {
                inventory::iter::<AttrKeyInfo>()
                    .map(|info| (info.name, info))
                    .collect()
            });
        let keys_by_name = &*KEYS_BY_NAME;

        let mut attrs = Attrs::new();
        while let Some(key_name) = access.next_key::<String>()? {
            let Some(&key) = keys_by_name.get(key_name.as_str()) else {
                // Silently ignore unknown keys
                access.next_value::<serde::de::IgnoredAny>()?;
                continue;
            };

            // Create a seed to deserialize the value using erased_serde
            let seed = ValueDeserializeSeed {
                deserialize_erased: key.deserialize_erased,
            };
            match access.next_value_seed(seed) {
                Ok(value) => {
                    attrs.values.insert(key.name, value);
                }
                Err(err) => {
                    return Err(serde::de::Error::custom(format!(
                        "failed to deserialize value for key {}: {}",
                        key_name, err
                    )));
                }
            }
        }

        Ok(attrs)
    }
}

/// Helper struct to deserialize values using erased_serde
struct ValueDeserializeSeed {
    deserialize_erased:
        fn(&mut dyn ErasedDeserializer) -> Result<Box<dyn SerializableValue>, erased_serde::Error>,
}

impl<'de> serde::de::DeserializeSeed<'de> for ValueDeserializeSeed {
    type Value = Box<dyn SerializableValue>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::de::Deserializer<'de>,
    {
        let mut erased = <dyn erased_serde::Deserializer>::erase(deserializer);
        (self.deserialize_erased)(&mut erased).map_err(serde::de::Error::custom)
    }
}

impl<'de> Deserialize<'de> for Attrs {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(AttrsVisitor)
    }
}

// Converts an ASCII string to lowercase at compile time.
// Returns a const string with lowercase ASCII characters.
#[doc(hidden)]
pub const fn ascii_to_lowercase_const<const N: usize>(input: &str) -> [u8; N] {
    let bytes = input.as_bytes();
    let mut result = [0u8; N];
    let mut i = 0;

    while i < bytes.len() && i < N {
        let byte = bytes[i];
        if byte >= b'A' && byte <= b'Z' {
            result[i] = byte + 32; // Convert to lowercase
        } else {
            result[i] = byte;
        }
        i += 1;
    }

    result
}

// Macro to generate a const lowercase string at compile time
#[doc(hidden)]
#[macro_export]
macro_rules! const_ascii_lowercase {
    ($s:expr) => {{
        const INPUT: &str = $s;
        const LEN: usize = INPUT.len();
        const BYTES: [u8; LEN] = $crate::attrs::ascii_to_lowercase_const::<LEN>(INPUT);
        // Safety: We're converting ASCII to ASCII, so it's valid UTF-8
        unsafe { std::str::from_utf8_unchecked(&BYTES) }
    }};
}

/// Declares attribute keys using a lazy_static! style syntax.
///
/// # Syntax
///
/// ```ignore
/// declare_attrs! {
///     /// Documentation for the key (default visibility).
///     attr KEY_NAME: Type = default_value;
///     
///     /// Another key (default value is optional)
///     pub attr ANOTHER_KEY: AnotherType;
/// }
/// ```
///
/// # Arguments
///
/// * Optional visibility modifier (`pub`, `pub(crate)`, etc.)
/// * `attr` keyword (required)
/// * Key name (identifier)
/// * Type of values this key can store
/// * Optional default value
///
/// # Example
///
/// ```
/// use std::time::Duration;
///
/// use hyperactor::attrs::Attrs;
/// use hyperactor::attrs::declare_attrs;
///
/// declare_attrs! {
///     /// Timeout for RPC operations
///     pub attr TIMEOUT: Duration = Duration::from_secs(30);
///
///     /// Maximum number of retry attempts (no default specified)
///     attr MAX_RETRIES: u32;
/// }
///
/// let mut attrs = Attrs::new();
/// assert_eq!(attrs.get(TIMEOUT), Some(&Duration::from_secs(30)));
/// attrs.set(MAX_RETRIES, 5);
/// ```
#[macro_export]
macro_rules! declare_attrs {
    // Handle multiple attribute keys with optional default values
    ($(
        $(#[$attr:meta])*
        $vis:vis attr $name:ident: $type:ty $(= $default:expr)?;
    )*) => {
        $(
            $crate::declare_attrs! { @single $(#[$attr])* ; $vis attr $name: $type $(= $default)?; }
        )*
    };

    // Handle single attribute key with default value
    (@single $(#[$attr:meta])* ; $vis:vis attr $name:ident: $type:ty = $default:expr;) => {
        // Create a static default value
        $crate::paste! {
            static [<$name _DEFAULT>]: $type = $default;
        }

        $(#[$attr])*
        $vis static $name: $crate::attrs::Key<$type> = {
            const FULL_NAME: &str = concat!(std::module_path!(), "::", stringify!($name));
            const LOWER_NAME: &str = $crate::const_ascii_lowercase!(FULL_NAME);
            $crate::paste! {
                $crate::attrs::Key::with_default(
                    LOWER_NAME,
                    &[<$name _DEFAULT>]
                )
            }
        };

        // Register the key for serialization
        $crate::submit! {
            $crate::attrs::AttrKeyInfo {
                name: {
                    const FULL_NAME: &str = concat!(std::module_path!(), "::", stringify!($name));
                    $crate::const_ascii_lowercase!(FULL_NAME)
                },
                typehash: <$type as $crate::data::Named>::typehash,
                deserialize_erased: |deserializer| {
                    let value: $type = erased_serde::deserialize(deserializer)?;
                    Ok(Box::new(value) as Box<dyn $crate::attrs::SerializableValue>)
                },
            }
        }
    };

    // Handle single attribute key without default value
    (@single $(#[$attr:meta])* ; $vis:vis attr $name:ident: $type:ty;) => {
        $(#[$attr])*
        $vis static $name: $crate::attrs::Key<$type> = {
            const FULL_NAME: &str = concat!(std::module_path!(), "::", stringify!($name));
            const LOWER_NAME: &str = $crate::const_ascii_lowercase!(FULL_NAME);
            $crate::attrs::Key::new(LOWER_NAME)
        };

        // Register the key for serialization
        $crate::submit! {
            $crate::attrs::AttrKeyInfo {
                name: {
                    const FULL_NAME: &str = concat!(std::module_path!(), "::", stringify!($name));
                    $crate::const_ascii_lowercase!(FULL_NAME)
                },
                typehash: <$type as $crate::data::Named>::typehash,
                deserialize_erased: |deserializer| {
                    let value: $type = erased_serde::deserialize(deserializer)?;
                    Ok(Box::new(value) as Box<dyn $crate::attrs::SerializableValue>)
                },
            }
        }
    };
}

pub use declare_attrs;

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    declare_attrs! {
        attr TEST_TIMEOUT: Duration;
        attr TEST_COUNT: u32;
        attr TEST_NAME: String;
    }

    #[test]
    fn test_basic_operations() {
        let mut attrs = Attrs::new();

        // Test setting and getting values
        attrs.set(TEST_TIMEOUT, Duration::from_secs(5));
        attrs.set(TEST_COUNT, 42u32);
        attrs.set(TEST_NAME, "test".to_string());

        assert_eq!(attrs.get(TEST_TIMEOUT), Some(&Duration::from_secs(5)));
        assert_eq!(attrs.get(TEST_COUNT), Some(&42u32));
        assert_eq!(attrs.get(TEST_NAME), Some(&"test".to_string()));

        // Test contains_key
        assert!(attrs.contains_key(TEST_TIMEOUT));
        assert!(attrs.contains_key(TEST_COUNT));
        assert!(attrs.contains_key(TEST_NAME));

        // Test len
        assert_eq!(attrs.len(), 3);
        assert!(!attrs.is_empty());
    }

    #[test]
    fn test_get_mut() {
        let mut attrs = Attrs::new();
        attrs.set(TEST_COUNT, 10u32);

        if let Some(count) = attrs.get_mut(TEST_COUNT) {
            *count += 5;
        }

        assert_eq!(attrs.get(TEST_COUNT), Some(&15u32));
    }

    #[test]
    fn test_remove() {
        let mut attrs = Attrs::new();
        attrs.set(TEST_COUNT, 42u32);

        let removed = attrs.remove(TEST_COUNT);
        assert!(removed);
        assert_eq!(attrs.get(TEST_COUNT), None);
        assert!(!attrs.contains_key(TEST_COUNT));
    }

    #[test]
    fn test_clear() {
        let mut attrs = Attrs::new();
        attrs.set(TEST_TIMEOUT, Duration::from_secs(1));
        attrs.set(TEST_COUNT, 42u32);

        attrs.clear();
        assert!(attrs.is_empty());
        assert_eq!(attrs.len(), 0);
    }

    #[test]
    fn test_key_properties() {
        assert_eq!(
            TEST_TIMEOUT.name(),
            "hyperactor::attrs::tests::test_timeout"
        );
    }

    #[test]
    fn test_serialization() {
        let mut attrs = Attrs::new();
        attrs.set(TEST_TIMEOUT, Duration::from_secs(5));
        attrs.set(TEST_COUNT, 42u32);
        attrs.set(TEST_NAME, "test".to_string());

        // Test serialization
        let serialized = serde_json::to_string(&attrs).expect("Failed to serialize");

        // The serialized string should contain the key names and their values
        assert!(serialized.contains("hyperactor::attrs::tests::test_timeout"));
        assert!(serialized.contains("hyperactor::attrs::tests::test_count"));
        assert!(serialized.contains("hyperactor::attrs::tests::test_name"));
    }

    #[test]
    fn test_deserialization() {
        // Create original attrs
        let mut original_attrs = Attrs::new();
        original_attrs.set(TEST_TIMEOUT, Duration::from_secs(5));
        original_attrs.set(TEST_COUNT, 42u32);
        original_attrs.set(TEST_NAME, "test".to_string());

        // Serialize
        let serialized = serde_json::to_string(&original_attrs).expect("Failed to serialize");

        // Deserialize (no manual registry needed!)
        let deserialized_attrs: Attrs =
            serde_json::from_str(&serialized).expect("Failed to deserialize");

        // Verify the deserialized values
        assert_eq!(
            deserialized_attrs.get(TEST_TIMEOUT),
            Some(&Duration::from_secs(5))
        );
        assert_eq!(deserialized_attrs.get(TEST_COUNT), Some(&42u32));
        assert_eq!(deserialized_attrs.get(TEST_NAME), Some(&"test".to_string()));
    }

    #[test]
    fn test_roundtrip_serialization() {
        // Create original attrs
        let mut original = Attrs::new();
        original.set(TEST_TIMEOUT, Duration::from_secs(10));
        original.set(TEST_COUNT, 5u32);
        original.set(TEST_NAME, "test-service".to_string());

        // Serialize
        let serialized = serde_json::to_string(&original).unwrap();

        // Deserialize
        let deserialized: Attrs = serde_json::from_str(&serialized).unwrap();

        // Verify round-trip worked
        assert_eq!(
            deserialized.get(TEST_TIMEOUT),
            Some(&Duration::from_secs(10))
        );
        assert_eq!(deserialized.get(TEST_COUNT), Some(&5u32));
        assert_eq!(
            deserialized.get(TEST_NAME),
            Some(&"test-service".to_string())
        );
    }

    #[test]
    fn test_empty_attrs_serialization() {
        let attrs = Attrs::new();
        let serialized = serde_json::to_string(&attrs).unwrap();

        // Empty attrs should serialize to empty JSON object
        assert_eq!(serialized, "{}");

        let deserialized: Attrs = serde_json::from_str(&serialized).unwrap();

        assert!(deserialized.is_empty());
    }

    #[test]
    fn test_format_independence() {
        // Test that proves we're using the serializer directly, not JSON internally
        let mut attrs = Attrs::new();
        attrs.set(TEST_COUNT, 42u32);
        attrs.set(TEST_NAME, "test".to_string());

        // Serialize to different formats
        let json_output = serde_json::to_string(&attrs).unwrap();
        let yaml_output = serde_yaml::to_string(&attrs).unwrap();

        // JSON should have colons and quotes
        assert!(json_output.contains(":"));
        assert!(json_output.contains("\""));

        // JSON should serialize numbers as numbers, not strings
        assert!(json_output.contains("42"));
        assert!(!json_output.contains("\"42\""));

        // YAML should have colons but different formatting
        assert!(yaml_output.contains(":"));
        assert!(yaml_output.contains("42"));

        // YAML shouldn't quote simple strings or numbers
        assert!(!yaml_output.contains("\"42\""));

        // The outputs should be different (proving different serializers were used)
        assert_ne!(json_output, yaml_output);

        // Verify that both can be deserialized correctly
        let from_json: Attrs = serde_json::from_str(&json_output).unwrap();
        let from_yaml: Attrs = serde_yaml::from_str(&yaml_output).unwrap();

        assert_eq!(from_json.get(TEST_COUNT), Some(&42u32));
        assert_eq!(from_yaml.get(TEST_COUNT), Some(&42u32));
        assert_eq!(from_json.get(TEST_NAME), Some(&"test".to_string()));
        assert_eq!(from_yaml.get(TEST_NAME), Some(&"test".to_string()));
    }

    #[test]
    fn test_clone() {
        // Create original attrs with multiple types
        let mut original = Attrs::new();
        original.set(TEST_COUNT, 42u32);
        original.set(TEST_NAME, "test".to_string());
        original.set(TEST_TIMEOUT, std::time::Duration::from_secs(10));

        // Clone the attrs
        let cloned = original.clone();

        // Verify that the clone has the same values
        assert_eq!(cloned.get(TEST_COUNT), Some(&42u32));
        assert_eq!(cloned.get(TEST_NAME), Some(&"test".to_string()));
        assert_eq!(
            cloned.get(TEST_TIMEOUT),
            Some(&std::time::Duration::from_secs(10))
        );

        // Verify that modifications to the original don't affect the clone
        original.set(TEST_COUNT, 100u32);
        assert_eq!(original.get(TEST_COUNT), Some(&100u32));
        assert_eq!(cloned.get(TEST_COUNT), Some(&42u32)); // Clone should be unchanged

        // Verify that modifications to the clone don't affect the original
        let mut cloned_mut = cloned.clone();
        cloned_mut.set(TEST_NAME, "modified".to_string());
        assert_eq!(cloned_mut.get(TEST_NAME), Some(&"modified".to_string()));
        assert_eq!(original.get(TEST_NAME), Some(&"test".to_string())); // Original should be unchanged
    }

    #[test]
    fn test_debug_with_json() {
        let mut attrs = Attrs::new();
        attrs.set(TEST_COUNT, 42u32);
        attrs.set(TEST_NAME, "test".to_string());

        // Test that Debug implementation works and contains JSON representations
        let debug_output = format!("{:?}", attrs);

        // Should contain the struct name
        assert!(debug_output.contains("Attrs"));

        // Should contain JSON representations of the values
        assert!(debug_output.contains("42"));

        // Should contain the key names
        assert!(debug_output.contains("hyperactor::attrs::tests::test_count"));
        assert!(debug_output.contains("hyperactor::attrs::tests::test_name"));

        // For strings, the JSON representation should be the escaped version
        // Let's check that the test string is actually present in some form
        assert!(debug_output.contains("test"));
    }

    declare_attrs! {
        /// With default...
        attr TIMEOUT_WITH_DEFAULT: Duration = Duration::from_secs(10);

        /// Just to ensure visibilty is parsed.
        pub(crate) attr CRATE_LOCAL_ATTR: String;
    }

    #[test]
    fn test_defaults() {
        assert!(TIMEOUT_WITH_DEFAULT.has_default());
        assert!(!CRATE_LOCAL_ATTR.has_default());

        assert_eq!(
            Attrs::new().get(TIMEOUT_WITH_DEFAULT),
            Some(&Duration::from_secs(10))
        );
    }

    #[test]
    fn test_indexing() {
        let mut attrs = Attrs::new();

        assert_eq!(attrs[TIMEOUT_WITH_DEFAULT], Duration::from_secs(10));
        attrs[TIMEOUT_WITH_DEFAULT] = Duration::from_secs(100);
        assert_eq!(attrs[TIMEOUT_WITH_DEFAULT], Duration::from_secs(100));

        attrs.set(CRATE_LOCAL_ATTR, "test".to_string());
        assert_eq!(attrs[CRATE_LOCAL_ATTR], "test".to_string());
    }
}
