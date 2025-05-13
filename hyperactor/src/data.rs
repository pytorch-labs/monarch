//! This module contains core traits and implementation to manage remote data
//! types in Hyperactor.

/// A [`Named`] type is a type that has a globally unique name.
pub trait Named: Sized + 'static {
    /// The globally unique type name for the type.
    /// This should typically be the fully qualified Rust name of the type.
    fn typename() -> &'static str;

    /// A globally unique hash for this type.
    /// TODO: actually enforce perfect hashing
    fn typehash() -> u64 {
        // The `Named` macro overrides this implementation with one that
        // memoizes the hash.
        cityhasher::hash(Self::typename())
    }

    /// The TypeId for this type. TypeIds are unique only within a binary,
    /// and should not be used for global identification.
    fn typeid() -> TypeId {
        TypeId::of::<Self>()
    }

    /// The globally unique port for this type. Typed ports are in the range
    /// of 1<<63..1<<64-1.
    fn port() -> u64 {
        Self::typehash() | (1 << 63)
    }

    /// If the named type is an enum, this returns the name of the arm
    /// of the value self.
    fn arm(&self) -> Option<&'static str> {
        None
    }

    /// An unsafe version of 'arm', accepting a pointer to the value,
    /// for use in type-erased settings.
    unsafe fn arm_unchecked(self_: *const ()) -> Option<&'static str> {
        // SAFETY: This isn't safe. We're passing it on.
        unsafe { &*(self_ as *const Self) }.arm()
    }
}

#[doc(hidden)]
/// Dump trait for Named types that are also serializable/deserializable.
/// This is a utility used by [`Serialized::dump`], and is not intended
/// for direct use.
pub trait NamedDumpable: Named + Serialize + for<'de> Deserialize<'de> {
    /// Dump the data in Serialized to a JSON value.
    fn dump(data: Serialized) -> Result<serde_json::Value, anyhow::Error>;
}

impl<T: Named + Serialize + for<'de> Deserialize<'de>> NamedDumpable for T {
    fn dump(data: Serialized) -> Result<serde_json::Value, anyhow::Error> {
        let value = data.deserialized::<Self>()?;
        Ok(serde_json::to_value(value)?)
    }
}

macro_rules! impl_basic {
    ($t:ty) => {
        impl Named for $t {
            fn typename() -> &'static str {
                stringify!($t)
            }
        }
    };
}

impl_basic!(());
impl_basic!(bool);
impl_basic!(i8);
impl_basic!(u8);
impl_basic!(i16);
impl_basic!(u16);
impl_basic!(i32);
impl_basic!(u32);
impl_basic!(i64);
impl_basic!(u64);
impl_basic!(i128);
impl_basic!(u128);
impl_basic!(isize);
impl_basic!(usize);
impl_basic!(String);

// A macro that implements type-keyed interning of typenames. This is useful
// for implementing [`Named`] for generic types.
#[doc(hidden)] // not part of the public API
#[macro_export]
macro_rules! intern_typename {
    ($key:ty, $format_string:expr, $($args:ty),+) => {
        {
            static CACHE: std::sync::LazyLock<$crate::dashmap::DashMap<std::any::TypeId, &'static str>> =
              std::sync::LazyLock::new($crate::dashmap::DashMap::new);

            match CACHE.entry(std::any::TypeId::of::<$key>()) {
                $crate::dashmap::mapref::entry::Entry::Vacant(entry) => {
                    let typename = format!($format_string, $(<$args>::typename()),+).leak();
                    entry.insert(typename);
                    typename
                }
                $crate::dashmap::mapref::entry::Entry::Occupied(entry) => *entry.get(),
            }
        }
    };
}
use std::any::TypeId;
use std::collections::HashMap;
use std::fmt;
use std::sync::LazyLock;

pub use intern_typename;
use serde::Deserialize;
use serde::Serialize;
use serde::de::DeserializeOwned;

macro_rules! tuple_format_string {
    ($a:ident,) => { "{}" };
    ($a:ident, $($rest_a:ident,)+) => { concat!("{}, ", tuple_format_string!($($rest_a,)+)) };
}

macro_rules! impl_tuple_peel {
    ($name:ident, $($other:ident,)*) => (impl_tuple! { $($other,)* })
}

macro_rules! impl_tuple {
    () => ();
    ( $($name:ident,)+ ) => (
        impl<$($name:Named + 'static),+> Named for ($($name,)+) {
            fn typename() -> &'static str {
                intern_typename!(Self, concat!("(", tuple_format_string!($($name,)+), ")"), $($name),+)
            }
        }
        impl_tuple_peel! { $($name,)+ }
    )
}

impl_tuple! { E, D, C, B, A, Z, Y, X, W, V, U, T, }

impl<T: Named + 'static> Named for Option<T> {
    fn typename() -> &'static str {
        intern_typename!(Self, "Option<{}>", T)
    }
}

impl<T: Named + 'static> Named for Vec<T> {
    fn typename() -> &'static str {
        intern_typename!(Self, "Vec<{}>", T)
    }
}

impl<T: Named + 'static, E: Named + 'static> Named for Result<T, E> {
    fn typename() -> &'static str {
        intern_typename!(Self, "Result<{}, {}>", T, E)
    }
}

/// Really internal, but needs to be exposed for macro.
#[doc(hidden)]
#[derive(Debug)]
pub struct TypeInfo {
    /// Named::typename()
    pub typename: fn() -> &'static str,
    /// Named::typehash()
    pub typehash: fn() -> u64,
    /// Named::typeid()
    pub typeid: fn() -> TypeId,
    /// Named::typehash()
    pub port: fn() -> u64,
    /// A function that can transcode a serialized value to JSON.
    pub dump: Option<fn(Serialized) -> Result<serde_json::Value, anyhow::Error>>,
    /// Return the arm for this type, if available.
    pub arm_unchecked: unsafe fn(*const ()) -> Option<&'static str>,
}

#[allow(dead_code)]
impl TypeInfo {
    /// Get the typeinfo for the provided type hash.
    pub(crate) fn get(typehash: u64) -> Option<&'static TypeInfo> {
        TYPE_INFO.get(&typehash).map(|v| &**v)
    }

    /// Get the typeinfo for the provided type id.
    pub(crate) fn get_by_typeid(typeid: TypeId) -> Option<&'static TypeInfo> {
        TYPE_INFO_BY_TYPE_ID.get(&typeid).map(|v| &**v)
    }

    /// Get the typeinfo for the provided type.
    pub(crate) fn of<T: ?Sized + 'static>() -> Option<&'static TypeInfo> {
        Self::get_by_typeid(TypeId::of::<T>())
    }

    pub(crate) fn typename(&self) -> &'static str {
        (self.typename)()
    }
    pub(crate) fn typehash(&self) -> u64 {
        (self.typehash)()
    }
    pub(crate) fn typeid(&self) -> TypeId {
        (self.typeid)()
    }
    pub(crate) fn port(&self) -> u64 {
        (self.port)()
    }
    pub(crate) fn dump(&self, data: Serialized) -> Result<serde_json::Value, anyhow::Error> {
        if let Some(dump) = self.dump {
            (dump)(data)
        } else {
            anyhow::bail!("binary does not have dumper for {}", self.typehash())
        }
    }
    pub(crate) unsafe fn arm_unchecked(&self, value: *const ()) -> Option<&'static str> {
        // SAFETY: This isn't safe, we're passing it on.
        unsafe { (self.arm_unchecked)(value) }
    }
}

inventory::collect!(TypeInfo);

/// Type infos for all types that have been linked into the binary, keyed by typehash.
static TYPE_INFO: LazyLock<HashMap<u64, &'static TypeInfo>> = LazyLock::new(|| {
    inventory::iter::<TypeInfo>()
        .map(|entry| (entry.typehash(), entry))
        .collect()
});

/// Type infos for all types that have been linked into the binary, keyed by typeid.
static TYPE_INFO_BY_TYPE_ID: LazyLock<HashMap<std::any::TypeId, &'static TypeInfo>> =
    LazyLock::new(|| {
        TYPE_INFO
            .values()
            .map(|info| (info.typeid(), &**info))
            .collect()
    });

/// The encoding used for a serialized value.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
enum SerializedEncoding {
    Bincode,
    Json,
}

/// Represents a serialized value, wrapping the underlying serialization
/// and deserialization details, while ensuring that we pass correctly-serialized
/// message throughout the system.
///
/// Currently, Serialized passes through to bincode, but in the future we may include
/// content-encoding information to allow for other codecs as well.
#[derive(Clone, Serialize, Deserialize, PartialEq)]
pub struct Serialized {
    encoding: SerializedEncoding,

    /// The encoded data for the serialized value.
    #[serde(with = "serde_bytes")]
    data: Vec<u8>,
    /// The typehash of the serialized value, if available. This is used to provide
    /// typed introspection of the value.
    typehash: Option<u64>,
}

impl std::fmt::Debug for Serialized {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Serialized({})", HexFmt(self.data.as_slice()),)
    }
}

impl std::fmt::Display for Serialized {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.dump() {
            Ok(value) => {
                // unwrap okay, self.dump() would return Err otherwise.
                let typename = self.typename().unwrap();
                // take the basename of the type (e.g. "foo::bar::baz" -> "baz")
                let basename = typename.split("::").last().unwrap_or(typename);
                write!(f, "{}{}", basename, JsonFmt(&value))
            }
            Err(_) => write!(f, "{}", HexFmt(self.data.as_slice())),
        }
    }
}

impl Serialized {
    /// Construct a new serialized value by serializing the provided T-typed value.
    pub fn serialize<T: Serialize + Named>(value: &T) -> Result<Self, bincode::Error> {
        Ok(Self {
            encoding: SerializedEncoding::Bincode,
            data: bincode::serialize(value)?,
            typehash: Some(T::typehash()),
        })
    }

    /// Construct a new anonymous (unnamed) serialized value by serializing the provided T-typed value.
    pub fn serialize_anon<T: Serialize>(value: &T) -> Result<Self, bincode::Error> {
        Ok(Self {
            encoding: SerializedEncoding::Bincode,
            data: bincode::serialize(value)?,
            typehash: None,
        })
    }

    /// Deserialize a value to the provided type T.
    pub fn deserialized<T: DeserializeOwned>(&self) -> Result<T, anyhow::Error> {
        match self.encoding {
            SerializedEncoding::Bincode => {
                bincode::deserialize(&self.data).map_err(anyhow::Error::from)
            }
            SerializedEncoding::Json => {
                serde_json::from_slice(&self.data).map_err(anyhow::Error::from)
            }
        }
    }

    /// Transcode the serialized value to JSON. This operation will succeed if the type hash
    /// is embedded in the value, and the corresponding type is available in this binary.
    pub fn transcode_to_json(self) -> Result<Self, Self> {
        match self.encoding {
            SerializedEncoding::Bincode => {
                let json_value = match self.dump() {
                    Ok(json_value) => json_value,
                    Err(_) => return Err(self),
                };
                let json_data = match serde_json::to_vec(&json_value) {
                    Ok(json_data) => json_data,
                    Err(_) => return Err(self),
                };
                Ok(Self {
                    encoding: SerializedEncoding::Json,
                    data: json_data,
                    typehash: self.typehash,
                })
            }
            SerializedEncoding::Json => Ok(self),
        }
    }

    /// Dump the Serialized message into a JSON value. This will succeed if: 1) the typehash is embedded
    /// in the serialized value; 2) the named type is linked into the binary.
    pub fn dump(&self) -> Result<serde_json::Value, anyhow::Error> {
        match self.encoding {
            SerializedEncoding::Bincode => {
                let Some(typehash) = self.typehash() else {
                    anyhow::bail!("serialized value does not contain a typehash");
                };
                let Some(typeinfo) = TYPE_INFO.get(&typehash) else {
                    anyhow::bail!("binary does not have typeinfo for {}", typehash);
                };
                typeinfo.dump(self.clone())
            }
            SerializedEncoding::Json => {
                serde_json::from_slice(&self.data).map_err(anyhow::Error::from)
            }
        }
    }

    /// The typehash of the serialized value, if available.
    pub fn typehash(&self) -> Option<u64> {
        self.typehash
    }

    /// The typename of the serialized value, if available.
    pub fn typename(&self) -> Option<&'static str> {
        self.typehash
            .and_then(|typehash| TYPE_INFO.get(&typehash).map(|typeinfo| typeinfo.typename()))
    }
}

const MAX_BYTE_PREVIEW_LENGTH: usize = 8;

fn display_bytes_as_hash(f: &mut impl std::fmt::Write, bytes: &[u8]) -> std::fmt::Result {
    let hash = crc32fast::hash(bytes);
    write!(f, "CRC:{:x}", hash)?;
    // Implementing in this way lets us print without allocating a new intermediate string.
    for &byte in bytes.iter().take(MAX_BYTE_PREVIEW_LENGTH) {
        write!(f, " {:x}", byte)?;
    }
    if bytes.len() > MAX_BYTE_PREVIEW_LENGTH {
        write!(f, " [...{} bytes]", bytes.len() - MAX_BYTE_PREVIEW_LENGTH)?;
    }
    Ok(())
}

/// Formats a binary slice as hex when its display function is called.
pub struct HexFmt<'a>(pub &'a [u8]);

impl<'a> std::fmt::Display for HexFmt<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // calculate a 2 byte checksum to prepend to the message
        display_bytes_as_hash(f, self.0)
    }
}

/// Formats a JSON value for display, printing all keys but
/// truncating and displaying a hash if the content is too long.
pub struct JsonFmt<'a>(pub &'a serde_json::Value);

const MAX_JSON_VALUE_DISPLAY_LENGTH: usize = 8;

impl<'a> std::fmt::Display for JsonFmt<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        /// Truncate the input string to MAX_JSON_VALUE_DISPLAY_LENGTH and append
        /// the truncated hash of the full value for easy comparison.
        fn truncate_and_hash(value_str: &str) -> String {
            let truncated_str = &value_str[..MAX_JSON_VALUE_DISPLAY_LENGTH];
            let mut result = truncated_str.to_string();
            result.push_str(&format!("[...{} chars] ", value_str.len()));
            display_bytes_as_hash(&mut result, value_str.as_bytes()).unwrap();
            result
        }

        /// Recursively truncate a serde_json::Value object.
        fn truncate_json_values(value: &serde_json::Value) -> serde_json::Value {
            match value {
                serde_json::Value::String(s) => {
                    if s.len() > MAX_JSON_VALUE_DISPLAY_LENGTH {
                        serde_json::Value::String(truncate_and_hash(s))
                    } else {
                        value.clone()
                    }
                }
                serde_json::Value::Array(arr) => {
                    let array_str = serde_json::to_string(arr).unwrap();
                    if array_str.len() > MAX_JSON_VALUE_DISPLAY_LENGTH {
                        serde_json::Value::String(truncate_and_hash(&array_str))
                    } else {
                        value.clone()
                    }
                }
                serde_json::Value::Object(obj) => {
                    let truncated_obj: serde_json::Map<_, _> = obj
                        .iter()
                        .map(|(k, v)| (k.clone(), truncate_json_values(v)))
                        .collect();
                    serde_json::Value::Object(truncated_obj)
                }
                _ => value.clone(),
            }
        }

        let truncated = truncate_json_values(self.0);
        write!(f, "{}", truncated)
    }
}

#[cfg(test)]
mod tests {

    use serde::Deserialize;
    use serde::Serialize;

    use super::*;
    use crate as hyperactor; // for macros
    use crate::Named;

    #[derive(Named)]
    #[named(dump = false)]
    struct TestStruct;

    #[test]
    fn test_names() {
        assert_eq!(String::typename(), "String");
        assert_eq!(Option::<String>::typename(), "Option<String>");
        assert_eq!(Vec::<String>::typename(), "Vec<String>");
        assert_eq!(Vec::<Vec::<String>>::typename(), "Vec<Vec<String>>");
        assert_eq!(
            Vec::<Vec::<Vec::<String>>>::typename(),
            "Vec<Vec<Vec<String>>>"
        );
        assert_eq!(
            <(u64, String, Option::<isize>)>::typename(),
            "(u64, String, Option<isize>)"
        );
        assert_eq!(
            TestStruct::typename(),
            "hyperactor::data::tests::TestStruct"
        );
        assert_eq!(
            Vec::<TestStruct>::typename(),
            "Vec<hyperactor::data::tests::TestStruct>"
        );
    }

    #[test]
    fn test_ports() {
        assert_eq!(String::typehash(), 3947244799002047352u64);
        assert_eq!(String::port(), 13170616835856823160u64);
        assert_ne!(
            Vec::<Vec::<Vec::<String>>>::typehash(),
            Vec::<Vec::<Vec::<Vec::<String>>>>::typehash(),
        );
    }

    #[derive(Named, Serialize, Deserialize)]
    struct TestDumpStruct {
        a: String,
        b: u64,
        c: Option<i32>,
    }

    #[test]
    fn test_dump_struct() {
        let data = TestDumpStruct {
            a: "hello".to_string(),
            b: 1234,
            c: Some(5678),
        };
        let serialized = Serialized::serialize(&data).unwrap();
        let serialized_json = serialized.clone().transcode_to_json().unwrap();

        assert_eq!(serialized.encoding, SerializedEncoding::Bincode);
        assert_eq!(serialized_json.encoding, SerializedEncoding::Json);

        let json_string = String::from_utf8(serialized_json.data.clone()).unwrap();
        // The serialized data for JSON is just the (compact) JSON string.
        assert_eq!(json_string, "{\"a\":\"hello\",\"b\":1234,\"c\":5678}");

        for serialized in [serialized, serialized_json] {
            // Note, at this point, serialized has no knowledge other than its embedded typehash.

            assert_eq!(
                serialized.typename(),
                Some("hyperactor::data::tests::TestDumpStruct")
            );

            let json = serialized.dump().unwrap();
            assert_eq!(
                json,
                serde_json::json!({
                    "a": "hello",
                    "b": 1234,
                    "c": 5678,
                })
            );

            assert_eq!(
                format!("{}", serialized),
                "TestDumpStruct{\"a\":\"hello\",\"b\":1234,\"c\":5678}",
            );
        }
    }

    #[test]
    fn test_arms() {
        #[derive(Named)]
        #[named(dump = false)]
        enum TestArm {
            #[allow(dead_code)]
            A(u32),
            B,
            C(),
            D {
                #[allow(dead_code)]
                a: u32,
                #[allow(dead_code)]
                b: String,
            },
        }

        assert_eq!(TestArm::A(1234).arm(), Some("A"));
        assert_eq!(TestArm::B.arm(), Some("B"));
        assert_eq!(TestArm::C().arm(), Some("C"));
        assert_eq!(
            TestArm::D {
                a: 1234,
                b: "hello".to_string()
            }
            .arm(),
            Some("D")
        );
    }

    #[test]
    fn display_hex() {
        assert_eq!(
            format!("{}", HexFmt("hello world".as_bytes())),
            "CRC:d4a1185 68 65 6c 6c 6f 20 77 6f [...3 bytes]"
        );
        assert_eq!(format!("{}", HexFmt("".as_bytes())), "CRC:0");
        assert_eq!(
            format!("{}", HexFmt("a very long string that is long".as_bytes())),
            "CRC:c7e24f62 61 20 76 65 72 79 20 6c [...23 bytes]"
        );
    }

    #[test]
    fn test_json_fmt() {
        let json_value = serde_json::json!({
            "name": "test",
            "number": 42,
            "nested": {
                "key": "value"
            }
        });
        // JSON values with short values should print normally
        assert_eq!(
            format!("{}", JsonFmt(&json_value)),
            "{\"name\":\"test\",\"nested\":{\"key\":\"value\"},\"number\":42}",
        );

        let empty_json = serde_json::json!({});
        assert_eq!(format!("{}", JsonFmt(&empty_json)), "{}");

        let simple_array = serde_json::json!([1, 2, 3]);
        assert_eq!(format!("{}", JsonFmt(&simple_array)), "[1,2,3]");

        // JSON values with very long strings should be truncated
        let long_string_json = serde_json::json!({
            "long_string": "a".repeat(MAX_JSON_VALUE_DISPLAY_LENGTH * 5)
        });
        assert_eq!(
            format!("{}", JsonFmt(&long_string_json)),
            "{\"long_string\":\"aaaaaaaa[...40 chars] CRC:c95b8a25 61 61 61 61 61 61 61 61 [...32 bytes]\"}"
        );

        // JSON values with very long arrays should be truncated
        let long_array_json =
            serde_json::json!((1..=(MAX_JSON_VALUE_DISPLAY_LENGTH + 4)).collect::<Vec<_>>());
        assert_eq!(
            format!("{}", JsonFmt(&long_array_json)),
            "\"[1,2,3,4[...28 chars] CRC:e5c881af 5b 31 2c 32 2c 33 2c 34 [...20 bytes]\""
        );

        // Test for truncation within nested blocks
        let nested_json = serde_json::json!({
            "simple_number": 42,
            "simple_bool": true,
            "outer": {
                "long_string": "a".repeat(MAX_JSON_VALUE_DISPLAY_LENGTH + 10),
                "long_array": (1..=(MAX_JSON_VALUE_DISPLAY_LENGTH + 4)).collect::<Vec<_>>(),
                "inner": {
                    "simple_value": "short",
                }
            }
        });
        println!("{}", JsonFmt(&nested_json));
        assert_eq!(
            format!("{}", JsonFmt(&nested_json)),
            "{\"outer\":{\"inner\":{\"simple_value\":\"short\"},\"long_array\":\"[1,2,3,4[...28 chars] CRC:e5c881af 5b 31 2c 32 2c 33 2c 34 [...20 bytes]\",\"long_string\":\"aaaaaaaa[...18 chars] CRC:b8ac0e31 61 61 61 61 61 61 61 61 [...10 bytes]\"},\"simple_bool\":true,\"simple_number\":42}",
        );
    }
}
