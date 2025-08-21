/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#![allow(dead_code)]
pub mod castable;
pub mod export;

use std::fmt::Debug;

use anyhow::Result;
use async_trait::async_trait;
use hyperactor::Actor;
use hyperactor::Context;
use hyperactor::HandleClient;
use hyperactor::Handler;
use hyperactor::Named;
use hyperactor::OncePortRef;
use hyperactor::RefClient;
use hyperactor::forward;
use hyperactor::instrument;
use hyperactor::instrument_infallible;
use hyperactor::observe;
use serde::Deserialize;
use serde::Serialize;

#[derive(Handler, Debug, Named)]
enum ShoppingList {
    // Oneway messages dispatch messages asynchronously, with no reply.
    Add(String),
    Remove {
        item: String,
    }, // both tuple and struct variants are supported.

    // Call messages dispatch a request, expecting a reply to the
    // provided port, which must be in the last position.
    Exists(String, #[reply] OncePortRef<bool>),

    // Tests macro hygience. We use 'result' as a keyword in the implementation.
    Clobber {
        arg: String,
        #[reply]
        result: OncePortRef<bool>,
    },
}

#[derive(Handler, HandleClient, RefClient, Debug, Serialize, Deserialize, Named)]
#[log_level(info)]
enum TestVariantForms {
    OneWayStruct {
        a: u64,
        b: u64,
    },

    #[log_level(error)]
    OneWayTuple(u64, u64),

    OneWayTupleNoArgs(),

    OneWayStructNoArgs {},

    CallStruct {
        a: u64,
        #[reply]
        b: OncePortRef<u64>,
    },

    CallTuple(u64, #[reply] OncePortRef<u64>),

    CallTupleNoArgs(#[reply] OncePortRef<u64>),

    CallStructNoArgs {
        #[reply]
        a: OncePortRef<u64>,
    },
}

#[instrument(fields(name = 4))]
async fn yolo() -> Result<i32, i32> {
    Ok(10)
}

#[instrument_infallible(fields(crow = "black"))]
async fn yeet() -> String {
    String::from("cawwww")
}

#[test]
fn basic() {
    // nothing, just checks whether this file will compile
}

#[derive(Debug, Handler, HandleClient)]
enum GenericArgMessage<A: Clone + Sync + Send + Debug + 'static> {
    Variant(A),
}

#[derive(Debug)]
struct GenericArgActor {}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct GenericArgParams {}

#[async_trait]
impl Actor for GenericArgActor {
    type Params = GenericArgParams;

    async fn new(_params: Self::Params) -> Result<Self> {
        Ok(Self {})
    }
}

#[async_trait]
#[forward(GenericArgMessage<usize>)]
impl GenericArgMessageHandler<usize> for GenericArgActor {
    async fn variant(&mut self, _cx: &Context<Self>, _val: usize) -> Result<()> {
        Ok(())
    }
}

#[derive(Actor, Default, Debug)]
struct DefaultActorTest {
    value: u64,
}

static_assertions::assert_impl_all!(DefaultActorTest: Actor);

#[derive(Actor, Default, Debug)]
#[actor(passthrough)]
struct PassthroughActorTest {
    value: u64,
}

static_assertions::assert_impl_all!(PassthroughActorTest: Actor);
static_assertions::assert_type_eq_all!(
    <PassthroughActorTest as hyperactor::Actor>::Params,
    PassthroughActorTest
);

#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::fmt;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::Ordering;

    use super::*;

    // Define a custom error type for testing
    #[derive(Debug)]
    struct CustomError {
        message: String,
    }

    impl CustomError {
        fn new(message: &str) -> Self {
            Self {
                message: message.to_string(),
            }
        }
    }

    impl fmt::Display for CustomError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.message)
        }
    }

    impl Error for CustomError {}

    // Example using the basic instrument macro
    #[observe("custom_module")]
    async fn process_data(data: &str) -> Result<String, CustomError> {
        // Simulate some processing time
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        if data.contains("error") {
            return Err(CustomError::new("Data contains error"));
        }

        Ok(format!("Processed: {}", data))
    }

    // Example using the instrument_with macro with custom metric names and attributes
    #[observe("custom_module")]
    async fn handle_api_request(
        endpoint: &str,
        method: &str,
        payload: &str,
    ) -> Result<String, CustomError> {
        // Simulate API processing time
        tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;

        if endpoint.contains("invalid") {
            return Err(CustomError::new("Invalid endpoint"));
        }

        if method == "POST" && payload.is_empty() {
            return Err(CustomError::new("Empty payload for POST request"));
        }

        Ok(format!("Response from {} {}", method, endpoint))
    }

    // Example of a function that doesn't return a Result but might throw exceptions
    #[observe("custom_module")]
    async fn log_activity(activity: &str) {
        // Simulate logging activity
        tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
    }

    // Example of a function that will panic
    #[observe("custom_module")]
    async fn risky_operation(should_panic: bool) {
        // Simulate operation
        tokio::time::sleep(tokio::time::Duration::from_millis(3)).await;

        if should_panic {
            panic!("Something went wrong!");
        }
    }

    #[tokio::test]
    async fn test_process_data_success() {
        let result = process_data("normal data").await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Processed: normal data");
    }

    #[tokio::test]
    async fn test_process_data_error() {
        let result = process_data("data with error").await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().to_string(), "Data contains error");
    }

    #[tokio::test]
    async fn test_handle_api_request_success() {
        // Test GET request
        let result = handle_api_request("/users", "GET", "").await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Response from GET /users");

        // Test POST request with payload
        let result = handle_api_request("/posts", "POST", "content").await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Response from POST /posts");
    }

    #[tokio::test]
    async fn test_handle_api_request_errors() {
        // Test invalid endpoint
        let result = handle_api_request("/invalid/path", "GET", "").await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().to_string(), "Invalid endpoint");

        // Test empty payload for POST
        let result = handle_api_request("/posts", "POST", "").await;
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "Empty payload for POST request"
        );
    }

    #[tokio::test]
    async fn test_log_activity() {
        // This should complete without errors
        log_activity("User login").await;
    }

    #[tokio::test]
    async fn test_risky_operation_success() {
        // This should complete without errors
        risky_operation(false).await;
    }

    #[tokio::test]
    #[should_panic(expected = "Something went wrong!")]
    async fn test_risky_operation_panic() {
        risky_operation(true).await;
    }
}
