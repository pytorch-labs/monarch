/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Configuration for Monarch Hyperactor.
//!
//! This module provides monarch-specific configuration attributes that extend
//! the base hyperactor configuration system.

use hyperactor::attrs::declare_attrs;

// Declare monarch-specific configuration keys
declare_attrs! {
    /// Use a single asyncio runtime for all Python actors, rather than one per actor
    /// Note: use shared runtime if you have a lot of Python actors, otherwise too many threads can be spawned
    pub attr SHARED_ASYNCIO_RUNTIME: bool = true;
}
