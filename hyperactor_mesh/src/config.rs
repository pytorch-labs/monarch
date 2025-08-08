/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use hyperactor::attrs::declare_attrs;

declare_attrs! {
    pub attr ROUTER_CONFIG_NO_GLOBAL_FALLBACK: bool = false;
}

/// Extend hyperactor global config with mesh-specific settings.
pub fn init_global_from_env() {
    hyperactor::config::global::init_from_env();

    let config_lock = hyperactor::config::global::lock();

    if std::env::var("HYPERACTOR_ROUTER_CONFIG_NO_GLOBAL_FALLBACK").is_ok() {
        let guard = config_lock.override_key(ROUTER_CONFIG_NO_GLOBAL_FALLBACK, true);
        std::mem::forget(guard);
    }
}

#[cfg(test)]
mod tests {
    use hyperactor::config;

    use super::*;

    #[test]
    fn test_init_global_from_env() {
        config::global::reset_to_defaults();
        assert_eq!(
            config::global::get(hyperactor::config::MESSAGE_DELIVERY_TIMEOUT),
            std::time::Duration::from_secs(30)
        );
        assert!(!config::global::get(ROUTER_CONFIG_NO_GLOBAL_FALLBACK));

        // SAFETY: We rely on no concurrent access here.
        unsafe {
            std::env::set_var("HYPERACTOR_ROUTER_CONFIG_NO_GLOBAL_FALLBACK", "1");
        }
        init_global_from_env();
        assert!(config::global::get(ROUTER_CONFIG_NO_GLOBAL_FALLBACK));

        // SAFETY: We rely on no concurrent access here.
        unsafe {
            std::env::remove_var("HYPERACTOR_ROUTER_CONFIG_NO_GLOBAL_FALLBACK");
        }
        config::global::reset_to_defaults();
    }
}
