// Clippy can't see through quote! to use of proc-macro2
#![allow(unused_crate_dependencies)]

extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;

/// Parse a compact selection expression into a [`Selection`]. See
/// [`selection::parse`] for syntax documentation.
#[proc_macro]
pub fn sel(input: TokenStream) -> TokenStream {
    match ndslice::selection::token_parser::parse_tokens(input.into()) {
        Ok(selection) => {
            let tokens = ndslice::selection::token_parser::selection_to_tokens(&selection);
            quote!(#tokens).into()
        }
        Err(e) => {
            let msg = format!("sel! parse failed: {}", e);
            quote!(compile_error!(#msg)).into()
        }
    }
}
