use proc_macro::TokenStream;
use quote::quote;
use syn::Expr;
use syn::ItemFn;
use syn::Lit;
use syn::MetaNameValue;
use syn::parse_macro_input;

/// A test macro that wraps tokio::test and adds a configurable timeout.
///
/// # Examples
///
/// ```rust
/// #[async_timed_test(timeout_secs = 5)]
/// async fn my_test() {
///     // Test that should complete within 5 seconds
///     tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
/// }
/// ```
#[proc_macro_attribute]
pub fn async_timed_test(attr: TokenStream, input: TokenStream) -> TokenStream {
    let attr = parse_macro_input!(attr as MetaNameValue);
    let input_fn = parse_macro_input!(input as ItemFn);

    if !attr.path.is_ident("timeout_secs") {
        return TokenStream::from(
            syn::Error::new_spanned(attr.path, "only timeout_secs allowed as argument")
                .to_compile_error(),
        );
    }
    let timeout_secs = match attr.value {
        Expr::Lit(ref lit) => match &lit.lit {
            Lit::Int(val) => val.base10_parse::<u64>().unwrap(),
            _ => {
                return TokenStream::from(
                    syn::Error::new_spanned(
                        attr.value,
                        "unexpected value for timeout_secs, please pass an integer literal",
                    )
                    .to_compile_error(),
                );
            }
        },
        _ => {
            return TokenStream::from(
                syn::Error::new_spanned(
                    attr.value,
                    "unexpected value for timeout_secs, please pass an integer literal",
                )
                .to_compile_error(),
            );
        }
    };

    let fn_block = &input_fn.block;
    let fn_attrs = &input_fn.attrs;
    let fn_vis = &input_fn.vis;
    let sig = &input_fn.sig;

    if sig.asyncness.is_none() {
        return TokenStream::from(
            syn::Error::new_spanned(sig, "test function must be async").to_compile_error(),
        );
    }

    let output = quote! {
        // Use the multithread runtime so that a hanging task will not block the
        // timeout from firing.
        #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
        #(#fn_attrs)*
        #fn_vis #sig {
            use tokio::time::{timeout, Duration};

            let test_future = async #fn_block;

            match timeout(Duration::from_secs(#timeout_secs), test_future).await {
                Ok(result) => result,
                Err(_) => panic!("test timed out after {} seconds", #timeout_secs),
            }
        }
    };

    output.into()
}
