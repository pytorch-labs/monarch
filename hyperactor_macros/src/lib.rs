/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#![feature(proc_macro_def_site)]

extern crate proc_macro;

use convert_case::Case;
use convert_case::Casing;
use indoc::indoc;
use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::format_ident;
use quote::quote;
use syn::Attribute;
use syn::Data;
use syn::DataEnum;
use syn::DeriveInput;
use syn::Expr;
use syn::ExprLit;
use syn::Field;
use syn::Fields;
use syn::Ident;
use syn::ItemFn;
use syn::ItemImpl;
use syn::Lit;
use syn::Meta;
use syn::MetaNameValue;
use syn::Token;
use syn::Type;
use syn::bracketed;
use syn::parse::Parse;
use syn::parse::ParseStream;
use syn::parse_macro_input;
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;

const REPLY_VARIANT_ERROR: &str = indoc! {r#"
`call` message expects a typed `OncePortRef` or `OncePortHandle` argument in the last position

= help: use `MyCall(Arg1Type, Arg2Type, .., OncePortRef<ReplyType>)`
= help: use `MyCall(Arg1Type, Arg2Type, .., OncePortHandle<ReplyType>)`
"#};

const REPLY_USAGE_ERROR: &str = indoc! {r#"
`call` message expects at most one `reply` argument

= help: use `MyCall(Arg1Type, Arg2Type, .., #[reply] OncePortRef<ReplyType>)`
= help: use `MyCall(Arg1Type, Arg2Type, .., #[reply] OncePortHandle<ReplyType>)`
"#};

enum FieldFlag {
    None,
    Reply,
}

/// Represents a variant of an enum.
enum Variant {
    /// A named variant (i.e., `MyVariant { .. }`).
    Named {
        enum_name: Ident,
        name: Ident,
        field_names: Vec<Ident>,
        field_types: Vec<Type>,
        field_flags: Vec<FieldFlag>,
    },
    /// An anonymous variant (i.e., `MyVariant(..)`).
    Anon {
        enum_name: Ident,
        name: Ident,
        field_types: Vec<Type>,
        field_flags: Vec<FieldFlag>,
    },
}

impl Variant {
    /// The number of fields in the variant.
    fn len(&self) -> usize {
        self.field_types().len()
    }

    /// The name of the enum containing the variant.
    fn enum_name(&self) -> &Ident {
        match self {
            Variant::Named { enum_name, .. } => enum_name,
            Variant::Anon { enum_name, .. } => enum_name,
        }
    }

    /// The name of the variant itself.
    fn name(&self) -> &Ident {
        match self {
            Variant::Named { name, .. } => name,
            Variant::Anon { name, .. } => name,
        }
    }

    /// The snake_name of the variant itself.
    fn snake_name(&self) -> Ident {
        Ident::new(
            &self.name().to_string().to_case(Case::Snake),
            self.name().span(),
        )
    }

    /// The variant's qualified name.
    fn qualified_name(&self) -> proc_macro2::TokenStream {
        let enum_name = self.enum_name();
        let name = self.name();
        quote! { #enum_name::#name }
    }

    /// Names of the fields in the variant. Anonymous variants are named
    /// according to their position in the argument list.
    fn field_names(&self) -> Vec<Ident> {
        match self {
            Variant::Named { field_names, .. } => field_names.clone(),
            Variant::Anon { field_types, .. } => (0usize..field_types.len())
                .map(|idx| format_ident!("arg{}", idx))
                .collect(),
        }
    }

    /// The types of the fields int the variant.
    fn field_types(&self) -> &Vec<Type> {
        match self {
            Variant::Named { field_types, .. } => field_types,
            Variant::Anon { field_types, .. } => field_types,
        }
    }

    /// Return the field flags for this variant.
    fn field_flags(&self) -> &Vec<FieldFlag> {
        match self {
            Variant::Named { field_flags, .. } => field_flags,
            Variant::Anon { field_flags, .. } => field_flags,
        }
    }

    /// The constructor for the variant, using the field names directly.
    fn constructor(&self) -> proc_macro2::TokenStream {
        let qualified_name = self.qualified_name();
        let field_names = self.field_names();
        match self {
            Variant::Named { .. } => quote! { #qualified_name { #(#field_names),* } },
            Variant::Anon { .. } => quote! { #qualified_name(#(#field_names),*) },
        }
    }
}

/// Represents a message that can be sent to a handler, each message is associated with
/// a variant.
#[allow(clippy::large_enum_variant)]
enum Message {
    /// A call message is a request-response message, the last argument is
    /// a [`hyperactor::OncePortRef`] or [`hyperactor::OncePortHandle`].
    Call {
        variant: Variant,
        /// Tells whether the reply argument is a handle.
        reply_port_is_handle: bool,
        /// The underlying return type (i.e., the type of the reply port).
        return_type: Type,
        /// the log level for generated instrumentation for handlers of this message.
        log_level: Option<Ident>,
    },
    OneWay {
        variant: Variant,
        /// the log level for generated instrumentation for handlers of this message.
        log_level: Option<Ident>,
    },
}

impl Message {
    fn new(span: Span, variant: Variant, log_level: Option<Ident>) -> Result<Self, syn::Error> {
        match &variant
            .field_flags()
            .iter()
            .zip(variant.field_types())
            .filter_map(|(flag, ty)| match flag {
                FieldFlag::Reply => Some(ty),
                FieldFlag::None => None,
            })
            .collect::<Vec<&Type>>()[..]
        {
            [] => Ok(Self::OneWay { variant, log_level }),
            [reply_port_ty] => {
                let syn::Type::Path(type_path) = reply_port_ty else {
                    return Err(syn::Error::new(span, REPLY_VARIANT_ERROR));
                };
                let Some(last_segment) = type_path.path.segments.last() else {
                    return Err(syn::Error::new(span, REPLY_VARIANT_ERROR));
                };
                if last_segment.ident != "OncePortRef" && last_segment.ident != "OncePortHandle" {
                    return Err(syn::Error::new_spanned(last_segment, REPLY_VARIANT_ERROR));
                }
                let syn::PathArguments::AngleBracketed(args) = &last_segment.arguments else {
                    return Err(syn::Error::new_spanned(last_segment, REPLY_VARIANT_ERROR));
                };
                let Some(syn::GenericArgument::Type(return_ty)) = args.args.first() else {
                    return Err(syn::Error::new_spanned(&args.args, REPLY_VARIANT_ERROR));
                };
                let reply_port_is_handle = last_segment.ident == "OncePortHandle";
                let return_type = return_ty.clone();
                Ok(Self::Call {
                    variant,
                    reply_port_is_handle,
                    return_type,
                    log_level,
                })
            }
            _ => Err(syn::Error::new(span, REPLY_USAGE_ERROR)),
        }
    }

    /// The arguments of this message.
    fn args(&self) -> Vec<(Ident, Type)> {
        match self {
            Message::Call { variant, .. } => variant
                .field_names()
                .into_iter()
                .zip(variant.field_types().clone())
                .take(variant.len() - 1)
                .collect(),
            Message::OneWay { variant, .. } => variant
                .field_names()
                .into_iter()
                .zip(variant.field_types().clone())
                .collect(),
        }
    }

    fn variant(&self) -> &Variant {
        match self {
            Message::Call { variant, .. } => variant,
            Message::OneWay { variant, .. } => variant,
        }
    }

    fn reply_port_position(&self) -> Option<usize> {
        self.variant()
            .field_flags()
            .iter()
            .position(|flag| matches!(flag, FieldFlag::Reply))
    }

    /// The reply port argument of this message.
    fn reply_port_arg(&self) -> Option<(Ident, Type)> {
        match self {
            Message::Call { variant, .. } => {
                let pos = self.reply_port_position()?;
                Some((
                    variant.field_names()[pos].clone(),
                    variant.field_types()[pos].clone(),
                ))
            }
            Message::OneWay { .. } => None,
        }
    }
}

fn parse_log_level(attrs: &[Attribute]) -> Result<Option<Ident>, syn::Error> {
    let level: Option<String> = match attrs.iter().find(|attr| attr.path().is_ident("log_level")) {
        Some(attr) => {
            let Ok(meta) = attr.meta.require_list() else {
                return Err(syn::Error::new(
                    Span::call_site(),
                    indoc! {"
                            `log_level` attribute must specify level. Supported levels = error, warn, info, debug, trace

                            = help use `#[log_level(info)]` or `#[log_level(error)]`
                        "},
                ));
            };
            let parsed = meta.parse_args_with(Punctuated::<Ident, Token![,]>::parse_terminated)?;
            if parsed.len() != 1 {
                return Err(syn::Error::new(
                    Span::call_site(),
                    indoc! {"
                            `log_level` attribute must specify exactly one level

                            = help use `#[log_level(warn)]` or `#[log_level(info)]`
                        "},
                ));
            };
            Some(parsed.first().unwrap().to_string())
        }
        None => None,
    };

    if level.is_none() {
        return Ok(None);
    }
    let level = level.unwrap();

    match level.as_str() {
        "error" | "warn" | "info" | "debug" | "trace" => {}
        _ => {
            return Err(syn::Error::new(
                Span::call_site(),
                indoc! {"
                            `log_level` attribute must be one of 'error, warn, info, debug, trace'

                            = help use `#[log_level(warn)]` or `#[log_level(info)]`
                        "},
            ));
        }
    }

    Ok(Some(Ident::new(
        level.to_ascii_uppercase().as_str(),
        Span::call_site(),
    )))
}

fn parse_field_flag(field: &Field) -> FieldFlag {
    for attr in field.attrs.iter() {
        match &attr.meta {
            syn::Meta::Path(path) if path.is_ident("reply") => return FieldFlag::Reply,
            _ => {}
        }
    }
    FieldFlag::None
}

/// Parse a message enum into its constituent messages.
fn parse_message_enum(input: DeriveInput) -> Result<Vec<Message>, syn::Error> {
    let variants = if let Data::Enum(data_enum) = &input.data {
        &data_enum.variants
    } else {
        return Err(syn::Error::new_spanned(
            input,
            "handlers can only be derived for enums",
        ));
    };

    let mut messages = Vec::new();

    for variant in variants {
        let name = variant.ident.clone();
        let attrs = &variant.attrs;

        let message_variant = match &variant.fields {
            syn::Fields::Unnamed(fields_) => Variant::Anon {
                enum_name: input.ident.clone(),
                name,
                field_types: fields_
                    .unnamed
                    .iter()
                    .map(|field| field.ty.clone())
                    .collect(),
                field_flags: fields_.unnamed.iter().map(parse_field_flag).collect(),
            },
            syn::Fields::Named(fields_) => Variant::Named {
                enum_name: input.ident.clone(),
                name,
                field_names: fields_
                    .named
                    .iter()
                    .map(|field| field.ident.clone().unwrap())
                    .collect(),
                field_types: fields_.named.iter().map(|field| field.ty.clone()).collect(),
                field_flags: fields_.named.iter().map(parse_field_flag).collect(),
            },
            _ => {
                return Err(syn::Error::new_spanned(
                    variant,
                    indoc! {r#"
                      `Handler` currently only supports named or tuple struct variants

                      = help use `MyCall(Arg1Type, Arg2Type, ..)`,
                      = help use `MyCall { arg1: Arg1Type, arg2: Arg2Type, .. }`,
                      = help use `MyCall(Arg1Type, Arg2Type, .., #[reply] OncePortRef<ReplyType>)`
                      = help use `MyCall { arg1: Arg1Type, arg2: Arg2Type, .., reply: #[reply] OncePortRef<ReplyType>)`
                      = help use `MyCall(Arg1Type, Arg2Type, .., #[reply] OncePortHandle<ReplyType>)`
                      = help use `MyCall { arg1: Arg1Type, arg2: Arg2Type, .., reply: #[reply] OncePortHandle<ReplyType>)`
                    "#},
                ));
            }
        };
        let log_level = parse_log_level(attrs)?;

        messages.push(Message::new(
            variant.fields.span(),
            message_variant,
            log_level,
        )?);
    }

    Ok(messages)
}

/// Derive a custom handler trait for given an enum containing tuple
/// structs.  The handler trait defines a method corresponding
/// to each of the enum's variants, and a `handle` function
/// that dispatches messages to the correct method.  The macro
/// supports two messaging patterns: "call" and "oneway". A call is a
/// request-response message; a [`hyperactor::mailbox::OncePortRef`] or
/// [`hyperactor::mailbox::OncePortHandle`] in the last position is used
/// to send the return value.
///
/// The macro also derives a client trait that can be automatically implemented
/// by specifying [`HandleClient`] for `ActorHandle<Actor>` and [`RefClient`]
/// for `ActorRef<Actor>` accordingly. We require two implementations because
/// not `ActorRef`s require that its message type is serializable.
///
/// The associated [`hyperactor_macros::handler`] macro can be used to add
/// a dispatching handler directly to an [`hyperactor::Actor`].
///
/// # Example
///
/// The following example creates a "shopping list" actor responsible for
/// maintaining a shopping list.
///
/// ```
/// use std::collections::HashSet;
/// use std::time::Duration;
///
/// use async_trait::async_trait;
/// use hyperactor::Actor;
/// use hyperactor::HandleClient;
/// use hyperactor::Handler;
/// use hyperactor::Instance;
/// use hyperactor::Named;
/// use hyperactor::OncePortRef;
/// use hyperactor::RefClient;
/// use hyperactor::proc::Proc;
/// use serde::Deserialize;
/// use serde::Serialize;
///
/// #[derive(Handler, HandleClient, RefClient, Debug, Serialize, Deserialize, Named)]
/// enum ShoppingList {
///     // Oneway messages dispatch messages asynchronously, with no reply.
///     Add(String),
///     Remove(String),
///
///     // Call messages dispatch a request, expecting a reply to the
///     // provided port, which must be in the last position.
///     Exists(String, #[reply] OncePortRef<bool>),
///
///     List(#[reply] OncePortRef<Vec<String>>),
/// }
///
/// // Define an actor.
/// #[derive(Debug)]
/// #[hyperactor::export(
///     spawn = true,
///     handlers = [
///         ShoppingList,
///     ],
/// )]
/// struct ShoppingListActor(HashSet<String>);
///
/// #[async_trait]
/// impl Actor for ShoppingListActor {
///     type Params = ();
///
///     async fn new(_params: ()) -> Result<Self, anyhow::Error> {
///         Ok(Self(HashSet::new()))
///     }
/// }
///
/// // ShoppingListHandler is the trait generated by derive(Handler) above.
/// // We implement the trait here for the actor, defining a handler for
/// // each ShoppingList message.
/// //
/// // The `forward` attribute installs a handler that forwards messages
/// // to the `ShoppingListHandler` implementation directly. This can also
/// // be done manually:
/// //
/// // ```ignore
/// //<ShoppingListActor as ShoppingListHandler>
/// //     ::handle(self, comm, message).await
/// // ```
/// #[async_trait]
/// #[hyperactor::forward(ShoppingList)]
/// impl ShoppingListHandler for ShoppingListActor {
///     async fn add(&mut self, _this: &Instance<Self>, item: String) -> Result<(), anyhow::Error> {
///         eprintln!("insert {}", item);
///         self.0.insert(item);
///         Ok(())
///     }
///
///     async fn remove(
///         &mut self,
///         _this: &Instance<Self>,
///         item: String,
///     ) -> Result<(), anyhow::Error> {
///         eprintln!("remove {}", item);
///         self.0.remove(&item);
///         Ok(())
///     }
///
///     async fn exists(
///         &mut self,
///         _this: &Instance<Self>,
///         item: String,
///     ) -> Result<bool, anyhow::Error> {
///         Ok(self.0.contains(&item))
///     }
///
///     async fn list(&mut self, _this: &Instance<Self>) -> Result<Vec<String>, anyhow::Error> {
///         Ok(self.0.iter().cloned().collect())
///     }
/// }
///
/// #[tokio::main]
/// async fn main() -> Result<(), anyhow::Error> {
///     let mut proc = Proc::local();
///
///     // Spawn our actor, and get a handle for rank 0.
///     let shopping_list_actor: hyperactor::ActorHandle<ShoppingListActor> =
///         proc.spawn("shopping", ()).await?;
///
///     // We join the system, so that we can send messages to actors.
///     let client = proc.attach("client").unwrap();
///
///     // todo: consider making this a macro to remove the magic names
///
///     // Derive(Handler) generates client methods, which call the
///     // remote handler provided an instance (send + open capability),
///     // the destination actor, and the method arguments.
///
///     shopping_list_actor.add(&client, "milk".into()).await?;
///     shopping_list_actor.add(&client, "eggs".into()).await?;
///
///     println!(
///         "got milk? {}",
///         shopping_list_actor.exists(&client, "milk".into()).await?
///     );
///     println!(
///         "got yoghurt? {}",
///         shopping_list_actor
///             .exists(&client, "yoghurt".into())
///             .await?
///     );
///
///     shopping_list_actor.remove(&client, "milk".into()).await?;
///     println!(
///         "got milk now? {}",
///         shopping_list_actor.exists(&client, "milk".into()).await?
///     );
///
///     println!(
///         "shopping list: {:?}",
///         shopping_list_actor.list(&client).await?
///     );
///
///     let _ = proc.destroy_and_wait(Duration::from_secs(1), None).await?;
///     Ok(())
/// }
/// ```
#[proc_macro_derive(Handler, attributes(reply))]
pub fn derive_handler(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name: Ident = input.ident.clone();
    let (impl_generics, ty_generics, _) = input.generics.split_for_impl();

    let messages = match parse_message_enum(input.clone()) {
        Ok(messages) => messages,
        Err(err) => return TokenStream::from(err.to_compile_error()),
    };

    // Trait definition methods for the handler.
    let mut handler_trait_methods = Vec::new();

    // The arms of the match used in the message dispatcher.
    let mut match_arms = Vec::new();

    // Trait implemented by clients.
    let mut client_trait_methods = Vec::new();

    let global_log_level = parse_log_level(&input.attrs).ok().unwrap_or(None);

    for message in messages {
        match message {
            Message::Call {
                ref variant,
                ref reply_port_is_handle,
                ref return_type,
                ref log_level,
            } => {
                let (arg_names, arg_types): (Vec<_>, Vec<_>) = message.args().into_iter().unzip();
                let variant_name_snake = variant.snake_name();
                let enum_name = variant.enum_name();
                let _variant_qualified_name = variant.qualified_name();
                let log_level = match (&global_log_level, log_level) {
                    (_, Some(local)) => local.clone(),
                    (Some(global), None) => global.clone(),
                    _ => Ident::new("DEBUG", Span::call_site()),
                };
                let _log_level = if *reply_port_is_handle {
                    quote! {
                        tracing::Level::#log_level
                    }
                } else {
                    quote! {
                        tracing::Level::TRACE
                    }
                };
                let log_message = quote! {
                        hyperactor::metrics::MESSAGES_RECEIVED.add(1, hyperactor::kv_pairs!(
                            "rpc" => "call",
                            "actor_id" => this.self_id().to_string(),
                            "message_type" => stringify!(#enum_name),
                            "variant" => stringify!(#variant_name_snake),
                        ));
                };

                handler_trait_methods.push(quote! {
                    #[doc = "The generated handler method for this enum variant."]
                    async fn #variant_name_snake(
                        &mut self,
                        this: &hyperactor::Instance<Self>,
                        #(#arg_names: #arg_types),*)
                        -> Result<#return_type, hyperactor::anyhow::Error>;
                });

                client_trait_methods.push(quote! {
                    #[doc = "The generated client method for this enum variant."]
                    async fn #variant_name_snake(
                        &self,
                        caps: &(impl hyperactor::cap::CanSend + hyperactor::cap::CanOpenPort),
                        #(#arg_names: #arg_types),*)
                        -> Result<#return_type, hyperactor::anyhow::Error>;
                });

                let (reply_port_arg, _) = message.reply_port_arg().unwrap();
                let constructor = variant.constructor();
                let construct_result_future = quote! { use hyperactor::Message; let result = self.#variant_name_snake(this, #(#arg_names),*).await?; };
                if *reply_port_is_handle {
                    match_arms.push(quote! {
                        #constructor => {
                            #log_message
                            // TODO: should we propagate this error (to supervision), or send it back as an "RPC error"?
                            // This would require Result<Result<..., in order to handle RPC errors.
                            #construct_result_future
                            #reply_port_arg.send(result).map_err(hyperactor::anyhow::Error::from)
                        }
                    });
                } else {
                    match_arms.push(quote! {
                        #constructor => {
                            #log_message
                            // TODO: should we propagate this error (to supervision), or send it back as an "RPC error"?
                            // This would require Result<Result<..., in order to handle RPC errors.
                            #construct_result_future
                            #reply_port_arg.send(this, result).map_err(hyperactor::anyhow::Error::from)
                        }
                    });
                }
            }
            Message::OneWay {
                ref variant,
                ref log_level,
            } => {
                let (arg_names, arg_types): (Vec<_>, Vec<_>) = message.args().into_iter().unzip();
                let variant_name_snake = variant.snake_name();
                let enum_name = variant.enum_name();
                let log_level = match (&global_log_level, log_level) {
                    (_, Some(local)) => local.clone(),
                    (Some(global), None) => global.clone(),
                    _ => Ident::new("TRACE", Span::call_site()),
                };
                let _log_level = quote! {
                    tracing::Level::#log_level
                };
                let log_message = quote! {
                        hyperactor::metrics::MESSAGES_RECEIVED.add(1, hyperactor::kv_pairs!(
                            "rpc" => "call",
                            "actor_id" => this.self_id().to_string(),
                            "message_type" => stringify!(#enum_name),
                            "variant" => stringify!(#variant_name_snake),
                        ));
                        // tracing::event!(target: "message", #log_level, rpc = "call",  payload=?message, "send");
                };

                handler_trait_methods.push(quote! {
                    #[doc = "The generated handler method for this enum variant."]
                    async fn #variant_name_snake(
                        &mut self,
                        this: &hyperactor::Instance<Self>,
                        #(#arg_names: #arg_types),*)
                        -> Result<(), hyperactor::anyhow::Error>;
                });

                client_trait_methods.push(quote! {
                    #[doc = "The generated client method for this enum variant."]
                    async fn #variant_name_snake(
                        &self,
                        caps: &impl hyperactor::cap::CanSend,
                        #(#arg_names: #arg_types),*)
                        -> Result<(), hyperactor::anyhow::Error>;
                });

                let constructor = variant.constructor();

                match_arms.push(quote! {
                    #constructor => {
                        #log_message
                        self.#variant_name_snake(this, #(#arg_names),*).await
                    },
                });
            }
        }
    }

    let handler_trait_name = format_ident!("{}Handler", name);
    let client_trait_name = format_ident!("{}Client", name);

    let expanded = quote! {
        #[doc = "The custom handler trait for this message type."]
        #[hyperactor::async_trait::async_trait]
        pub trait #handler_trait_name #impl_generics: hyperactor::Actor + Send + Sync  {
            #(#handler_trait_methods)*

            #[doc = "Handle the next message."]
            async fn handle(
                &mut self,
                this: &hyperactor::Instance<Self>,
                message: #name #ty_generics,
            ) -> hyperactor::anyhow::Result<()>  {
                 // Dispatch based on message type.
                 match message {
                     #(#match_arms)*
                }
            }
        }

        #[doc = "The custom client trait for this message type."]
        #[hyperactor::async_trait::async_trait]
        pub trait #client_trait_name #impl_generics: Send + Sync  {
            #(#client_trait_methods)*
        }
    };

    TokenStream::from(expanded)
}

/// Derives a client implementation on `ActorHandle<Actor>`.
/// See [`Handler`] documentation for details.
#[proc_macro_derive(HandleClient, attributes(log_level))]
pub fn derive_handle_client(input: TokenStream) -> TokenStream {
    derive_client(input, true)
}

/// Derives a client implementation on `ActorRef<Actor>`.
/// See [`Handler`] documentation for details.
#[proc_macro_derive(RefClient, attributes(log_level))]
pub fn derive_ref_client(input: TokenStream) -> TokenStream {
    derive_client(input, false)
}

fn derive_client(input: TokenStream, is_handle: bool) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident.clone();

    let messages = match parse_message_enum(input.clone()) {
        Ok(messages) => messages,
        Err(err) => return TokenStream::from(err.to_compile_error()),
    };

    // The client implementation methods.
    let mut impl_methods = Vec::new();

    let send_message = if is_handle {
        quote! { self.send(message)? }
    } else {
        quote! { self.send(caps, message)? }
    };
    let global_log_level = parse_log_level(&input.attrs).ok().unwrap_or(None);

    for message in messages {
        match message {
            Message::Call {
                ref variant,
                ref reply_port_is_handle,
                ref return_type,
                ref log_level,
            } => {
                let (arg_names, arg_types): (Vec<_>, Vec<_>) = message.args().into_iter().unzip();
                let variant_name_snake = variant.snake_name();
                let enum_name = variant.enum_name();

                let (reply_port_arg, _) = message.reply_port_arg().unwrap();
                let constructor = variant.constructor();
                let log_level = match (&global_log_level, log_level) {
                    (_, Some(local)) => local.clone(),
                    (Some(global), None) => global.clone(),
                    _ => Ident::new("DEBUG", Span::call_site()),
                };
                let log_level = if is_handle {
                    quote! {
                        tracing::Level::#log_level
                    }
                } else {
                    quote! {
                        tracing::Level::TRACE
                    }
                };
                let log_message = quote! {
                        hyperactor::metrics::MESSAGES_SENT.add(1, hyperactor::kv_pairs!(
                            "rpc" => "call",
                            "actor_id" => self.actor_id().to_string(),
                            "message_type" => stringify!(#enum_name),
                            "variant" => stringify!(#variant_name_snake),
                        ));
                        tracing::event!(target: "message", #log_level, rpc = "call",  payload=?message, "send");

                };
                if *reply_port_is_handle {
                    impl_methods.push(quote! {
                        #[hyperactor::instrument(level=#log_level, rpc = "call", message_type=#name)]
                        async fn #variant_name_snake(
                            &self,
                            caps: &(impl hyperactor::cap::CanSend + hyperactor::cap::CanOpenPort),
                            #(#arg_names: #arg_types),*)
                            -> Result<#return_type, hyperactor::anyhow::Error> {
                            let (#reply_port_arg, reply_receiver) =
                                hyperactor::mailbox::open_once_port::<#return_type>(caps);
                            let message = #constructor;
                            #log_message;
                            #send_message;
                            reply_receiver.recv().await.map_err(hyperactor::anyhow::Error::from)
                        }
                    });
                } else {
                    impl_methods.push(quote! {
                        #[hyperactor::instrument(level=#log_level, rpc="call", message_type=#name)]
                        async fn #variant_name_snake(
                            &self,
                            caps: &(impl hyperactor::cap::CanSend + hyperactor::cap::CanOpenPort),
                            #(#arg_names: #arg_types),*)
                            -> Result<#return_type, hyperactor::anyhow::Error> {
                            let (#reply_port_arg, reply_receiver) =
                                hyperactor::mailbox::open_once_port::<#return_type>(caps);
                            let #reply_port_arg = #reply_port_arg.bind();
                            let message = #constructor;
                            #log_message;
                            #send_message;
                            reply_receiver.recv().await.map_err(hyperactor::anyhow::Error::from)
                        }
                    });
                }
            }
            Message::OneWay {
                ref variant,
                ref log_level,
            } => {
                let (arg_names, arg_types): (Vec<_>, Vec<_>) = message.args().into_iter().unzip();
                let variant_name_snake = variant.snake_name();
                let enum_name = variant.enum_name();
                let constructor = variant.constructor();
                let log_level = match (&global_log_level, log_level) {
                    (_, Some(local)) => local.clone(),
                    (Some(global), None) => global.clone(),
                    _ => Ident::new("DEBUG", Span::call_site()),
                };
                let log_level = if is_handle {
                    quote! {
                        tracing::Level::TRACE
                    }
                } else {
                    quote! {
                        tracing::Level::#log_level
                    }
                };
                let log_message = quote! {
                    hyperactor::metrics::MESSAGES_SENT.add(1, hyperactor::kv_pairs!(
                        "rpc" => "oneway",
                        "actor_id" => self.actor_id().to_string(),
                        "message_type" => stringify!(#enum_name),
                        "variant" => stringify!(#variant_name_snake),
                    ));
                    tracing::event!(target: "message", #log_level,  handle = stringify!(#variant_name_snake), rpc = "oneway",  payload=?message, "send");
                };
                impl_methods.push(quote! {
                    async fn #variant_name_snake(
                        &self,
                        caps: &impl hyperactor::cap::CanSend,
                        #(#arg_names: #arg_types),*)
                        -> Result<(), hyperactor::anyhow::Error> {
                        let message = #constructor;
                        #log_message;
                        #send_message;
                        Ok(())
                    }
                });
            }
        }
    }

    let trait_name = format_ident!("{}Client", name);

    let (_, ty_generics, _) = input.generics.split_for_impl();

    // Add a new generic parameter 'A'
    let a_ident = Ident::new("A", proc_macro2::Span::from(proc_macro::Span::def_site()));
    let mut trait_generics = input.generics.clone();
    trait_generics.params.insert(
        0,
        syn::GenericParam::Type(syn::TypeParam {
            ident: a_ident.clone(),
            attrs: vec![],
            colon_token: None,
            bounds: Punctuated::new(),
            eq_token: None,
            default: None,
        }),
    );
    let (impl_generics, _, _) = trait_generics.split_for_impl();

    let expanded = if is_handle {
        quote! {
            #[hyperactor::async_trait::async_trait]
            impl #impl_generics #trait_name #ty_generics for hyperactor::ActorHandle<#a_ident>
              where #a_ident: hyperactor::Handler<#name #ty_generics> {
                #(#impl_methods)*
            }
        }
    } else {
        quote! {
            #[hyperactor::async_trait::async_trait]
            impl #impl_generics #trait_name #ty_generics for hyperactor::ActorRef<#a_ident>
              where #a_ident: hyperactor::actor::RemoteHandles<#name #ty_generics> {
                #(#impl_methods)*
            }
        }
    };

    TokenStream::from(expanded)
}

const FORWARD_ARGUMENT_ERROR: &str = indoc! {r#"
`forward` expects the message type that is being forwarded

= help: use `#[forward(MessageType)]`
"#};

/// Forward messages of the provided type to this handler implementation.
#[proc_macro_attribute]
pub fn forward(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr_args = parse_macro_input!(attr with Punctuated::<syn::PathSegment, syn::Token![,]>::parse_terminated);
    if attr_args.len() != 1 {
        return TokenStream::from(
            syn::Error::new_spanned(attr_args, FORWARD_ARGUMENT_ERROR).to_compile_error(),
        );
    }

    let message_type = attr_args.first().unwrap();
    let input = parse_macro_input!(item as ItemImpl);

    let self_type = match *input.self_ty {
        syn::Type::Path(ref type_path) => {
            let segment = type_path.path.segments.last().unwrap();
            segment.clone() //ident.clone()
        }
        _ => {
            return TokenStream::from(
                syn::Error::new_spanned(input.self_ty, "`forward` argument must be a type")
                    .to_compile_error(),
            );
        }
    };

    let trait_name = match input.trait_ {
        Some((_, ref trait_path, _)) => trait_path.segments.last().unwrap().clone(),
        None => {
            return TokenStream::from(
                syn::Error::new_spanned(input.self_ty, "no trait in implementation block")
                    .to_compile_error(),
            );
        }
    };

    let expanded = quote! {
        #input

        #[hyperactor::async_trait::async_trait]
        impl hyperactor::Handler<#message_type> for #self_type {
            async fn handle(
                &mut self,
                this: &hyperactor::Instance<Self>,
                message: #message_type,
            ) -> hyperactor::anyhow::Result<()> {
                <Self as #trait_name>::handle(self, this, message).await
            }
        }
    };

    TokenStream::from(expanded)
}

/// Use this macro in place of tracing::instrument to prevent spamming our tracing table.
/// We set a default level of INFO while always setting ERROR if the function returns Result::Err giving us
/// consistent and high quality structured logs. Because this wraps around tracing::instrument, all parameters
/// mentioned in https://fburl.com/9jlkb5q4 should be valid. For functions that don't return a [`Result`] type, use
/// [`instrument_infallible`]
///
/// ```
/// #[telemetry::instrument]
/// async fn yolo() -> anyhow::Result<i32> {
///     Ok(420)
/// }
/// ```
#[proc_macro_attribute]
pub fn instrument(args: TokenStream, input: TokenStream) -> TokenStream {
    let args =
        parse_macro_input!(args with Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated);
    let input = parse_macro_input!(input as ItemFn);
    let output = quote! {
        #[hyperactor::tracing::instrument(err, skip_all, #args)]
        #input
    };

    TokenStream::from(output)
}

/// Use this macro in place of tracing::instrument to prevent spamming our tracing table.
/// Because this wraps around tracing::instrument, all parameters mentioned in
/// https://fburl.com/9jlkb5q4 should be valid.
///
/// ```
/// #[telemetry::instrument]
/// async fn yolo() -> i32 {
///     420
/// }
/// ```
#[proc_macro_attribute]
pub fn instrument_infallible(args: TokenStream, input: TokenStream) -> TokenStream {
    let args =
        parse_macro_input!(args with Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated);
    let input = parse_macro_input!(input as ItemFn);

    let output = quote! {
        #[hyperactor::tracing::instrument(skip_all, #args)]
        #input
    };

    TokenStream::from(output)
}

/// Derive the [`hyperactor::data::Named`] trait for a struct with the
/// provided type URI. The name of the type is its fully-qualified Rust
/// path. The name may be overridden by providing a string value for the
/// `name` attribute.
#[proc_macro_derive(Named, attributes(named))]
pub fn named_derive(input: TokenStream) -> TokenStream {
    // Parse the input struct or enum
    let input = parse_macro_input!(input as DeriveInput);
    let struct_name = &input.ident;

    let mut typename = quote! {
        concat!(std::module_path!(), "::", stringify!(#struct_name))
    };
    let mut dump = true;

    for attr in &input.attrs {
        if attr.path().is_ident("named") {
            if let Ok(meta) = attr.parse_args_with(
                syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated,
            ) {
                for item in meta {
                    if let Meta::NameValue(MetaNameValue {
                        path,
                        value: Expr::Lit(expr_lit),
                        ..
                    }) = item
                    {
                        if path.is_ident("dump") {
                            if let Lit::Bool(lit_bool) = expr_lit.lit {
                                dump = lit_bool.value;
                            }
                        } else if path.is_ident("name") {
                            if let Lit::Str(name) = expr_lit.lit {
                                typename = quote! { #name };
                            }
                        }
                    }
                }
            }
        }
    }

    let cached_typehash = Ident::new(
        &format!("{}_CACHED_TYPEHASH", struct_name).to_case(Case::UpperSnake),
        Span::call_site(),
    );

    let dumper = if dump {
        quote! { Some(<#struct_name as hyperactor::data::NamedDumpable>::dump) }
    } else {
        quote! { None }
    };

    // Generate 'arm' for enums only.
    let arm_impl = match &input.data {
        Data::Enum(DataEnum { variants, .. }) => {
            let match_arms = variants.iter().map(|v| {
                let variant_name = &v.ident;
                let variant_str = variant_name.to_string();
                match &v.fields {
                    Fields::Unit => quote! { Self::#variant_name => Some(#variant_str) },
                    Fields::Unnamed(_) => quote! { Self::#variant_name(..) => Some(#variant_str) },
                    Fields::Named(_) => quote! { Self::#variant_name { .. } => Some(#variant_str) },
                }
            });
            quote! {
                fn arm(&self) -> Option<&'static str> {
                    match self {
                        #(#match_arms,)*
                    }
                }
            }
        }
        _ => quote! {},
    };

    // Ideally we would compute the has directly in the macro itself, however, we don't
    // have access to the fully expanded pathname here as we use the intrinsic std::module_path!() macro.
    let expanded = quote! {
        static #cached_typehash: std::sync::LazyLock<u64> = std::sync::LazyLock::new(|| {
            hyperactor::cityhasher::hash(<#struct_name as hyperactor::data::Named>::typename())
        });

        impl hyperactor::data::Named for #struct_name {
            fn typename() -> &'static str { #typename }
            fn typehash() -> u64 { *#cached_typehash }
            #arm_impl
        }

        hyperactor::submit! {
            hyperactor::data::TypeInfo {
                typename: <#struct_name as hyperactor::data::Named>::typename,
                typehash: <#struct_name as hyperactor::data::Named>::typehash,
                typeid: <#struct_name as hyperactor::data::Named>::typeid,
                port: <#struct_name as hyperactor::data::Named>::port,
                dump: #dumper,
                arm_unchecked: <#struct_name as hyperactor::data::Named>::arm_unchecked,
            }
        }
    };

    TokenStream::from(expanded)
}

/// Attribute Struct for [`fn export`] macro.
struct ExportAttr {
    spawn: bool,
    handlers: Vec<Type>,
}

impl Parse for ExportAttr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut spawn = false;
        let mut handlers = vec![];

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;

            if key == "spawn" {
                let expr: Expr = input.parse()?;
                if let Expr::Lit(ExprLit {
                    lit: Lit::Bool(b), ..
                }) = expr
                {
                    spawn = b.value;
                } else {
                    return Err(syn::Error::new_spanned(
                        expr,
                        "expected boolean for `spawn`",
                    ));
                }
            } else if key == "handlers" {
                let content;
                bracketed!(content in input);
                let types = content.parse_terminated(Type::parse, Token![,])?;
                if types.is_empty() {
                    return Err(syn::Error::new_spanned(
                        types,
                        "`handlers` must include at least one type",
                    ));
                }
                handlers = types.into_iter().collect();
            } else {
                return Err(syn::Error::new_spanned(
                    key,
                    "unexpected key in `#[export(...)]`. Only supports `spawn` and `handlers`",
                ));
            }

            // optional trailing comma
            let _ = input.parse::<Token![,]>();
        }

        Ok(ExportAttr { spawn, handlers })
    }
}

/// Exports handlers for this actor. The set of exported handlers
/// determine the messages that may be sent to remote references of
/// the actor ([`hyperaxtor::ActorRef`]). Only messages that implement
/// [`hyperactor::RemoteMessage`] may be exported.
///
/// Additionally, an exported actor may be remotely spawned,
/// indicated by `spawn = true`. Such actors must also ensure that
/// their parameter type implements [`hyperactor::RemoteMessage`].
///
/// # Example
///
/// In the following example, `MyActor` can be spawned remotely. It also has
/// exports handlers for two message types, `MyMessage` and `MyOtherMessage`.
/// Consequently, `ActorRef`s of the actor's type may dispatch messages of these
/// types.
///
/// ```ignore
/// #[export(
///     spawn = true,
///     handlers = [
///         MyMessage,
///         MyOtherMessage,
///     ],
/// )]
/// struct MyActor {}
/// ```
#[proc_macro_attribute]
pub fn export(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input: DeriveInput = parse_macro_input!(item as DeriveInput);
    let data_type_name = &input.ident;

    let ExportAttr { spawn, handlers } = parse_macro_input!(attr as ExportAttr);

    let mut handles = Vec::new();
    let mut bindings = Vec::new();

    for ty in &handlers {
        handles.push(quote! {
            impl hyperactor::actor::RemoteHandles<#ty> for #data_type_name {}
        });
        bindings.push(quote! {
            ports.bind::<#ty>();
        });
    }

    let mut expanded = quote! {
        #input

        impl hyperactor::actor::RemoteActor for #data_type_name {}

        #(#handles)*

        // Always export the `Signal` type.
        impl hyperactor::actor::RemoteHandles<hyperactor::actor::Signal> for #data_type_name {}

        impl hyperactor::actor::Binds<#data_type_name> for #data_type_name {
            fn bind(ports: &hyperactor::proc::Ports<Self>) {
                #(#bindings)*
            }
        }

        impl hyperactor::data::Named for #data_type_name {
            fn typename() -> &'static str { concat!(std::module_path!(), "::", stringify!(#data_type_name)) }
        }
    };

    if spawn {
        expanded.extend(quote! {

            hyperactor::remote!(#data_type_name);
        });
    }

    TokenStream::from(expanded)
}
