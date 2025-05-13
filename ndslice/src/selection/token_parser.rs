//! A `TokenStream` to [`Selection`] parser used by the `sel!`
//! procedural macro.
//!
//! This module implements a compile-time parser that converts a
//! [`proc_macro2::TokenStream`] into a [`Selection`] syntax tree.
//!
//! The grammar and interpretation of selection expressions are the
//! same as those described in the [`parse`] module. See that module
//! for full documentation of the syntax and semantics.
//!
//! See [`parse_tokens`] for the entry point, and
//! [`selection_to_tokens`] for the inverse.

use std::iter::Peekable;

use proc_macro2::Delimiter;
use proc_macro2::TokenStream;
use proc_macro2::TokenTree;
use quote::quote;

use crate::selection::Selection;
use crate::selection::dsl;
use crate::shape;

// Selection expressions grammar:
// ```text
// expression ::= union
// union      ::= intersection ('|' intersection)*
// intersection ::= dimension ('&' dimension)*
// dimension  ::= group (',' group)*
// group      ::= range | index | * | ? | (expression)
// ```

/// Parses a [`proc_macro2::TokenStream`] representing a selection
/// expression into a [`Selection`] syntax tree.
///
/// This is intended for use at compile time in the `sel!` procedural macro,
/// and is the inverse of [`selection_to_tokens`].
pub fn parse_tokens(tokens: TokenStream) -> Result<Selection, String> {
    let mut iter = tokens.into_iter().peekable();
    parse_expression(&mut iter)
}

pub fn selection_to_tokens(sel: &Selection) -> proc_macro2::TokenStream {
    match sel {
        Selection::True => quote!(Selection::True),
        Selection::False => quote!(Selection::False),
        Selection::All(inner) => {
            let inner = selection_to_tokens(inner);
            quote!(Selection::All(Box::new(#inner)))
        }
        Selection::Range(r, inner) => {
            let start = r.0;
            let end = match &r.1 {
                Some(e) => quote!(Some(#e)),
                None => quote!(None),
            };
            let step = r.2;
            let inner = selection_to_tokens(inner);
            quote! {
                ::ndslice::selection::Selection::Range(
                    ::ndslice::shape::Range(#start, #end, #step),
                    Box::new(#inner)
                )
            }
        }
        Selection::Any(inner) => {
            let inner = selection_to_tokens(inner);
            quote!(Selection::Any(Box::new(#inner)))
        }
        Selection::Intersection(a, b) => {
            let a = selection_to_tokens(a);
            let b = selection_to_tokens(b);
            quote!(Selection::Intersection(Box::new(#a), Box::new(#b)))
        }
        Selection::Union(a, b) => {
            let a = selection_to_tokens(a);
            let b = selection_to_tokens(b);
            quote!(Selection::Union(Box::new(#a), Box::new(#b)))
        }
        _ => unimplemented!(),
    }
}

fn parse_expression<I>(tokens: &mut Peekable<I>) -> Result<Selection, String>
where
    I: Iterator<Item = TokenTree>,
{
    let mut lhs = parse_intersection(tokens)?;
    while let Some(TokenTree::Punct(p)) = tokens.peek() {
        if p.as_char() == '|' {
            tokens.next(); // consume |
            let rhs = parse_intersection(tokens)?;
            lhs = dsl::union(lhs, rhs);
        } else {
            break;
        }
    }
    Ok(lhs)
}

fn parse_intersection<I>(tokens: &mut Peekable<I>) -> Result<Selection, String>
where
    I: Iterator<Item = TokenTree>,
{
    let mut lhs = parse_dimensions(tokens)?;
    while let Some(TokenTree::Punct(p)) = tokens.peek() {
        if p.as_char() == '&' {
            tokens.next(); // consume &
            let rhs = parse_dimensions(tokens)?;
            lhs = dsl::intersection(lhs, rhs);
        } else {
            break;
        }
    }
    Ok(lhs)
}

fn parse_dimensions<I>(tokens: &mut Peekable<I>) -> Result<Selection, String>
where
    I: Iterator<Item = TokenTree>,
{
    let mut dims = vec![];

    loop {
        dims.push(parse_atom(tokens)?);

        match tokens.peek() {
            Some(TokenTree::Punct(p)) if p.as_char() == ',' => {
                tokens.next(); // consume comma
            }
            _ => break,
        }
    }

    let mut result = dsl::true_();

    for dim in dims.into_iter().rev() {
        result = apply_dimension_chain(dim, result)?;
    }

    Ok(result)
}

fn apply_dimension_chain(sel: Selection, tail: Selection) -> Result<Selection, String> {
    Ok(match sel {
        Selection::All(inner) => dsl::all(apply_dimension_chain(*inner, tail)?),
        Selection::Any(inner) => dsl::any(apply_dimension_chain(*inner, tail)?),
        Selection::Range(r, inner) => dsl::range(r, apply_dimension_chain(*inner, tail)?),
        Selection::Union(a, b) => dsl::union(
            apply_dimension_chain(*a, tail.clone())?,
            apply_dimension_chain(*b, tail)?,
        ),
        Selection::Intersection(a, b) => dsl::intersection(
            apply_dimension_chain(*a, tail.clone())?,
            apply_dimension_chain(*b, tail)?,
        ),
        Selection::True => tail,
        Selection::False => dsl::false_(),
        other => {
            return Err(format!(
                "unexpected selection type in dimension chain: {:?}",
                other
            ));
        }
    })
}

fn parse_atom<I>(tokens: &mut Peekable<I>) -> Result<Selection, String>
where
    I: Iterator<Item = TokenTree>,
{
    match tokens.peek() {
        Some(TokenTree::Punct(p)) if p.as_char() == '*' => {
            tokens.next();
            Ok(dsl::all(dsl::true_()))
        }
        Some(TokenTree::Punct(p)) if p.as_char() == '?' => {
            tokens.next();
            Ok(dsl::any(dsl::true_()))
        }
        Some(TokenTree::Punct(p)) if p.as_char() == ':' => {
            tokens.next(); // consume ':'

            // Optional end
            let end = match tokens.peek() {
                Some(TokenTree::Literal(_lit)) => {
                    let lit = tokens
                        .next()
                        .ok_or_else(|| "expected literal after ':'".to_string())?;
                    Some(
                        lit.to_string()
                            .parse::<usize>()
                            .map_err(|e| e.to_string())?,
                    )
                }
                _ => None,
            };

            // Optional step
            let step = match tokens.peek() {
                Some(TokenTree::Punct(p)) if p.as_char() == ':' => {
                    tokens.next(); // consume second ':'
                    let lit = match tokens.next() {
                        Some(TokenTree::Literal(lit)) => lit,
                        other => return Err(format!("expected step after ::, got {:?}", other)),
                    };
                    lit.to_string()
                        .parse::<usize>()
                        .map_err(|e| e.to_string())?
                }
                _ => 1,
            };

            Ok(dsl::range(shape::Range(0, end, step), dsl::true_()))
        }
        Some(TokenTree::Literal(_)) => {
            // literal-prefixed range or index
            parse_range_or_index(tokens)
        }
        Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Parenthesis => {
            let group = tokens.next().unwrap(); // consume group
            let mut inner = match group {
                TokenTree::Group(g) => g.stream().into_iter().peekable(),
                _ => unreachable!(),
            };
            parse_expression(&mut inner)
        }
        Some(t) => Err(format!("unexpected token: {:?}", t)),
        None => Err("unexpected end of input".to_string()),
    }
}

fn parse_range_or_index<I>(tokens: &mut Peekable<I>) -> Result<Selection, String>
where
    I: Iterator<Item = TokenTree>,
{
    // Peek and parse the start literal
    let start_lit = match tokens.next() {
        Some(TokenTree::Literal(lit)) => lit,
        other => return Err(format!("expected number, got {:?}", other)),
    };

    let start = start_lit
        .to_string()
        .parse::<usize>()
        .map_err(|e| format!("invalid number: {}", e))?;

    // Check if this is a range by looking for a colon
    if let Some(TokenTree::Punct(p)) = tokens.peek() {
        if p.as_char() == ':' {
            tokens.next(); // consume ':'

            // Try to parse optional end
            let end = match tokens.peek() {
                Some(TokenTree::Literal(_lit)) => {
                    let lit = tokens.next().unwrap();
                    Some(
                        lit.to_string()
                            .parse::<usize>()
                            .map_err(|e| format!("invalid range end: {}", e))?,
                    )
                }
                Some(TokenTree::Punct(p)) if p.as_char() == ':' => None,
                _ => None,
            };

            // Try to parse optional step
            let step = match tokens.peek() {
                Some(TokenTree::Punct(p)) if p.as_char() == ':' => {
                    tokens.next(); // consume second ':'
                    let lit = tokens.next().ok_or("expected number for step after ::")?;
                    match lit {
                        TokenTree::Literal(lit) => lit
                            .to_string()
                            .parse::<usize>()
                            .map_err(|e| format!("invalid step size: {}", e))?,
                        other => return Err(format!("expected literal for step, got {:?}", other)),
                    }
                }
                _ => 1,
            };

            Ok(dsl::range(shape::Range(start, end, step), dsl::true_()))
        } else {
            // Not a range, treat as index
            Ok(dsl::range(
                shape::Range(start, Some(start + 1), 1),
                dsl::true_(),
            ))
        }
    } else {
        // No colon → single index
        Ok(dsl::range(
            shape::Range(start, Some(start + 1), 1),
            dsl::true_(),
        ))
    }
}
