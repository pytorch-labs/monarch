/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::any::type_name;
use std::num::ParseIntError;
use std::str::FromStr;

/// Errors that can occur while parsing a string.
#[derive(Debug, thiserror::Error)]
pub enum ParserError {
    #[error("parse error: expected '{expected}' but got end of input")]
    UnexpectedEndOfInput { expected: &'static str },

    #[error("parse error: expected '{expected}' but got '{actual}'")]
    WrongToken {
        expected: &'static str,
        actual: String,
    },

    #[error("parse error: token '{actual}' is not a '{expected_type}'")]
    WrongTokenType {
        expected_type: &'static str,
        actual: String,
    },

    #[error("parse error: {error}: expected integer but got '{token}'")]
    NotAnInteger {
        token: String,
        #[source]
        error: ParseIntError,
    },
}

/// A simple parser, focused on providing an ergonomic API to consume lexemes and
/// to encourage useful errors.
///
/// At its simplest, a parser is an iterator over lexemes; it additionally provides
/// more advanced methods with which to (sometimes optionally) consume lexemes.
pub struct Parser<'a> {
    str: &'a str,
    delims: &'a [&'a str],
}

impl<'a> Parser<'a> {
    /// Create a new parser that uses the provided delimiters to to define
    /// lexical boundaries. Each delimiter is also a lexeme.
    pub fn new(str: &'a str, delims: &'a [&'a str]) -> Self {
        Self { str, delims }
    }

    /// Peek the next available lexeme, returning `None` if the the parser has
    /// reached the end of its input
    pub fn peek(&self) -> Option<&'a str> {
        self.split().map(|(token, _)| token)
    }

    /// Like `peek`, but return a parsing error if the parser has reached the
    /// end of its input.
    pub fn peek_or_err(&self, expected: &'static str) -> Result<&'a str, ParserError> {
        self.split()
            .map(|(token, _)| token)
            .ok_or(ParserError::UnexpectedEndOfInput { expected })
    }

    /// Returns an error if the next token is not `expected`. The token is consumed
    /// if it is `expected`.
    pub fn expect(&mut self, expected: &'static str) -> Result<(), ParserError> {
        let token = self.peek_or_err(expected)?;
        if token != expected {
            Err(ParserError::WrongToken {
                expected,
                actual: token.to_string(),
            })
        } else {
            let _ = self.next();
            Ok(())
        }
    }

    /// Returns the next token, or an error if the parser has reached the end of
    /// its input.
    pub fn next_or_err(&mut self, expected: &'static str) -> Result<&'a str, ParserError> {
        self.next()
            .ok_or(ParserError::UnexpectedEndOfInput { expected })
    }

    /// Try to parse the next token as a `T`. The token is consumed if on success.
    pub fn try_parse<T: FromStr>(&mut self) -> Result<T, ParserError> {
        let token = self.peek_or_err("a token")?;
        let result = token.parse().map_err(|e| ParserError::WrongTokenType {
            expected_type: type_name::<T>(),
            actual: token.to_string(),
        });
        if result.is_ok() {
            let _ = self.next();
        }
        result
    }

    /// Returns true if the parser has reached the end of its input.
    pub fn is_empty(&self) -> bool {
        self.str.trim().is_empty()
    }

    fn split(&self) -> Option<(&'a str, &'a str)> {
        if self.str.is_empty() {
            return None;
        }

        match self
            .delims
            .iter()
            .enumerate()
            .flat_map(|(index, d)| self.str.find(d).map(|pos| (index, pos)))
            .min_by_key(|&(_, v)| v)
        {
            Some((index, 0)) => Some((self.delims[index], &self.str[self.delims[index].len()..])),
            Some((_, pos)) => Some((&self.str[..pos].trim(), &self.str[pos..])),
            None => Some((self.str.trim(), "")),
        }
    }
}

impl<'a> Iterator for Parser<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        self.split().map(|(token, rest)| {
            self.str = rest;
            token
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let mut p = Parser::new("foo,bar", &[","]);
        assert_eq!(p.next(), Some("foo"));
        assert_eq!(p.next(), Some(","));
        assert_eq!(p.peek(), Some("bar"));
        assert_eq!(p.next(), Some("bar"));
        assert_eq!(p.next(), None);
        assert_eq!(p.peek(), None);
    }
}
