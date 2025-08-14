/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

use anyhow::Result;
use anyhow::anyhow;
use lazy_static::lazy_static;
use rusqlite::Connection;
use rusqlite::functions::FunctionFlags;
use serde::Serialize;
use serde_json::Value as JValue;
use serde_rusqlite::*;
use tracing::Event;
use tracing::Subscriber;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::Layer;
use tracing_subscriber::filter::Targets;
use tracing_subscriber::prelude::*;

pub trait TableDef {
    fn name(&self) -> &'static str;
    fn columns(&self) -> &'static [&'static str];
    fn create_table_stmt(&self) -> String {
        let name = self.name();
        let columns = self
            .columns()
            .iter()
            .map(|col| format!("{col} TEXT "))
            .collect::<Vec<String>>()
            .join(",");
        format!("create table if not exists {name} (seq INTEGER primary key, {columns})")
    }
    fn insert_stmt(&self) -> String {
        let name = self.name();
        let columns = self.columns().join(", ");
        let params = self
            .columns()
            .iter()
            .map(|c| format!(":{c}"))
            .collect::<Vec<String>>()
            .join(", ");
        format!("insert into {name} ({columns}) values ({params})")
    }
}

impl TableDef for (&'static str, &'static [&'static str]) {
    fn name(&self) -> &'static str {
        self.0
    }

    fn columns(&self) -> &'static [&'static str] {
        self.1
    }
}

#[derive(Clone, Debug)]
pub struct Table {
    pub columns: &'static [&'static str],
    pub create_table_stmt: String,
    pub insert_stmt: String,
}

impl From<(&'static str, &'static [&'static str])> for Table {
    fn from(value: (&'static str, &'static [&'static str])) -> Self {
        Self {
            columns: value.columns(),
            create_table_stmt: value.create_table_stmt(),
            insert_stmt: value.insert_stmt(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableName {
    ActorLifecycle,
    Messages,
    LogEvents,
}

impl TableName {
    pub const ACTOR_LIFECYCLE_STR: &'static str = "actor_lifecycle";
    pub const MESSAGES_STR: &'static str = "messages";
    pub const LOG_EVENTS_STR: &'static str = "log_events";

    pub fn as_str(&self) -> &'static str {
        match self {
            TableName::ActorLifecycle => Self::ACTOR_LIFECYCLE_STR,
            TableName::Messages => Self::MESSAGES_STR,
            TableName::LogEvents => Self::LOG_EVENTS_STR,
        }
    }

    pub fn get_table(&self) -> &'static Table {
        match self {
            TableName::ActorLifecycle => &ACTOR_LIFECYCLE,
            TableName::Messages => &MESSAGES,
            TableName::LogEvents => &LOG_EVENTS,
        }
    }
}

lazy_static! {
    static ref ACTOR_LIFECYCLE: Table = (
        TableName::ActorLifecycle.as_str(),
        [
            "actor_id",
            "actor",
            "name",
            "supervised_actor",
            "actor_status",
            "module_path",
            "line",
            "file",
        ]
        .as_slice()
    )
        .into();
    static ref MESSAGES: Table = (
        TableName::Messages.as_str(),
        [
            "span_id",
            "time_us",
            "src",
            "dest",
            "payload",
            "module_path",
            "line",
            "file",
        ]
        .as_slice()
    )
        .into();
    static ref LOG_EVENTS: Table = (
        TableName::LogEvents.as_str(),
        [
            "span_id",
            "time_us",
            "name",
            "message",
            "actor_id",
            "level",
            "line",
            "file",
            "module_path",
        ]
        .as_slice()
    )
        .into();
    static ref ALL_TABLES: Vec<Table> = vec![
        ACTOR_LIFECYCLE.clone(),
        MESSAGES.clone(),
        LOG_EVENTS.clone()
    ];
}

pub struct SqliteLayer {
    conn: Arc<Mutex<Connection>>,
}
use tracing::field::Visit;

#[derive(Debug, Clone, Default, Serialize)]
struct SqlVisitor(HashMap<String, JValue>);

impl Visit for SqlVisitor {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.0.insert(
            field.name().to_string(),
            JValue::String(format!("{:?}", value)),
        );
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.0
            .insert(field.name().to_string(), JValue::String(value.to_string()));
    }

    fn record_i64(&mut self, field: &tracing::field::Field, value: i64) {
        self.0
            .insert(field.name().to_string(), JValue::Number(value.into()));
    }

    fn record_f64(&mut self, field: &tracing::field::Field, value: f64) {
        let n = serde_json::Number::from_f64(value).unwrap();
        self.0.insert(field.name().to_string(), JValue::Number(n));
    }

    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        self.0
            .insert(field.name().to_string(), JValue::Number(value.into()));
    }

    fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
        self.0.insert(field.name().to_string(), JValue::Bool(value));
    }
}

macro_rules! insert_event {
    ($table:expr, $conn:ident, $event:ident) => {
        let mut v: SqlVisitor = Default::default();
        $event.record(&mut v);
        let meta = $event.metadata();
        v.0.insert(
            "module_path".to_string(),
            meta.module_path().map(String::from).into(),
        );
        v.0.insert("line".to_string(), meta.line().into());
        v.0.insert("file".to_string(), meta.file().map(String::from).into());
        $conn.prepare_cached(&$table.insert_stmt)?.execute(
            serde_rusqlite::to_params_named_with_fields(v, $table.columns)?
                .to_slice()
                .as_slice(),
        )?;
    };
}

impl SqliteLayer {
    pub fn new() -> Result<Self> {
        let conn = Connection::open_in_memory()?;

        for table in ALL_TABLES.iter() {
            conn.execute(&table.create_table_stmt, [])?;
        }
        conn.create_scalar_function(
            "assert",
            2,
            FunctionFlags::SQLITE_UTF8 | FunctionFlags::SQLITE_DETERMINISTIC,
            move |ctx| {
                let condition: bool = ctx.get(0)?;
                let message: String = ctx.get(1)?;

                if !condition {
                    return Err(rusqlite::Error::UserFunctionError(
                        anyhow!("assertion failed:{condition} {message}",).into(),
                    ));
                }

                Ok(condition)
            },
        )?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    fn insert_event(&self, event: &Event<'_>) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        match (event.metadata().target(), event.metadata().name()) {
            (TableName::MESSAGES_STR, _) => {
                insert_event!(TableName::Messages.get_table(), conn, event);
            }
            (TableName::ACTOR_LIFECYCLE_STR, _) => {
                insert_event!(TableName::ActorLifecycle.get_table(), conn, event);
            }
            _ => {
                insert_event!(TableName::LogEvents.get_table(), conn, event);
            }
        }
        Ok(())
    }

    pub fn connection(&self) -> Arc<Mutex<Connection>> {
        self.conn.clone()
    }
}

impl<S: Subscriber> Layer<S> for SqliteLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: tracing_subscriber::layer::Context<'_, S>) {
        self.insert_event(event).unwrap();
    }
}

#[allow(dead_code)]
fn print_table(conn: &Connection, table_name: TableName) -> Result<()> {
    let table_name_str = table_name.as_str();

    // Get column names
    let mut stmt = conn.prepare(&format!("PRAGMA table_info({})", table_name_str))?;
    let column_info = stmt.query_map([], |row| {
        row.get::<_, String>(1) // Column name is at index 1
    })?;

    let columns: Vec<String> = column_info.collect::<Result<Vec<_>, _>>()?;

    // Print header
    println!("=== {} ===", table_name_str.to_uppercase());
    println!("{}", columns.join(" | "));
    println!("{}", "-".repeat(columns.len() * 10));

    // Print rows
    let mut stmt = conn.prepare(&format!("SELECT * FROM {}", table_name_str))?;
    let rows = stmt.query_map([], |row| {
        let mut values = Vec::new();
        for (i, column) in columns.iter().enumerate() {
            // Handle different column types properly
            let value = if i == 0 && *column == "seq" {
                // First column is always the INTEGER seq column
                match row.get::<_, Option<i64>>(i)? {
                    Some(v) => v.to_string(),
                    None => "NULL".to_string(),
                }
            } else {
                // All other columns are TEXT
                match row.get::<_, Option<String>>(i)? {
                    Some(v) => v,
                    None => "NULL".to_string(),
                }
            };
            values.push(value);
        }
        Ok(values.join(" | "))
    })?;

    for row in rows {
        println!("{}", row?);
    }
    println!();
    Ok(())
}

pub fn with_tracing_db() -> Arc<Mutex<Connection>> {
    let layer = SqliteLayer::new().unwrap();
    let conn = layer.connection();

    let layer = layer.with_filter(
        Targets::new()
            .with_default(LevelFilter::TRACE)
            .with_targets(vec![
                ("tokio", LevelFilter::OFF),
                ("opentelemetry", LevelFilter::OFF),
                ("runtime", LevelFilter::OFF),
            ]),
    );
    tracing_subscriber::registry().with(layer).init();
    conn
}

#[cfg(test)]
mod tests {
    use tracing::info;

    use super::*;

    #[test]
    fn test_sqlite_layer() -> Result<()> {
        let conn = with_tracing_db();

        info!(target:"messages", test_field = "test_value", "Test msg");
        info!(target:"log_events", test_field = "test_value", "Test event");

        let count: i64 =
            conn.lock()
                .unwrap()
                .query_row("SELECT COUNT(*) FROM messages", [], |row| row.get(0))?;
        print_table(&conn.lock().unwrap(), TableName::LogEvents)?;
        assert!(count > 0);
        Ok(())
    }
}
