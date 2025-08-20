/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use perfetto_trace_proto_rust::perfetto_trace::CounterDescriptor;
use perfetto_trace_proto_rust::perfetto_trace::DebugAnnotation;
use perfetto_trace_proto_rust::perfetto_trace::DebugAnnotationName;
use perfetto_trace_proto_rust::perfetto_trace::EventCategory;
use perfetto_trace_proto_rust::perfetto_trace::EventName;
use perfetto_trace_proto_rust::perfetto_trace::InternedData;
use perfetto_trace_proto_rust::perfetto_trace::InternedString;
use perfetto_trace_proto_rust::perfetto_trace::ProcessDescriptor;
use perfetto_trace_proto_rust::perfetto_trace::ThreadDescriptor;
use perfetto_trace_proto_rust::perfetto_trace::TracePacket;
use perfetto_trace_proto_rust::perfetto_trace::TrackDescriptor;
use perfetto_trace_proto_rust::perfetto_trace::TrackEvent;
use perfetto_trace_proto_rust::perfetto_trace::track_event::Type::TYPE_COUNTER;
use perfetto_trace_proto_rust::perfetto_trace::track_event::Type::TYPE_INSTANT;
use perfetto_trace_proto_rust::perfetto_trace::track_event::Type::TYPE_SLICE_BEGIN;
use perfetto_trace_proto_rust::perfetto_trace::track_event::Type::TYPE_SLICE_END;
use protobuf::MessageField;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;

#[derive(
    Deserialize,
    Serialize,
    Debug,
    Clone,
    Default,
    PartialEq,
    Eq,
    Hash,
    valuable::Valuable
)]
pub struct RawEvent {
    pub time: u64,
    pub time_us: u64,
    pub span_id: u64,
    pub parent_span_id: Option<u64>,
    pub name: Option<String>,
    pub event_type: String,
    pub message: Option<String>,
    pub actor_id: Option<String>,
    pub file: Option<String>,
    pub lineno: u32,
    pub os_pid: i32,
    pub tokio_task_id: i64,
    pub target: String,
    pub err: Option<String>,
    pub stacktrace: Option<String>,
}

impl From<&RawEvent> for Vec<DebugAnnotation> {
    fn from(value: &RawEvent) -> Self {
        let mut out = Self::new();
        let mut ann = DebugAnnotation::new();
        ann.set_name("message".into());
        ann.set_string_value(value.message.clone().unwrap_or_default());
        out.push(ann);
        out
    }
}

impl PartialOrd for RawEvent {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RawEvent {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.time_us == other.time_us {
            return match (self.event_type.as_str(), other.event_type.as_str()) {
                ("start_span", "start_span") => std::cmp::Ordering::Equal,
                ("start_span", _) => std::cmp::Ordering::Less,
                (_, "start_span") => std::cmp::Ordering::Greater,
                ("async_enter", ot) if ot != "async_enter" => std::cmp::Ordering::Less,
                _ => std::cmp::Ordering::Equal,
            };
        }
        self.time_us.cmp(&other.time_us)
    }
}

impl From<&RawEvent> for DebugAnnotation {
    fn from(value: &RawEvent) -> Self {
        let mut root = Self::new();
        root.set_name("Raw Event".into());
        let values = serde_json::json!(value);
        for (k, v) in values.as_object().unwrap() {
            let mut kv = Self::new();
            kv.set_name(k.clone());
            match v {
                Value::Number(n) => kv.set_int_value(n.as_i64().unwrap_or_default()),
                _ => kv.set_string_value(v.to_string()),
            }
            root.dict_entries.push(kv);
        }

        root
    }
}

#[derive(Default)]
struct Strings {
    next: u64,
    flushed: u64,
    data: HashMap<String, u64>,
}
impl Strings {
    fn intern<'a, T: Into<&'a str>>(&mut self, val: T) -> u64 {
        let val = val.into();
        if let Some(id) = self.data.get(val) {
            return *id;
        }
        self.next += 1;
        let id = self.next;
        self.data.insert(val.to_string(), id);
        id
    }

    fn needs_flush(&self) -> bool {
        self.flushed < self.next
    }

    fn flush(&mut self) {
        self.flushed = self.next;
    }

    fn to_flush(&self) -> impl Iterator<Item = (&String, &u64)> {
        self.data.iter().filter(|item| *item.1 >= self.flushed)
    }
}

pub trait Sink {
    fn consume(&mut self, packet: TracePacket);
}
pub struct CounterTrack<'a, T: Sink> {
    desc: TrackDescriptor,
    ctx: &'a mut Ctx<T>,
}

impl<'a, T: Sink> CounterTrack<'a, T> {
    pub fn name(mut self, name: &str) -> Self {
        self.desc.set_name(name.to_owned());
        self
    }
    pub fn parent(mut self, id: u64) -> Self {
        self.desc.set_parent_uuid(id);
        self
    }

    pub fn process(mut self, pid: i32) -> Self {
        let mut pd = ProcessDescriptor::new();
        pd.set_pid(pid);
        self.desc.process = MessageField::some(pd);
        self
    }

    pub fn consume(self) -> u64 {
        let mut tp = TracePacket::new();
        let id = self.desc.uuid();
        let mut desc = self.desc;
        desc.counter = MessageField::some(CounterDescriptor::new());
        tp.set_track_descriptor(desc);
        self.ctx.consume(tp);
        id
    }
}

pub struct Track<'a, T: Sink> {
    desc: TrackDescriptor,
    ctx: &'a mut Ctx<T>,
}

impl<'a, T: Sink> Track<'a, T> {
    pub fn name(mut self, name: &str) -> Self {
        self.desc.set_name(name.to_owned());
        self
    }
    pub fn parent(mut self, id: u64) -> Self {
        self.desc.set_parent_uuid(id);
        self
    }

    pub fn process(mut self, pid: i32) -> Self {
        let mut pd = ProcessDescriptor::new();
        pd.set_pid(pid);
        self.desc.process = MessageField::some(pd);
        self
    }

    pub fn consume(self) -> u64 {
        let mut tp = TracePacket::new();
        let id = self.desc.uuid();
        tp.set_track_descriptor(self.desc);
        self.ctx.consume(tp);
        id
    }
}
pub struct Instant<'a, T: Sink> {
    event: TrackEvent,
    ctx: &'a mut Ctx<T>,
}

impl<'a, T: Sink> Instant<'a, T> {
    pub fn name(mut self, name: &str) -> Self {
        let id = self.ctx.event_names.intern(name);
        self.event.set_name_iid(id);
        self
    }

    pub fn debug(mut self, values: &serde_json::Value) -> Self {
        self.event
            .debug_annotations
            .push(self.ctx.debug_annotation("debug", values));
        self
    }

    pub fn consume(self) {
        let mut tp = TracePacket::new();
        let mut event = self.event;
        tp.set_timestamp(event.timestamp_absolute_us() as u64 * 1000);
        event.clear_timestamp_absolute_us();
        event.set_type(TYPE_INSTANT);
        tp.set_track_event(event);
        self.ctx.consume(tp);
    }
}

pub struct Counter<'a, T: Sink> {
    event: TrackEvent,
    ctx: &'a mut Ctx<T>,
}

impl<'a, T: Sink> Counter<'a, T> {
    pub fn name(mut self, name: &str) -> Self {
        let id = self.ctx.event_names.intern(name);
        self.event.set_name_iid(id);
        self
    }

    pub fn track(mut self, id: u64) -> Self {
        self.event.set_track_uuid(id);
        self
    }

    pub fn int(mut self, v: i64) -> Self {
        self.event.set_counter_value(v);
        self
    }

    pub fn float(mut self, v: f64) -> Self {
        self.event.set_double_counter_value(v);
        self
    }

    pub fn consume(self) {
        let mut tp = TracePacket::new();
        let mut event = self.event;
        tp.set_timestamp(event.timestamp_absolute_us() as u64 * 1000);
        event.clear_timestamp_absolute_us();
        event.set_type(TYPE_COUNTER);
        tp.set_track_event(event);
        self.ctx.consume(tp);
    }
}

pub struct StartSlice<'a, T: Sink> {
    event: TrackEvent,
    ctx: &'a mut Ctx<T>,
}

impl<'a, T: Sink> StartSlice<'a, T> {
    pub fn name(mut self, name: &str) -> Self {
        let id = self.ctx.event_names.intern(name);
        self.event.set_name_iid(id);
        self
    }

    pub fn get_name(&self) -> &str {
        self.event.name()
    }

    pub fn debug(mut self, values: &serde_json::Value) -> Self {
        self.event
            .debug_annotations
            .push(self.ctx.debug_annotation("debug", values));
        self
    }

    pub fn consume(self) {
        let mut tp = TracePacket::new();
        let mut event = self.event;
        tp.set_timestamp(event.timestamp_absolute_us() as u64 * 1000);
        event.clear_timestamp_absolute_us();
        event.set_type(TYPE_SLICE_BEGIN);
        tp.set_track_event(event);
        self.ctx.consume(tp);
    }
}
pub struct EndSlice<'a, T: Sink> {
    event: TrackEvent,
    ctx: &'a mut Ctx<T>,
}

impl<'a, T: Sink> EndSlice<'a, T> {
    pub fn consume(self) {
        let mut tp = TracePacket::new();
        let mut event = self.event;
        event.set_type(TYPE_SLICE_END);
        tp.set_timestamp(event.timestamp_absolute_us() as u64 * 1000);
        event.clear_timestamp_absolute_us();
        tp.set_track_event(event);
        self.ctx.consume(tp);
    }
}

pub struct Ctx<T: Sink> {
    next_uuid: AtomicU64,
    seq: u32,

    // useful for translating from other frameworks
    remappped_ids: HashMap<u64, u64>,

    // interned data
    event_names: Strings,
    event_categories: Strings,
    debug_annotation_names: Strings,
    debug_annotation_strings: Strings,

    sink: T,
}

impl<T: Sink> Ctx<T> {
    pub fn remap(&mut self, old: u64) -> u64 {
        let next = self.next_uuid();
        *self.remappped_ids.entry(old).or_insert(next)
    }

    pub fn new(sink: T) -> Self {
        let seq = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as u32;

        let mut ctx = Self {
            sink,
            seq,
            remappped_ids: Default::default(),
            next_uuid: Default::default(),
            event_names: Default::default(),
            event_categories: Default::default(),
            debug_annotation_names: Default::default(),
            debug_annotation_strings: Default::default(),
        };
        ctx.next_uuid();
        ctx.init_trace();
        ctx
    }

    pub fn init_trace(&mut self) {
        let mut id = TracePacket::new();
        self.seq += 1;
        id.set_trusted_packet_sequence_id(self.seq);
        id.set_incremental_state_cleared(true);
        id.first_packet_on_sequence = Some(true);
        id.sequence_flags = Some(3);
        self.consume(id);
    }

    pub fn instant(&mut self, track: u64, time_us: u64) -> Instant<'_, T> {
        let mut te = TrackEvent::new();
        te.set_track_uuid(track);
        te.set_timestamp_absolute_us(time_us as i64);
        Instant {
            event: te,
            ctx: self,
        }
    }

    pub fn start_slice(&mut self, track: u64, time_us: u64) -> StartSlice<'_, T> {
        let mut te = TrackEvent::new();
        te.set_track_uuid(track);
        te.set_timestamp_absolute_us(time_us as i64);
        StartSlice {
            event: te,
            ctx: self,
        }
    }

    pub fn counter(&mut self, time_us: u64) -> Counter<'_, T> {
        let mut te = TrackEvent::new();
        te.set_timestamp_absolute_us(time_us as i64);
        Counter {
            event: te,
            ctx: self,
        }
    }

    pub fn end_slice(&mut self, track: u64, time_us: u64) -> EndSlice<'_, T> {
        let mut te = TrackEvent::new();
        te.set_track_uuid(track);
        te.set_timestamp_absolute_us(time_us as i64);
        EndSlice {
            event: te,
            ctx: self,
        }
    }

    pub fn new_process(&mut self, pid: i32) -> u64 {
        let id = self.next_uuid();
        let mut td = TrackDescriptor::new();
        let mut pd = ProcessDescriptor::new();
        pd.set_pid(pid);
        td.set_uuid(id);
        td.process = MessageField::some(pd);
        let mut tp = TracePacket::new();
        tp.set_track_descriptor(td);
        self.sink.consume(tp);
        id
    }

    pub fn new_thread(&mut self, pid: i32, tid: i32, name: String) -> u64 {
        let id = self.next_uuid();
        let mut td = TrackDescriptor::new();
        td.set_uuid(id);
        let mut thd = ThreadDescriptor::new();
        thd.set_pid(pid);
        thd.set_tid(tid);
        thd.set_thread_name(name);
        td.thread = MessageField::some(thd);
        let mut tp = TracePacket::new();
        tp.set_track_descriptor(td);
        self.sink.consume(tp);
        id
    }

    pub fn new_track(&mut self, id: u64) -> Track<'_, T> {
        let mut td = TrackDescriptor::new();
        td.set_uuid(id);
        Track {
            ctx: self,
            desc: td,
        }
    }
    pub fn new_counter_track(&mut self, id: u64) -> CounterTrack<'_, T> {
        let mut td = TrackDescriptor::new();
        td.set_uuid(id);
        CounterTrack {
            ctx: self,
            desc: td,
        }
    }

    pub fn next_uuid(&self) -> u64 {
        self.next_uuid
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    pub fn consume(&mut self, mut packet: TracePacket) {
        if self.needs_flush() {
            self.flush_interned_data();
        }
        packet.set_trusted_packet_sequence_id(self.seq);
        packet.sequence_flags = Some(2); // needs interned data

        self.sink.consume(packet);
    }

    fn needs_flush(&self) -> bool {
        self.debug_annotation_names.needs_flush()
            | self.event_names.needs_flush()
            | self.event_categories.needs_flush()
            | self.debug_annotation_strings.needs_flush()
    }

    fn flush_interned_data(&mut self) {
        let mut id = TracePacket::new();
        id.interned_data = MessageField::some(self.interned_data());
        id.set_trusted_packet_sequence_id(self.seq);
        id.sequence_flags = Some(2);
        self.sink.consume(id);
        self.event_categories.flush();
        self.event_names.flush();
        self.debug_annotation_names.flush();
        self.debug_annotation_strings.flush();
    }

    fn interned_data(&self) -> InternedData {
        // Populate interned_data with the interned strings from the HashMaps
        let mut interned_data = InternedData::new();

        for (category, id) in self.event_categories.to_flush() {
            let mut event_category = EventCategory::new();
            event_category.set_iid(*id);
            event_category.set_name(category.clone());
            interned_data.event_categories.push(event_category);
        }

        for (name, id) in self.event_names.to_flush() {
            let mut event_name = EventName::new();
            event_name.set_iid(*id);
            event_name.set_name(name.clone());
            interned_data.event_names.push(event_name);
        }

        for (name, id) in self.debug_annotation_names.to_flush() {
            let mut debug_annotation_name = DebugAnnotationName::new();
            debug_annotation_name.set_iid(*id);
            debug_annotation_name.set_name(name.clone());
            interned_data
                .debug_annotation_names
                .push(debug_annotation_name);
        }
        for (val, id) in self.debug_annotation_names.to_flush() {
            let mut ii = InternedString::new();
            ii.set_iid(*id);
            ii.set_str(val.clone().into_bytes());
            interned_data.debug_annotation_string_values.push(ii);
        }

        interned_data
    }

    pub fn sink(self) -> T {
        self.sink
    }

    fn debug_annotation(&mut self, name: &str, value: &serde_json::Value) -> DebugAnnotation {
        let mut dbg = DebugAnnotation::new();
        let id = self.debug_annotation_names.intern(name);
        dbg.set_name_iid(id);
        match value {
            serde_json::Value::Null => {}
            serde_json::Value::Bool(b) => dbg.set_bool_value(*b),
            serde_json::Value::Number(number) => {
                if number.is_i64() {
                    dbg.set_int_value(number.as_i64().unwrap());
                } else {
                    dbg.set_double_value(number.as_f64().unwrap());
                }
            }
            serde_json::Value::String(s) => {
                let id = self.debug_annotation_names.intern(s.as_str());
                dbg.set_string_value_iid(id);
            }
            serde_json::Value::Array(values) => {
                for (name, val) in values.iter().enumerate() {
                    dbg.array_values
                        .push(self.debug_annotation(name.to_string().as_str(), val));
                }
            }
            serde_json::Value::Object(map) => {
                for (name, val) in map {
                    dbg.dict_entries
                        .push(self.debug_annotation(name.as_str(), val));
                }
            }
        }
        dbg
    }
}

struct Slice<'a, T: Sink> {
    ctx: &'a mut Ctx<T>,
    id: u64,
    parent: Option<u64>,
    name: u64,
}
