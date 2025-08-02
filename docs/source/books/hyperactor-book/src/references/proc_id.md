# `ProcId`

A `ProcId` identifies a single runtime instance within a world. All actors exist within a proc, and message routing between actors is scoped by the proc’s identity.
```rust
#[derive(
    Debug,
    Serialize,
    Deserialize,
    Clone,
    PartialEq,
    Eq,
    PartialOrd,
    Hash,
    Ord,
    Named
)]
pub struct ProcId(pub WorldId, pub usize);
```

## Construction

You can construct a `ProcId` directly:
```rust
use hyperactor::reference::{WorldId, ProcId};

let proc = ProcId(WorldId("training".into()), 0);
```
Or statically using the `id!` macro:
```rust
use hyperactor::id;

let proc = id!(training[0]); // Equivalent to ProcId(WorldId("training".into()), 0)
```

## Methods

```rust
impl ProcId {
    pub fn world_id(&self) -> &WorldId;
    pub fn world_name(&self) -> &str;
    pub fn rank(&self) -> usize;
    pub fn actor_id(&self, name: impl Into<String>, pid: usize) -> ActorId;
}
```
- `.world_id()` gives the `WorldId` this proc belongs to.
- `.rank()` returns the proc’s index.
- `.actor_id(name, pid)` constructs an `ActorId` for an actor hosted on this proc.

# Notes

Ranks greater than or equal to `1 << (usize::BITS - 1)` are considered user-space procs. These are typically created with `WorldId::random_user_proc()` and are not assigned by the system.

## Traits

ProcId implements:
- `Display` — formatted as `world[rank]`
- `FromStr` — parses from strings like "training[0]"
- `Ord`, `Eq`, `Hash` — usable in maps and sorted structures
- `Named` — enables port lookup and type reflection
