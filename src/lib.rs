mod failover;
mod gear;
mod slot;
mod wheel;

use std::time::Instant;

pub use failover::*;
pub use wheel::*;

#[derive(Debug, Clone, Copy, thiserror::Error)]
#[error("{0} timers failed to reschedule")]
pub struct PollError(usize);

pub trait Timer {
    type Context;
    fn fire(&mut self, now: Instant, ctx: &mut Self::Context) -> Option<Instant>;
}

/// Handle for cancelling a pending timer.
///
/// # Ownership Semantics
///
/// `TimerHandle` is intentionally not `Clone` or `Copy`. This ensures:
///
/// - **Unique ownership**: Only one handle exists per scheduled timer.
/// - **Single cancellation**: Once `cancel()` consumes the handle, no
///   double-cancel is possible.
///
/// # Creation
///
/// Handles are only created by [`BitWheelInner::insert`]. The internal
/// fields (gear, slot, key, when_offset) are not publicly constructible,
/// guaranteeing that any handle passed to `cancel()` refers to a valid
/// timer entry (assuming it hasn't already fired).
///
/// # Validity
///
/// A handle becomes invalid once its `when_offset` tick has passed
/// (i.e., the timer has fired). Calling `cancel()` on an expired handle
/// safely returns `None`.
#[derive(Debug, PartialEq, Eq, Hash)]
#[repr(C, align(16))]
pub struct TimerHandle {
    when_offset: u64,
    key: u32,
    gear: u8,
    slot: u8,
    overflow: bool,
}
