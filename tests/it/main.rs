//! Property tests for miniident's public API.
//!
//! Each module checks one area against an oracle or an algebraic invariant. Run them with a
//! high case count through `task test:prop`.

mod forgetting;
mod identifiability;
mod linalg;
mod prbs;
mod sensitivity;
mod strategies;
