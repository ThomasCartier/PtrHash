//! Implementations of various hashers to use with PtrHash.
//!
//! ## Integer keys
//!
//! For hashing keys, prefer [`IntHash`].
//! If input keys are sufficiently random, [`RandomIntHash`] is slightly faster, which is an alias for [`FxHash`].
//! For truly random keys, one could even use [`NoHash`], but this is mostly just a baseline for benchmarks.
//!
//! If [`IntHash`] lacks sufficient randomness, use [`Xxh3Int`] instead, which is considerably slower but much stronger.
//!
//! ## String keys
//!
//! For string keys, use [`StringHash`] for 64-bit hashes and [`StringHash128`] for 128-bit hashes.
//! These are aliases for 64bit and 128bit versions of xxhash (XXH3), respectively.
//!
//! Another option is to use [`FxHash`] instead.
//!
//! In general any type implementing `Hasher` can be used, but it may be more
//! efficient to implement [`KeyHasher`] yourself for your key type, to directly
//! call specialized functions rather than going through the generic `Hasher`
//! interface.
//!
use crate::KeyT;
use std::{borrow::Borrow, collections::HashMap, fmt::Debug};

/// The [`KeyHasher`] trait returns a 64 or 128-bit `Hash`. From this, two `u64` values are extracted.
///
/// When 64-bit hashes are enough, we simply return the same hash (the `u64`
/// `Self` value) as the low and high part.
///
/// When 128-bit hashes are needed, the two functions return the low/high half of bits.
///
/// Our method never needs the full hash value, and instead uses the two hashes
/// in different places to extract sufficient entropy.
pub trait Hash: Copy + Debug + Default + Send + Sync + Eq + rdst::RadixKey {
    /// Returns the low 64bits of the hash.
    fn low(&self) -> u64;
    /// Returns the high 64bits of the hash.
    fn high(&self) -> u64;
}

impl Hash for u64 {
    fn low(&self) -> u64 {
        *self
    }
    fn high(&self) -> u64 {
        *self
    }
}

impl Hash for u128 {
    fn low(&self) -> u64 {
        *self as u64
    }
    fn high(&self) -> u64 {
        (*self >> 64) as u64
    }
}

/// Wrapper trait for various hash functions.
pub trait KeyHasher<Key: ?Sized>: Clone + Sync {
    type H: Hash;
    fn hash(x: &Key, seed: u64) -> Self::H;
}

/// All external hashers work.
impl<Key: KeyT, H: core::hash::Hasher + Default + Clone + Sync> KeyHasher<Key> for H {
    type H = u64;
    #[inline(always)]
    fn hash(x: &Key, seed: u64) -> u64 {
        let mut hasher = H::default();
        Key::hash(x, &mut hasher);
        hasher.finish() ^ seed
    }
}

// Aliases

/// A slightly faster but weaker hash for sufficiently random integers. Uses [`fxhash::FxHasher64`].
pub type RandomIntHash = fxhash::FxHasher64;
pub type FxHash = fxhash::FxHasher64;
/// Use [`xxhash_rust::xxh3::Xxh3Default`] implementation of XXH3 for 64-bit string hashing.
pub type StringHash = xxhash_rust::xxh3::Xxh3Default;
/// Type alias for xxhash (XXH3) hasher.
///
/// Prefer [`Xxh3Int`] for integers, which avoids some overhead of the default hasher.
pub type Xxh3 = xxhash_rust::xxh3::Xxh3Default;
/// Use [`xxhash_rust::xxh3::Xxh3Default`] implementation of XXH3 for 128-bit string hashing.
pub type StringHash128 = Xxh3_128;

// Implementations

/// 128-bit version of XXH3.
#[derive(Clone)]
pub struct Xxh3_128;
impl<Key: KeyT> KeyHasher<Key> for Xxh3_128 {
    type H = u128;
    #[inline(always)]
    fn hash(x: &Key, seed: u64) -> u128 {
        let mut hasher = xxhash_rust::xxh3::Xxh3Default::default();
        x.hash(&mut hasher);
        hasher.digest128() ^ (seed as u128 | (seed as u128) << 64)
    }
}

/// A sufficiently good hash for non-random integers.
///
/// ```ignore
/// let (hi, lo) = (value ^ seed).wrapping_mul(C);
/// return (hi ^ lo).wrapping_mul(C);
/// ```
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[derive(Clone)]
pub struct IntHash;

/// Mixing constant.
pub const C: u64 = 0x517cc1b727220a95;

/// No hash at all; just `value ^ seed`. Use with caution. Mostly for benchmarking.
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[derive(Clone)]
pub struct NoHash;

/// Inlined version of Xxh3 for integer keys.
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[derive(Clone)]
pub struct Xxh3Int;

// Macro to implement hashes for all integer types.
macro_rules! int_hashers {
    ($($t:ty),*) => {
        $(
            impl KeyHasher<$t> for NoHash {
                type H = u64;
                #[inline(always)]
                fn hash(x: &$t, seed: u64) -> u64 {
                    *x as u64 ^ seed
                }
            }

            impl KeyHasher<$t> for IntHash {
                type H = u64;
                #[inline(always)]
                fn hash(x: &$t, seed: u64) -> u64 {
                    let r = (*x as u64 ^ seed) as u128 * C as u128;
                    let low = r as u64;
                    let high = (r >> 64) as u64;
                    (low ^ high).wrapping_mul(C)
                }
            }

            impl KeyHasher<$t> for Xxh3Int {
                type H = u64;
                #[inline(always)]
                fn hash(x: &$t, seed: u64) -> u64 {
                    xxhash_rust::xxh3::xxh3_64_with_seed(&(*x as u64).to_le_bytes(), seed)
                }
            }
        )*
    };
}
int_hashers!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize);
