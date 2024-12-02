#![feature(iter_array_chunks)]
//! PTRHash is a minimal perfect hash function.
//!
//! Usage example:
//! ```rust
//! use ptr_hash::{PtrHash, PtrHashParams};
//!
//! // Generate some random keys.
//! let n = 1_000_000_000;
//! let keys = ptr_hash::util::generate_keys(n);
//!
//! // Build the datastructure.
//! let mphf = <PtrHash>::new(&keys, PtrHashParams::default());
//!
//! // Get the minimal index of a key.
//! let key = 0;
//! let idx = mphf.index_minimal(&key);
//! assert!(idx < n);
//!
//! // Get the non-minimal index of a key. Slightly faster.
//! let _idx = mphf.index(&key);
//!
//! // An iterator over the indices of the keys.
//! // 32: number of iterations ahead to prefetch.
//! // true: remap to a minimal key in [0, n).
//! let indices = mphf.index_stream::<32, true>(&keys);
//! assert_eq!(indices.sum::<usize>(), (n * (n - 1)) / 2);
//!
//! // Test that all items map to different indices
//! let mut taken = vec![false; n];
//! for key in keys {
//!     let idx = mphf.index_minimal(&key);
//!     assert!(!taken[idx]);
//!     taken[idx] = true;
//! }
//! ```
//#![cfg_attr(target_arch = "aarch64", feature(stdsimd))]
#![allow(clippy::needless_range_loop)]

/// Customizable Hasher trait.
pub mod hash;
/// Extendable backing storage trait and types.
pub mod pack;
/// Some internal logging and testing utilities.
pub mod util;

pub mod bucket_fn;
mod bucket_idx;
mod build;
mod reduce;
mod shard;
mod sort_buckets;
mod stats;
#[cfg(test)]
mod test;

use bitvec::{bitvec, vec::BitVec};
use bucket_fn::BucketFn;
use bucket_fn::Linear;
use cacheline_ef::CachelineEfVec;
use either::Either;
use itertools::izip;
use itertools::Itertools;
use pack::EliasFano;
use pack::MutPacked;
use rand::{random, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use rdst::RadixSort;
use std::{borrow::Borrow, default::Default, marker::PhantomData, time::Instant};

use crate::{hash::*, pack::Packed, reduce::*, util::log_duration};

/// Select the sharding method to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, Default)]
pub enum Sharding {
    #[default]
    None,
    Memory,
    Disk,
}

/// Parameters for PtrHash construction.
///
/// Since these are not used in inner loops they are simple variables instead of template arguments.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
pub struct PtrHashParams<BF> {
    /// Set to false to disable remapping to a minimal PHF.
    pub remap: bool,
    /// Use `n/alpha` slots approximately.
    pub alpha: f64,
    /// Use `c*n/lg(n)` buckets.
    pub c: f64,
    /// Bucket function
    pub bucket_fn: BF,
    /// #slots/part will be the largest power of 2 not larger than this.
    /// Default is 2^18.
    pub slots_per_part: usize,
    /// Upper bound on number of keys per shard.
    /// Default is 2^33, or 32GB of hashes per shard.
    pub keys_per_shard: usize,
    /// When true, write each shard to a file instead of iterating multiple
    /// times.
    pub sharding: Sharding,

    /// Print bucket size and pilot stats after construction.
    pub print_stats: bool,
}

/// Default parameter values should provide reasonably fast construction for all n up to 2^32:
/// - `alpha=0.98`
/// - `c=9.0`
/// - `slots_per_part=2^18=262144`
impl Default for PtrHashParams<Linear> {
    fn default() -> Self {
        Self {
            remap: true,
            alpha: 0.98,
            c: 9.0,
            bucket_fn: Linear,
            slots_per_part: 1 << 18,
            // By default, limit to 2^32 keys per shard, whose hashes take 8B*2^32=32GB.
            keys_per_shard: 1 << 32,
            sharding: Sharding::None,
            print_stats: false,
        }
    }
}

// Externally visible aliases for convenience.

/// An alias for PtrHash with default generic arguments.
/// Using this, you can write `DefaultPtrHash::new()` instead of `<PtrHash>::new()`.
pub type DefaultPtrHash<H, Key> = PtrHash<Key, CachelineEfVec, H, Vec<u8>>;

/// Using EliasFano for the remap is slower but uses slightly less memory.
pub type EfPtrHash<H, Key> = PtrHash<Key, EliasFano, H, Vec<u8>>;

/// Trait that keys must satisfy.
pub trait KeyT: Default + Send + Sync + std::hash::Hash {}
impl<T: Default + Send + Sync + std::hash::Hash> KeyT for T {}

// Some fixed algorithmic decisions.
type Rp = FastReduce;
type Rb = FastReduce;
type Rs = MulReduce;
type Pilot = u64;
type PilotHash = u64;

/// PtrHash datastructure.
/// The recommended way to use PtrHash with default types.
///
/// `F`: The packing to use for remapping free slots, default `TinyEf`.
/// `Hx`: The hasher to use for keys, default `FxHash`.
/// `V`: The pilots type. Usually `Vec<u8>`, or `&[u8]` for Epserde.
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[derive(Clone)]
pub struct PtrHash<
    Key: KeyT = u64,
    BF: BucketFn = bucket_fn::Linear,
    F: Packed = CachelineEfVec,
    Hx: Hasher<Key> = hash::FxHash,
    V: AsRef<[u8]> = Vec<u8>,
> {
    params: PtrHashParams<BF>,

    /// The number of keys.
    n: usize,
    /// The total number of parts.
    num_parts: usize,
    /// The number of shards.
    num_shards: usize,
    /// The maximal number of parts per shard.
    /// The last shard may have fewer parts.
    parts_per_shard: usize,
    /// The total number of slots.
    s_total: usize,
    /// The total number of buckets.
    b_total: usize,
    /// The number of slots per part, always a power of 2.
    s: usize,
    /// Since s is a power of 2, we can compute multiplications using a shift
    /// instead.
    s_bits: u32,
    /// The number of buckets per part.
    b: usize,

    // Precomputed fast modulo operations.
    /// Fast %shards.
    rem_shards: Rp,
    /// Fast %parts.
    rem_parts: Rp,
    /// Fast &b.
    rem_b: Rb,
    /// Fast &b_total.
    rem_b_total: Rb,

    /// Fast %s.
    rem_s: Rs,

    // Computed state.
    /// The global seed.
    seed: u64,
    /// The pilots.
    pilots: V,
    /// Remap the out-of-bound slots to free slots.
    remap: F,
    _key: PhantomData<Key>,
    _hx: PhantomData<Hx>,
}

/// Construction methods.
impl<Key: KeyT, BF: BucketFn, F: MutPacked, Hx: Hasher<Key>> PtrHash<Key, BF, F, Hx, Vec<u8>> {
    /// Create a new PtrHash instance from the given keys.
    ///
    /// NOTE: Only up to 2^40 keys are supported.
    ///
    /// Default parameters `alpha=0.98` and `c=9.0` should give fast
    /// construction that always succeeds, using `2.69 bits/key`.  Depending on
    /// the number of keys, you may be able to lower `c` (or slightly increase
    /// `alpha`) to reduce memory usage, at the cost of increasing construction
    /// time.
    ///
    /// By default, keys are partitioned into buckets of size ~250000, and parts are processed in parallel.
    /// This will use all available threads. To limit to fewer threads, use:
    /// ```rust
    /// let threads = 1;
    /// rayon::ThreadPoolBuilder::new()
    /// .num_threads(threads)
    /// .build_global()
    /// .unwrap();
    /// ```
    ///
    /// NOTE: Use `<PtrHash>::new()` or `DefaultPtrHash::new()` instead of simply `PtrHash::new()`.
    pub fn new(keys: &[Key], params: PtrHashParams<BF>) -> Self {
        let mut ptr_hash = Self::init(keys.len(), params);
        ptr_hash.compute_pilots(keys.par_iter());
        ptr_hash
    }

    /// Same as `new` above, but takes a `ParallelIterator` over keys instead of a slice.
    /// The iterator must be cloneable for two reasons:
    /// - Construction can fail for the first seed (e.g. due to duplicate
    ///   hashes), in which case a new pass over keys is need.
    /// NOTE: The exact API may change here depending on what's most convenient to use.
    pub fn new_from_par_iter<'a>(
        n: usize,
        keys: impl ParallelIterator<Item = impl Borrow<Key>> + Clone + 'a,
        params: PtrHashParams<BF>,
    ) -> Self {
        let mut ptr_hash = Self::init(n, params);
        ptr_hash.compute_pilots(keys);
        ptr_hash
    }

    /// PtrHash with random pilots, for benchmarking query speed.
    pub fn new_random(n: usize, params: PtrHashParams<BF>) -> Self {
        let mut ptr_hash = Self::init(n, params);
        let k = (0..ptr_hash.b_total)
            .map(|_| random::<u8>() as Pilot)
            .collect();
        ptr_hash.pilots = MutPacked::new(k);
        let rem_s_total = FastReduce::new(ptr_hash.s_total);
        let mut remap_vals = (ptr_hash.n..ptr_hash.s_total)
            .map(|_| rem_s_total.reduce(random::<u64>()) as _)
            .collect_vec();
        remap_vals.radix_sort_unstable();
        ptr_hash.remap = MutPacked::new(remap_vals);
        ptr_hash.print_bits_per_element();
        ptr_hash
    }

    /// Only initialize the parameters; do not compute the pilots yet.
    fn init(n: usize, mut params: PtrHashParams<BF>) -> Self {
        assert!(n > 1, "Things break if n=1.");
        assert!(n < (1 << 40), "Number of keys must be less than 2^40.");

        // Target number of slots in total over all parts.
        let s_total_target = (n as f64 / params.alpha) as usize;

        // Target number of buckets in total.
        let b_total_target = params.c * (n as f64) / (n as f64).log2();

        assert!(
            params.slots_per_part <= u32::MAX as _,
            "Each part must have <2^32 slots"
        );
        // We start with the given maximum number of slots per part, since
        // that is what should fit in L1 or L2 cache.
        // Thus, the number of partitions is:
        let s = 1 << params.slots_per_part.ilog2();
        if let Sharding::None = params.sharding {
            params.keys_per_shard = n;
        }
        let num_shards = n.div_ceil(params.keys_per_shard);
        let parts_per_shard = s_total_target.div_ceil(s).div_ceil(num_shards);
        let num_parts = num_shards * parts_per_shard;

        let s_total = s * num_parts;
        // b divisible by 3 is exploited by bucket_thirds.
        let b = ((b_total_target / (num_parts as f64)).ceil() as usize).next_multiple_of(3);
        let b_total = b * num_parts;
        // TODO: Figure out if large gcd(b,s) is a problem for the original PtrHash.

        if params.print_stats {
            eprintln!("        keys: {n:>10}");
            eprintln!("      shards: {num_shards:>10}");
            eprintln!("       parts: {num_parts:>10}");
            eprintln!("   slots/prt: {s:>10}");
            eprintln!("   slots tot: {s_total:>10}");
            eprintln!(" buckets/prt: {b:>10}");
            eprintln!(" buckets tot: {b_total:>10}");
            eprintln!("keys/ bucket: {:>13.2}", n as f64 / b_total as f64);
        }
        params.bucket_fn.set_buckets_per_part(b as u64);

        Self {
            params,
            n,
            num_parts,
            num_shards,
            parts_per_shard,
            s_total,
            s,
            s_bits: s.ilog2(),
            b_total,
            b,
            rem_shards: Rp::new(num_shards),
            rem_parts: Rp::new(num_parts),
            rem_b: Rb::new(b),
            rem_b_total: Rb::new(b_total),
            rem_s: Rs::new(s),
            seed: 0,
            pilots: Default::default(),
            remap: F::default(),
            _key: PhantomData,
            _hx: PhantomData,
        }
    }

    fn compute_pilots<'a>(
        &mut self,
        keys: impl ParallelIterator<Item = impl Borrow<Key>> + Clone + 'a,
    ) {
        let overall_start = std::time::Instant::now();
        // Initialize arrays;
        let mut taken: Vec<BitVec> = vec![];
        let mut pilots: Vec<u8> = vec![];

        let mut tries = 0;
        const MAX_TRIES: usize = 3;

        let mut rng = ChaCha8Rng::seed_from_u64(31415);

        // Loop over global seeds `s`.
        's: loop {
            tries += 1;
            assert!(
                tries <= MAX_TRIES,
                "Failed to find a global seed after {MAX_TRIES} tries for {} keys.",
                self.s
            );
            if tries > 1 {
                eprintln!("Try {tries} for global seed.");
            }

            // Choose a global seed s.
            self.seed = rng.gen();

            // Reset output-memory.
            pilots.clear();
            pilots.resize(self.b_total, 0);

            for taken in taken.iter_mut() {
                taken.clear();
                taken.resize(self.s, false);
            }
            taken.resize_with(self.num_parts, || bitvec![0; self.s]);

            // Iterate over shards.
            let shard_hashes = match self.params.sharding {
                Sharding::None => Either::Left(self.no_sharding(keys.clone())),
                Sharding::Memory => {
                    Either::Right(Either::Left(self.shard_keys_to_disk(keys.clone())))
                }
                Sharding::Disk => {
                    Either::Right(Either::Right(self.shard_keys_in_memory(keys.clone())))
                }
            };
            let shard_pilots = pilots.chunks_mut(self.b * self.parts_per_shard);
            let shard_taken = taken.chunks_mut(self.parts_per_shard);
            // eprintln!("Num shards (keys) {}", shard_keys.());
            for (shard, (hashes, pilots, taken)) in
                izip!(shard_hashes, shard_pilots, shard_taken).enumerate()
            {
                // Determine the buckets.
                let start = std::time::Instant::now();
                let Some((hashes, part_starts)) = self.sort_parts(shard, hashes) else {
                    // Found duplicate hashes.
                    continue 's;
                };
                let start = log_duration("sort buckets", start);

                // Compute pilots.
                if !self.build_shard(shard, &hashes, &part_starts, pilots, taken) {
                    continue 's;
                }
                log_duration("find pilots", start);
            }

            // Found a suitable seed.
            if tries > 1 {
                eprintln!("Found seed after {tries} tries.");
            }

            break 's;
        }

        let start = std::time::Instant::now();
        self.remap_free_slots(taken);
        log_duration("remap free", start);

        // Pack the data.
        self.pilots = pilots;

        self.print_bits_per_element();
        log_duration("total build", overall_start);
    }

    fn remap_free_slots(&mut self, taken: Vec<BitVec>) {
        assert_eq!(
            taken.iter().map(|t| t.count_zeros()).sum::<usize>(),
            self.s_total - self.n,
            "Not the right number of free slots left!\n total slots {} - n {}",
            self.s_total,
            self.n
        );

        if !self.params.remap || self.s_total == self.n {
            return;
        }

        // Compute the free spots.
        let mut v = Vec::with_capacity(self.s_total - self.n);
        let get = |t: &Vec<BitVec>, idx: usize| t[idx / self.s][idx % self.s];
        for i in taken
            .iter()
            .enumerate()
            .flat_map(|(p, t)| {
                let offset = p * self.s;
                t.iter_zeros().map(move |i| offset + i)
            })
            .take_while(|&i| i < self.n)
        {
            while !get(&taken, self.n + v.len()) {
                v.push(i as u64);
            }
            v.push(i as u64);
        }
        self.remap = MutPacked::new(v);
    }
}

/// Indexing methods.
impl<Key: KeyT, BF: BucketFn, F: Packed, Hx: Hasher<Key>, V: AsRef<[u8]>>
    PtrHash<Key, BF, F, Hx, V>
{
    /// Return the number of bits per element used for the pilots (`.0`) and the
    /// remapping (`.1)`.
    pub fn bits_per_element(&self) -> (f32, f32) {
        let pilots = self.pilots.as_ref().size_in_bytes() as f32 / self.n as f32;
        let remap = self.remap.size_in_bytes() as f32 / self.n as f32;
        (8. * pilots, 8. * remap)
    }

    pub fn print_bits_per_element(&self) {
        let (p, r) = self.bits_per_element();
        if self.params.print_stats {
            eprintln!(
                "bits/element: {:>13.2}  (pilots {p:4.2}, remap {r:4.2})",
                p + r
            );
        }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    /// index() always returns below this bound.
    pub fn max_index(&self) -> usize {
        self.s_total
    }

    /// Get a non-minimal index of the given key.
    /// Use `index_minimal` to get a key in `[0, n)`.
    #[inline]
    pub fn index(&self, key: &Key) -> usize {
        let hx = self.hash_key(key);
        let b = self.bucket(hx);
        let pilot = self.pilots.as_ref().index(b);
        self.slot(hx, pilot)
    }

    /// Faster version of `index` for when there is only a single part.
    #[inline]
    pub fn index_single_part(&self, key: &Key) -> usize {
        let hx = self.hash_key(key);
        let b = self.bucket_in_part(hx.high());
        let pilot = self.pilots.as_ref().index(b);
        self.slot_in_part(hx, pilot)
    }

    /// Get the index for `key` in `[0, n)`.
    ///
    /// Requires that the remap parameter is set to true.
    #[inline]
    pub fn index_minimal(&self, key: &Key) -> usize {
        let hx = self.hash_key(key);
        let b = self.bucket(hx);
        let p = self.pilots.as_ref().index(b);
        let slot = self.slot(hx, p);
        if slot < self.n {
            slot
        } else {
            self.remap.index(slot - self.n) as usize
        }
    }

    /// Takes an iterator over keys and returns an iterator over the indices of the keys.
    ///
    /// Uses a buffer of size K for prefetching ahead.
    // NOTE: It would be cool to use SIMD to determine buckets/positions in
    // parallel, but this is complicated, since SIMD doesn't support the
    // 64x64->128 multiplications needed in bucket/slot computations.
    #[inline]
    pub fn index_stream<'a, const B: usize, const MINIMAL: bool>(
        &'a self,
        keys: impl IntoIterator<Item = &'a Key> + 'a,
    ) -> impl Iterator<Item = usize> + 'a {
        // Append B values at the end of the iterator to make sure we wrap sufficiently.
        let mut hashes = keys
            .into_iter()
            .map(|x| self.hash_key(x))
            .chain([Default::default(); B]);

        // Ring buffers to cache the hash and bucket of upcoming queries.
        let mut next_hashes: [Hx::H; B] = [Hx::H::default(); B];
        let mut next_buckets: [usize; B] = [0; B];

        // Initialize and prefetch first B values.
        for idx in 0..B {
            next_hashes[idx] = hashes.next().unwrap();
            next_buckets[idx] = self.bucket(next_hashes[idx]);
            crate::util::prefetch_index(self.pilots.as_ref(), next_buckets[idx]);
        }
        hashes.enumerate().map(move |(idx, next_hash)| {
            let idx = idx % B;
            let cur_hash = next_hashes[idx];
            let cur_bucket = next_buckets[idx];
            next_hashes[idx] = next_hash;
            next_buckets[idx] = self.bucket(next_hashes[idx]);
            crate::util::prefetch_index(self.pilots.as_ref(), next_buckets[idx]);
            let pilot = self.pilots.as_ref().index(cur_bucket);
            // NOTE: Caching `part` slows things down, so it's recomputed as part of `self.slot`.
            // TODO: Verify
            let slot = self.slot(cur_hash, pilot);
            if MINIMAL && slot >= self.n {
                self.remap.index(slot - self.n) as usize
            } else {
                slot
            }
        })
    }

    /// Takes an iterator over keys and returns an iterator over the indices of the keys.
    ///
    /// Queries in batches of size K.
    ///
    /// NOTE: Does not process the remainder
    #[inline]
    pub fn index_batch_exact<'a, const K: usize, const MINIMAL: bool>(
        &'a self,
        xs: impl IntoIterator<Item = &'a Key> + 'a,
    ) -> impl Iterator<Item = usize> + 'a {
        let mut buckets: [usize; K] = [0; K];

        // Work on chunks of size K.
        let mut f = move |hx: [Hx::H; K]| {
            // Prefetch.
            for idx in 0..K {
                buckets[idx] = self.bucket(hx[idx]);
                crate::util::prefetch_index(self.pilots.as_ref(), buckets[idx]);
            }
            // Query.
            (0..K).map(move |idx| {
                let pilot = self.pilots.as_ref().index(buckets[idx]);
                let slot = self.slot(hx[idx], pilot);
                if MINIMAL && slot >= self.n {
                    self.remap.index(slot - self.n) as usize
                } else {
                    slot
                }
            })
        };
        let array_chunks = xs.into_iter().map(|x| self.hash_key(x)).array_chunks::<K>();
        array_chunks.into_iter().flat_map(move |chunk| f(chunk))
        // .chain(f(&array_chunks
        //     .into_remainder()
        //     .unwrap_or_default()
        //     .into_iter()))
    }

    fn hash_key(&self, x: &Key) -> Hx::H {
        Hx::hash(x, self.seed)
    }

    fn hash_pilot(&self, p: Pilot) -> PilotHash {
        MulHash::hash(&p, self.seed)
    }

    fn shard(&self, hx: Hx::H) -> usize {
        self.rem_shards.reduce(hx.high())
    }

    fn part(&self, hx: Hx::H) -> usize {
        self.rem_parts.reduce(hx.high())
    }

    /// Map `hx_remainder` to a bucket in the range [0, self.b).
    /// Hashes <self.p1 are mapped to large buckets [0, self.p2).
    /// Hashes >=self.p1 are mapped to small [self.p2, self.b).
    ///
    /// (Unless SPLIT_BUCKETS is false, in which case all hashes are mapped to [0, self.b).)
    fn bucket_in_part(&self, x: u64) -> usize {
        if BF::B_OUTPUT {
            self.params.bucket_fn.call(x) as usize
        } else {
            self.rem_b.reduce(self.params.bucket_fn.call(x))
        }
    }

    /// See bucket.rs for additional implementations.
    /// Returns the offset in the slots array for the current part and the bucket index.
    fn bucket(&self, hx: Hx::H) -> usize {
        if BF::LINEAR {
            return self.rem_b_total.reduce(hx.high());
        }

        // Extract the high bits for part selection; do normal bucket
        // computation within the part using the remaining bits.
        // NOTE: This is somewhat slow, but doing better is hard.
        let (part, hx) = self.rem_parts.reduce_with_remainder(hx.high());
        let bucket = self.bucket_in_part(hx);
        part * self.b + bucket
    }

    /// Slot uses the 64 low bits of the hash.
    fn slot(&self, hx: Hx::H, pilot: u64) -> usize {
        (self.part(hx) << self.s_bits) + self.slot_in_part(hx, pilot)
    }

    fn slot_in_part(&self, hx: Hx::H, pilot: Pilot) -> usize {
        self.slot_in_part_hp(hx, self.hash_pilot(pilot))
    }

    /// Slot uses the 64 low bits of the hash.
    fn slot_in_part_hp(&self, hx: Hx::H, hp: PilotHash) -> usize {
        // NOTE: Fastmod s is slower since it needs two multiplications instead of 1.
        // NOTE: A simple &(s-1) mask is not sufficient, since it only uses the low order bits.
        //       The part() and bucket() functions only use high order bits, which
        //       would leave the middle bits unused, causing hash collisions.
        self.rem_s.reduce(hx.low() ^ hp)
    }
}
