#![allow(unused)]
#![feature(type_changing_struct_update, try_blocks)]
use std::collections::HashMap;

use cacheline_ef::CachelineEfVec;
use ptr_hash::{
    bucket_fn::{BucketFn, CubicEps, Linear, Optimal, Skewed, Square},
    hash::FxHash,
    pack::{EliasFano, MutPacked, Packed},
    stats::BucketStats,
    util::{self, generate_keys, time},
    PtrHash, PtrHashParams,
};
use serde::Serialize;

const SMALL_N: usize = 20_000_000;
const LARGE_N: usize = 1_000_000_000;

/// Experiments:
/// 1. bucket sizes & evictions during construction
/// 2. construction speed, datastructure size, and query throughput for various parameters
/// 3. remap types
fn main() {
    // bucket_fn_stats();
    size();
    // remap();
    // part_size();
}

#[derive(Debug, Serialize, Default)]
struct Result {
    n: usize,
    alpha: f64,
    lambda: f64,
    bucketfn: String,
    construction_1: f64,
    construction_6: f64,
    pilots: f64,
    remap: f64,
    remap_type: String,
    total: f64,
    q1_phf: f64,
    q1_mphf: f64,
    q32_phf: f64,
    q32_mphf: f64,
}

/// Collect stats on bucket sizes and number of evictions during construction.
/// Vary the bucket assignment function.
fn bucket_fn_stats() {
    type MyPtrHash<BF> = PtrHash<u64, BF, CachelineEfVec, FxHash, Vec<u8>>;
    let n = 1_000_000_000;
    let keys = &generate_keys(n);

    fn build(keys: &Vec<u64>, lambda: f64, bucket_fn: impl BucketFn) -> BucketStats {
        let params = PtrHashParams {
            print_stats: true,
            lambda,
            bucket_fn,
            ..PtrHashParams::default()
        };
        let (_ph, stats) = MyPtrHash::new_with_stats(&keys, params);
        stats
    }

    {
        let lambda = 3.5;
        let mut stats = HashMap::new();
        stats.insert("linear", build(keys, lambda, Linear));
        stats.insert("skewed", build(keys, lambda, Skewed::default()));
        stats.insert("square", build(keys, lambda, Square));
        stats.insert("cubic", build(keys, lambda, CubicEps));
        stats.insert("optimal", build(keys, lambda, Optimal { eps: 1. / 256. }));

        write(&stats, "data/bucket_fn_stats_l35.json");
    }

    {
        let lambda = 4.0;
        let mut stats = HashMap::new();
        stats.insert("cubic", build(keys, lambda, CubicEps));

        write(&stats, "data/bucket_fn_stats_l40.json");
    }
}

fn write<T: Serialize>(stats: &T, path: &str) {
    let json = serde_json::to_string(stats).unwrap();
    std::fs::write(path, json).unwrap();
}

/// Return:
/// Construction time (1 and 6 threads)
/// Space (pilots, remap, total)
/// Query throughput (32 streaming)
fn test<R: MutPacked + Send>(
    keys: &Vec<u64>,
    alpha: f64,
    lambda: f64,
    bucket_fn: impl BucketFn + Send,
    // Construct on one thread?
    c1: bool,
) -> Result {
    type MyPtrHash<BF, R> = PtrHash<u64, BF, R, FxHash, Vec<u8>>;

    let params = PtrHashParams {
        alpha,
        lambda,
        bucket_fn,
        ..PtrHashParams::default()
    };

    // Construct on 1 thread.
    let c1 = if c1 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| util::time(|| <MyPtrHash<_, R>>::new(&keys, params)).1)
    } else {
        0.
    };

    // Construct on 6 threads.
    let (ph, c6) = util::time(|| <MyPtrHash<_, R>>::new(&keys, params));

    // Space usage.
    let (pilots, remap) = ph.bits_per_element();
    let total = pilots + remap;

    // Single threaded query throughput, non-minimal and minimal.
    // Query all keys 'loops' times.
    let loops = 1_000_000_000 / keys.len();
    let (_, q1_phf) = time(|| {
        (0..loops)
            .map(|_| keys.iter().map(|key| ph.index(key)).sum::<usize>())
            .sum::<usize>()
    });
    let (_, q1_mphf) = time(|| {
        (0..loops)
            .map(|_| keys.iter().map(|key| ph.index_minimal(key)).sum::<usize>())
            .sum::<usize>()
    });
    let (_, q32_phf) = time(|| {
        (0..loops)
            .map(|_| ph.index_stream::<32, false>(keys).sum::<usize>())
            .sum::<usize>()
    });
    let (_, q32_mphf) = time(|| {
        (0..loops)
            .map(|_| ph.index_stream::<32, true>(keys).sum::<usize>())
            .sum::<usize>()
    });

    Result {
        n: keys.len(),
        alpha,
        lambda,
        construction_1: c1,
        construction_6: c6,
        bucketfn: format!("{bucket_fn:?}"),
        remap_type: R::name(),
        pilots,
        remap,
        total,
        q1_phf,
        q1_mphf,
        q32_phf,
        q32_mphf,
    }
}

/// Construction time&space for various lambda.
fn size() {
    fn test<R: MutPacked>(
        keys: &Vec<u64>,
        alpha: f64,
        lambda: f64,
        bucket_fn: impl BucketFn,
    ) -> Option<Result> {
        type MyPtrHash<BF, R> = PtrHash<u64, BF, R, FxHash, Vec<u8>>;

        let params = PtrHashParams {
            alpha,
            lambda,
            bucket_fn,
            ..PtrHashParams::default()
        };
        eprintln!("Running {alpha} {lambda} {bucket_fn:?}");
        // Construct on 6 threads.
        let (ph, c6) = util::time(|| <MyPtrHash<_, R>>::try_new(&keys, params));

        // Space usage.
        let (pilots, remap) = ph.map(|ph| ph.bits_per_element())?;
        let total = pilots + remap;
        let r = Result {
            n: keys.len(),
            alpha,
            lambda,
            construction_6: c6,
            bucketfn: format!("{bucket_fn:?}"),
            pilots,
            remap,
            total,
            ..Result::default()
        };
        eprintln!("Result: {r:?}");
        Some(r)
    }

    let n = LARGE_N;
    let mut results = vec![];
    let keys = &generate_keys(n);
    for alpha in [0.98, 0.99, 0.995] {
        for lambda in 27..45 {
            let lambda = lambda as f64 / 10.;
            let Some(r) = test::<Vec<u32>>(keys, alpha, lambda, Linear) else {
                break;
            };
            results.push(r);
        }
        for lambda in 27..45 {
            let lambda = lambda as f64 / 10.;
            let Some(r) = test::<CachelineEfVec>(keys, alpha, lambda, CubicEps) else {
                break;
            };
            results.push(r);
        }
    }
    write(&results, &format!("data/size.json"));
}

/// Collect stats on size and query speed, varying alpha and lambda.
fn remap() {
    let mut results = vec![];
    let n = LARGE_N;
    let keys = &generate_keys(n);

    {
        let alpha = 0.99;
        let lambda = 3.0;
        results.push(test::<Vec<u32>>(keys, alpha, lambda, Linear, false));
        results.push(test::<CachelineEfVec>(keys, alpha, lambda, Linear, false));
        results.push(test::<EliasFano>(keys, alpha, lambda, Linear, false));
    }
    {
        let alpha = 0.98;
        let lambda = 4.0;
        results.push(test::<Vec<u32>>(keys, alpha, lambda, CubicEps, false));
        results.push(test::<CachelineEfVec>(keys, alpha, lambda, CubicEps, false));
        results.push(test::<EliasFano>(keys, alpha, lambda, CubicEps, false));
    }
    write(&results, "data/remap.json");
}

fn part_size() {
    fn test(
        keys: &Vec<u64>,
        alpha: f64,
        lambda: f64,
        bucket_fn: impl BucketFn,
        slots_per_part: usize,
    ) -> Option<Result> {
        type MyPtrHash<BF> = PtrHash<u64, BF, CachelineEfVec, FxHash, Vec<u8>>;

        let params = PtrHashParams {
            alpha,
            lambda,
            bucket_fn,
            slots_per_part,
            ..PtrHashParams::default()
        };
        eprintln!("Running {alpha} {lambda} {bucket_fn:?} {slots_per_part}");
        // Construct on 6 threads.
        let (ph, c6) = util::time(|| MyPtrHash::try_new(&keys, params));

        // Space usage.
        let (pilots, remap) = ph.map(|ph| ph.bits_per_element())?;
        let total = pilots + remap;
        let r = Result {
            n: keys.len(),
            alpha,
            lambda,
            construction_6: c6,
            bucketfn: format!("{bucket_fn:?}"),
            pilots,
            remap,
            total,
            ..Result::default()
        };
        eprintln!("Result: {r:?}");
        Some(r)
    }

    let mut results = vec![];
    let n = LARGE_N;
    let keys = &generate_keys(n);

    for s in 15..=20 {
        let alpha = 0.99;
        let lambda = 3.0;
        results.push(test(keys, alpha, lambda, Linear, 1 << s));
    }
    for s in 15..=20 {
        let alpha = 0.98;
        let lambda = 4.0;
        results.push(test(keys, alpha, lambda, CubicEps, 1 << s));
    }
    write(&results, "data/slots_per_part.json");
}
