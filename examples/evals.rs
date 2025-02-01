#![allow(unused)]
#![feature(type_changing_struct_update, try_blocks)]
use std::{cmp::min, collections::HashMap, hint::black_box, time::Instant};

use cacheline_ef::CachelineEfVec;
use ptr_hash::{
    bucket_fn::{self, BucketFn, CubicEps, Linear, Optimal, Skewed, Square},
    hash::{FxHash, Murmur2_64},
    pack::{EliasFano, MutPacked, Packed},
    stats::BucketStats,
    util::{self, generate_keys, time},
    PtrHash, PtrHashParams, Sharding,
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::Serialize;

/// Experiments:
/// 1. bucket sizes & evictions during construction
/// 2. construction speed, datastructure size, and query throughput for various parameters
/// 3. remap types
fn main() {
    // 4.1.1
    // bucket_fn_stats(); // <10min

    // 4.1.2
    // size(); // many hours

    // 4.1.3
    // remap(); // 12min

    // 4.1.4
    // sharding(Sharding::Hybrid(1 << 37), "data/sharding_hybrid.json"); // 55min
    // sharding(Sharding::Memory, "data/sharding_memory.json"); // 1h

    // 4.2.1
    // query_batching(); // 40min

    // 4.2.2
    // query_throughput(); // 12min
}

const SMALL_N: usize = 20_000_000;
const LARGE_N: usize = 1_000_000_000;

const PARAMS_SIMPLE: PtrHashParams<Linear> = PtrHashParams {
    alpha: 0.99,
    lambda: 3.0,
    bucket_fn: Linear,
    // defaults...
    slots_per_part: None,
    keys_per_shard: 1 << 31,
    sharding: Sharding::None,
    remap: true,
    print_stats: false,
};

const PARAMS_COMPACT: PtrHashParams<CubicEps> = PtrHashParams {
    alpha: 0.99,
    lambda: 4.0,
    bucket_fn: CubicEps,
    // defaults...
    slots_per_part: None,
    keys_per_shard: 1 << 31,
    sharding: Sharding::None,
    remap: true,
    print_stats: false,
};

#[derive(Debug, Serialize, Default)]
struct Result {
    n: usize,
    alpha: f64,
    lambda: f64,
    bucketfn: String,
    slots_per_part: usize,
    real_alpha: f64,
    construction_1: f64,
    construction_6: f64,
    pilots: f64,
    remap: f64,
    remap_type: String,
    total: f64,
    q1_phf: f64,
    q1_mphf: f64,
    q1_phf_bb: f64,
    q1_mphf_bb: f64,
    q32_phf: f64,
    q32_mphf: f64,
}

#[derive(Debug, Serialize, Default, Clone)]
struct QueryResult {
    n: usize,
    alpha: f64,
    lambda: f64,
    bucketfn: String,
    construction_6: f64,
    pilots: f64,
    remap: f64,
    remap_type: String,
    total: f64,
    batch_size: usize,
    threads: usize,
    // loop/stream/batch
    mode: String,
    q_phf: f64,
    q_mphf: f64,
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
            alpha: 0.99,
            ..PtrHashParams::default_compact()
        };
        MyPtrHash::new_with_stats(&keys, params).1
    }

    {
        let lambda = 3.5;
        let mut stats = HashMap::new();
        stats.insert("linear", build(keys, lambda, Linear));
        stats.insert("skewed", build(keys, lambda, Skewed::default()));
        stats.insert("optimal", build(keys, lambda, Optimal { eps: 1. / 256. }));
        stats.insert("square", build(keys, lambda, Square));
        stats.insert("cubic", build(keys, lambda, CubicEps));

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
            ..PtrHashParams::default_compact()
        };
        eprintln!("Running {alpha} {lambda} {bucket_fn:?}");
        // Construct on 6 threads.
        let (ph, c6) = util::time(|| <MyPtrHash<_, R>>::try_new(&keys, params));
        let ph = ph.as_ref();

        // Space usage.
        let (pilots, remap) = ph.map(|ph| ph.bits_per_element())?;
        let total = pilots + remap;
        let r = Result {
            n: keys.len(),
            alpha,
            slots_per_part: ph.map(|ph| ph.slots_per_part()).unwrap_or_default(),
            real_alpha: ph
                .map(|ph| keys.len() as f64 / ph.max_index() as f64)
                .unwrap_or_default(),
            lambda,
            construction_6: c6,
            bucketfn: format!("{bucket_fn:?}"),
            pilots,
            remap,
            remap_type: R::name(),
            total,
            ..Result::default()
        };
        eprintln!("Result: {r:?}");
        Some(r)
    }

    let n = LARGE_N;
    let mut results = vec![];
    let keys = &generate_keys(n);
    for alpha in [0.98, 0.99, 0.995, 0.998] {
        for lambda in 27..45 {
            let lambda = lambda as f64 / 10.;
            let Some(r) = test::<Vec<u32>>(keys, alpha, lambda, Linear) else {
                break;
            };
            results.push(r);
            eprintln!();
        }
        for lambda in 27..45 {
            let lambda = lambda as f64 / 10.;
            let Some(r) = (if alpha >= 0.995 {
                test::<Vec<u32>>(keys, alpha, lambda, CubicEps)
            } else {
                test::<CachelineEfVec>(keys, alpha, lambda, CubicEps)
            }) else {
                break;
            };
            results.push(r);
            eprintln!();
        }
    }
    write(&results, &format!("data/size.json"));
}

/// Collect stats on size and query speed, varying alpha and lambda.
fn remap() {
    /// Return:
    /// Construction time (1 and 6 threads)
    /// Space (pilots, remap, total)
    /// Query throughput (32 streaming)
    fn test<R: MutPacked + Send>(
        keys: &Vec<u64>,
        alpha: f64,
        lambda: f64,
        bucket_fn: impl BucketFn + Send,
    ) -> Result {
        type MyPtrHash<BF, R> = PtrHash<u64, BF, R, FxHash, Vec<u8>>;

        let params = PtrHashParams {
            alpha,
            lambda,
            bucket_fn,
            ..PtrHashParams::default_compact()
        };

        // Construct on 6 threads.
        let (ph, c6) = util::time(|| <MyPtrHash<_, R>>::new(&keys, params));

        // Space usage.
        let (pilots, remap) = ph.bits_per_element();
        let total = pilots + remap;

        // Single threaded query throughput, non-minimal and minimal.
        // Query all keys 'loops' times.
        let loops = 1_000_000_000 / keys.len();
        let q1_phf = time_query_f(keys, || {
            let mut sum = 0;
            for key in keys {
                sum += ph.index(key);
            }
            sum
        });
        let q1_mphf = time_query_f(keys, || {
            let mut sum = 0;
            for key in keys {
                sum += ph.index_minimal(key);
            }
            sum
        });
        let q1_phf_bb = time_query_f(keys, || {
            let mut sum = 0;
            for key in keys {
                black_box(());
                sum += ph.index(key);
            }
            sum
        });
        let q1_mphf_bb = time_query_f(keys, || {
            let mut sum = 0;
            for key in keys {
                black_box(());
                sum += ph.index_minimal(key);
            }
            sum
        });
        let q32_phf = time_query(keys, || ph.index_stream::<32, false, _>(keys));
        let q32_mphf = time_query(keys, || ph.index_stream::<32, true, _>(keys));

        Result {
            n: keys.len(),
            alpha,
            lambda,
            slots_per_part: ph.slots_per_part(),
            real_alpha: keys.len() as f64 / ph.max_index() as f64,
            construction_1: 0.,
            construction_6: c6,
            bucketfn: format!("{bucket_fn:?}"),
            remap_type: R::name(),
            pilots,
            remap,
            total,
            q1_phf,
            q1_mphf,
            q1_phf_bb,
            q1_mphf_bb,
            q32_phf,
            q32_mphf,
        }
    }

    let n = LARGE_N;
    let mut results = vec![];
    let keys = &generate_keys(n);

    {
        let alpha = 0.99;
        let lambda = 3.0;
        results.push(test::<Vec<u32>>(keys, alpha, lambda, Linear));
        results.push(test::<CachelineEfVec>(keys, alpha, lambda, Linear));
        results.push(test::<EliasFano>(keys, alpha, lambda, Linear));
    }
    {
        let alpha = 0.99;
        let lambda = 4.0;
        results.push(test::<Vec<u32>>(keys, alpha, lambda, CubicEps));
        results.push(test::<CachelineEfVec>(keys, alpha, lambda, CubicEps));
        results.push(test::<EliasFano>(keys, alpha, lambda, CubicEps));
    }
    write(&results, &format!("data/remap.json"));
}

fn sharding(sharding: Sharding, path: &str) {
    let n = 100_000_000_000 / 2;
    let range = 0..n as u64;
    let keys = range.into_par_iter();
    let start = Instant::now();
    let bucket_fn = CubicEps;
    let ptr_hash = PtrHash::<_, _, CachelineEfVec, Murmur2_64>::new_from_par_iter(
        n,
        keys,
        PtrHashParams {
            lambda: 3.9,
            alpha: 0.99,
            // ~16GiB of keys per shard.
            keys_per_shard: 1 << 31,
            // Max 128GiB of memory at a time.
            sharding,
            bucket_fn,
            ..PtrHashParams::default_compact()
        },
    );
    let c6 = start.elapsed().as_secs_f64();
    let (pilots, remap) = ptr_hash.bits_per_element();
    let total = pilots + remap;
    let r = Result {
        n,
        lambda: 3.9,
        alpha: 0.99,
        construction_6: c6,
        bucketfn: format!("{:?}", bucket_fn),
        pilots,
        remap,
        total,
        ..Result::default()
    };

    eprintln!("Sharding {sharding:?}: {c6}s",);
    write(&r, path);
}

fn query_batching() {
    fn test(keys: &Vec<u64>, params: PtrHashParams<impl BucketFn>, rs: &mut Vec<QueryResult>) {
        type MyPtrHash<BF> = PtrHash<u64, BF, Vec<u32>, FxHash, Vec<u8>>;
        eprintln!("Building {params:?}");
        // Construct on 6 threads.
        let (ph, c6) = util::time(|| MyPtrHash::new(&keys, params));

        // Space usage.
        let (pilots, remap) = ph.bits_per_element();
        let total = pilots + remap;

        let r0 = QueryResult {
            n: keys.len(),
            alpha: params.alpha,
            lambda: params.lambda,
            construction_6: c6,
            bucketfn: format!("{:?}", params.bucket_fn),
            pilots,
            remap,
            total,
            remap_type: "none".to_string(),
            ..Default::default()
        };

        let q_phf = time_query_f(keys, || {
            let mut sum = 0;
            for key in keys {
                sum += ph.index(key);
            }
            sum
        });

        let r = QueryResult {
            batch_size: 0,
            mode: "loop".to_string(),
            q_phf,
            ..r0.clone()
        };
        eprintln!("Result: {r:?}");
        rs.push(r.clone());

        let q_phf = time_query_f(keys, || {
            let mut sum = 0;
            for key in keys {
                black_box(());
                sum += ph.index(key);
            }
            sum
        });

        let r = QueryResult {
            batch_size: 0,
            mode: "loop_bb".to_string(),
            q_phf,
            ..r0.clone()
        };
        eprintln!("Result: {r:?}");
        rs.push(r.clone());

        fn batch<const A: usize, BF: BucketFn>(
            ph: &PtrHash<u64, BF, Vec<u32>, FxHash, Vec<u8>>,
            keys: &Vec<u64>,
            r: &QueryResult,
            rs: &mut Vec<QueryResult>,
        ) {
            let stream = time_query(keys, || ph.index_stream::<A, false, _>(keys));
            // Somehow, index_batch has very weird scaling behaviour in A.
            // index_batch2 *does* improve as A increases, and so we use that one instead.
            // let batch = time_query(keys, || ph.index_batch_exact::<A, false>(keys));
            let batch2 = time_query(keys, || ph.index_batch_exact2::<A, false>(keys));
            rs.push(QueryResult {
                batch_size: A,
                mode: "stream".to_string(),
                q_phf: stream,
                ..r.clone()
            });
            eprintln!("Result: {:?}", rs.last().unwrap());
            // rs.push(QueryResult {
            //     batch_size: A,
            //     mode: "batch".to_string(),
            //     q_phf: batch,
            //     ..r.clone()
            // });
            // eprintln!("Result: {:?}", rs.last().unwrap());
            rs.push(QueryResult {
                batch_size: A,
                mode: "batch2".to_string(),
                q_phf: batch2,
                ..r.clone()
            });
            eprintln!("Result: {:?}", rs.last().unwrap());
        }
        batch::<1, _>(&ph, keys, &r, rs);
        batch::<2, _>(&ph, keys, &r, rs);
        batch::<3, _>(&ph, keys, &r, rs);
        batch::<4, _>(&ph, keys, &r, rs);
        batch::<5, _>(&ph, keys, &r, rs);
        batch::<6, _>(&ph, keys, &r, rs);
        batch::<7, _>(&ph, keys, &r, rs);
        batch::<8, _>(&ph, keys, &r, rs);
        batch::<10, _>(&ph, keys, &r, rs);
        batch::<12, _>(&ph, keys, &r, rs);
        batch::<14, _>(&ph, keys, &r, rs);
        batch::<16, _>(&ph, keys, &r, rs);
        batch::<20, _>(&ph, keys, &r, rs);
        batch::<24, _>(&ph, keys, &r, rs);
        batch::<28, _>(&ph, keys, &r, rs);
        batch::<32, _>(&ph, keys, &r, rs);
        batch::<40, _>(&ph, keys, &r, rs);
        batch::<48, _>(&ph, keys, &r, rs);
        batch::<56, _>(&ph, keys, &r, rs);
        batch::<64, _>(&ph, keys, &r, rs);
    }

    let mut results = vec![];
    for n in [SMALL_N, LARGE_N] {
        let keys = &generate_keys(n);

        test(keys, PARAMS_SIMPLE, &mut results);
        test(keys, PARAMS_COMPACT, &mut results);
    }
    write(&results, "data/query_batching.json");
}

fn time_query<I: Iterator<Item = usize>>(keys: &[u64], f: impl Fn() -> I) -> f64 {
    time_query_f(keys, || f().sum::<usize>())
}

fn time_query_f(keys: &[u64], f: impl Fn() -> usize) -> f64 {
    let loops = 1_000_000_000 / keys.len();
    time(|| black_box((0..loops).map(|_| f()).sum::<usize>())).1
}

fn time_query_parallel<'k, I: Iterator<Item = usize>>(
    threads: usize,
    keys: &'k Vec<u64>,
    f: impl Fn(&'k [u64]) -> I + Send + Sync,
) -> f64 {
    time_query_parallel_f(
        threads,
        keys,
        #[inline(always)]
        |keys| f(keys).sum::<usize>(),
    )
}

fn time_query_parallel_f<'k>(
    threads: usize,
    keys: &'k Vec<u64>,
    f: impl Fn(&'k [u64]) -> usize + Send + Sync,
) -> f64 {
    let loops = 1_000_000_000 / keys.len();
    let chunk_size = keys.len().div_ceil(threads);

    time(move || {
        rayon::scope(|scope| {
            for thread_idx in 0..threads {
                let f = &f;
                scope.spawn(move |_| {
                    let mut sum = 0;

                    for l in 0..loops {
                        let idx = (thread_idx + l) % threads;
                        let start_idx = idx * chunk_size;
                        let end = min((idx + 1) * chunk_size, keys.len());
                        sum += f(&keys[start_idx..end]);
                    }
                    black_box(sum);
                });
            }
        });
    })
    .1
}

fn query_throughput() {
    fn test<R: MutPacked>(
        keys: &Vec<u64>,
        params: PtrHashParams<impl BucketFn>,
        rs: &mut Vec<QueryResult>,
    ) {
        type MyPtrHash<BF, R> = PtrHash<u64, BF, R, FxHash, Vec<u8>>;
        eprintln!("Building {params:?}");
        // Construct on 6 threads.
        let (ph, c6) = util::time(|| MyPtrHash::<_, R>::new(&keys, params));

        // Space usage.
        let (pilots, remap) = ph.bits_per_element();
        let total = pilots + remap;

        let r0 = QueryResult {
            n: keys.len(),
            alpha: params.alpha,
            lambda: params.lambda,
            construction_6: c6,
            bucketfn: format!("{:?}", params.bucket_fn),
            pilots,
            remap,
            total,
            remap_type: R::name(),
            ..Default::default()
        };

        let loops = 1_000_000_000 / keys.len();

        // When n is small, queries perfectly scale to >1 threads anyway.
        let max_threads = 6;
        for threads in 1..=max_threads {
            let q_phf = time_query_parallel_f(threads, keys, |keys| {
                let mut sum = 0;
                for key in keys {
                    black_box(());
                    sum += ph.index(key);
                }
                sum
            });
            let q_mphf = time_query_parallel_f(threads, keys, |keys| {
                let mut sum = 0;
                for key in keys {
                    black_box(());
                    sum += ph.index_minimal(key);
                }
                sum
            });

            let r = QueryResult {
                batch_size: 0,
                mode: "loop_bb".to_string(),
                q_phf,
                q_mphf,
                threads,
                ..r0.clone()
            };
            eprintln!("Result: {r:?}");
            rs.push(r.clone());

            let q_phf = time_query_parallel_f(threads, keys, |keys| {
                let mut sum = 0;
                for key in keys {
                    sum += ph.index(key);
                }
                sum
            });
            let q_mphf = time_query_parallel_f(threads, keys, |keys| {
                let mut sum = 0;
                for key in keys {
                    sum += ph.index_minimal(key);
                }
                sum
            });

            let r = QueryResult {
                batch_size: 0,
                mode: "loop".to_string(),
                q_phf,
                q_mphf,
                threads,
                ..r0.clone()
            };
            eprintln!("Result: {r:?}");
            rs.push(r.clone());

            const A: usize = 32;
            let stream_phf =
                time_query_parallel(threads, keys, |keys| ph.index_stream::<A, false, _>(keys));
            let stream_mphf =
                time_query_parallel(threads, keys, |keys| ph.index_stream::<A, true, _>(keys));

            rs.push(QueryResult {
                batch_size: A,
                mode: "stream".to_string(),
                q_phf: stream_phf,
                q_mphf: stream_mphf,
                threads,
                ..r.clone()
            });
            eprintln!("Result: {:?}", rs.last().unwrap());
        }
    }

    let mut results = vec![];
    for n in [SMALL_N, LARGE_N] {
        let keys = &generate_keys(n);

        test::<Vec<u32>>(keys, PARAMS_SIMPLE, &mut results);
        test::<CachelineEfVec>(keys, PARAMS_COMPACT, &mut results);
    }
    write(&results, "data/query_throughput.json");
}
