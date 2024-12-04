#![allow(unused)]
use bucket_fn::{BucketFn, Linear};
use cacheline_ef::CachelineEfVec;
use clap::{Parser, Subcommand};
#[cfg(feature = "epserde")]
use epserde::prelude::*;
use itertools::Itertools;
use ptr_hash::{
    hash::{Hasher, Murmur2_64},
    pack::Packed,
    *,
};
use std::{
    cmp::min,
    hint::black_box,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
    time::SystemTime,
};

#[derive(clap::Parser)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

const DEFAULT_LAMBDA: f64 = 3.5;
const DEFAULT_ALPHA: f64 = 0.98;
const DEFAULT_SLOTS_PER_PART: usize = 1 << 18;
const DEFAULT_KEYS_PER_SHARD: usize = 1 << 33;
const DEFAULT_SHARDING: Sharding = Sharding::None;

#[derive(Subcommand)]
enum Command {
    /// Construct PtrHash.
    Build {
        #[arg(short)]
        n: usize,
        #[arg(short, default_value_t = DEFAULT_LAMBDA)]
        lambda: f64,
        #[arg(short, default_value_t = DEFAULT_ALPHA)]
        alpha: f64,
        #[arg(short, default_value_t = DEFAULT_SLOTS_PER_PART)]
        s: usize,
        #[arg(short, default_value_t = DEFAULT_KEYS_PER_SHARD)]
        keys_per_shard: usize,
        #[arg(long, value_enum, default_value_t = DEFAULT_SHARDING)]
        sharding: Sharding,
        #[arg(long)]
        stats: bool,
        #[arg(short, long, default_value_t = 0)]
        threads: usize,
    },

    /// Measure query time on randomly-constructed PtrHash.
    Query {
        #[arg(short)]
        n: usize,
        /// Path to file containing one key per line.
        #[arg(long)]
        keys: Option<PathBuf>,
        #[arg(short, default_value_t = DEFAULT_LAMBDA)]
        lambda: f64,
        #[arg(short, default_value_t = DEFAULT_ALPHA)]
        alpha: f64,
        #[arg(short, default_value_t = DEFAULT_SLOTS_PER_PART)]
        s: usize,
        #[arg(short, default_value_t = DEFAULT_KEYS_PER_SHARD)]
        keys_per_shard: usize,
        #[arg(long, value_enum, default_value_t = DEFAULT_SHARDING)]
        sharding: Sharding,
        #[arg(long, default_value_t = 300000000)]
        total: usize,
        #[arg(long)]
        stats: bool,
        #[arg(short, long, default_value_t = 0)]
        threads: usize,
    },

    BucketFn {
        #[arg(short)]
        n: usize,
        #[arg(short, default_value_t = DEFAULT_LAMBDA)]
        lambda: f64,
        #[arg(short, default_value_t = DEFAULT_ALPHA)]
        alpha: f64,
        #[arg(short, default_value_t = DEFAULT_SLOTS_PER_PART)]
        s: usize,
        #[arg(short, default_value_t = DEFAULT_KEYS_PER_SHARD)]
        keys_per_shard: usize,
        #[arg(long, value_enum, default_value_t = DEFAULT_SHARDING)]
        sharding: Sharding,
        #[arg(long, default_value_t = 300000000)]
        total: usize,
        #[arg(long)]
        stats: bool,
        #[arg(short, long, default_value_t = 0)]
        threads: usize,
    },
}

type PH<Key, BF> = PtrHash<Key, BF, CachelineEfVec, hash::FxHash, Vec<u8>>;

fn main() -> anyhow::Result<()> {
    let Args { command } = Args::parse();

    match command {
        Command::Build {
            n,
            lambda,
            alpha,
            stats,
            s,
            keys_per_shard,
            threads,
            sharding,
        } => {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global()
                .unwrap();
            let keys = ptr_hash::util::generate_keys(n);
            let pt = PH::<_, Linear>::new(
                &keys,
                PtrHashParams {
                    lambda,
                    alpha,
                    print_stats: stats,
                    slots_per_part: s,
                    keys_per_shard,
                    sharding,
                    ..Default::default()
                },
            );

            #[cfg(feature = "epserde")]
            {
                Serialize::store(&pt, "pt.bin")?;
            }
        }
        Command::Query {
            mut n,
            lambda,
            alpha,
            total,
            stats,
            s,
            keys_per_shard,
            sharding,
            threads,
            keys,
        } => {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global()
                .unwrap();

            let params = PtrHashParams {
                lambda,
                alpha,
                print_stats: stats,
                slots_per_part: s,
                keys_per_shard,
                sharding,
                ..Default::default()
            };

            if let Some(keys_file) = keys {
                // read file keys_file
                let file = std::fs::read_to_string(keys_file).unwrap();
                let keys = file.lines().map(|l| l.as_bytes()).collect_vec();
                // let pt = DefaultPtrHash::<Murmur2_64, _>::new(&keys, params);
                // benchmark_queries(total, &keys, &pt);
                bench_hashers(total, &params, &keys);
            } else {
                let keys = ptr_hash::util::generate_keys(n);
                #[cfg(feature = "epserde")]
                let pt = <PtrHash>::mmap("pt.bin", Flags::default())?;
                #[cfg(not(feature = "epserde"))]
                let pt = <PtrHash>::new_random(n, params);
                // benchmark_queries::<1, _, _, _, _>(total, &keys, &pt);
                // benchmark_queries::<2, _, _, _, _>(total, &keys, &pt);
                // benchmark_queries::<3, _, _, _, _>(total, &keys, &pt);
                // benchmark_queries::<4, _, _, _, _>(total, &keys, &pt);
                // benchmark_queries::<8, _, _, _, _>(total, &keys, &pt);
                // benchmark_queries::<10, _, _, _, _>(total, &keys, &pt);
                // benchmark_queries::<12, _, _, _, _>(total, &keys, &pt);
                // benchmark_queries::<14, _, _, _, _>(total, &keys, &pt);
                // benchmark_queries::<16, _, _, _, _>(total, &keys, &pt);
                // benchmark_queries::<18, _, _, _, _>(total, &keys, &pt);
                // benchmark_queries::<20, _, _, _, _>(total, &keys, &pt);
                // benchmark_queries::<24, _, _, _, _>(total, &keys, &pt);
                benchmark_queries::<32, _, _, _, _, _>(total, &keys, &pt);
                // benchmark_queries::<48, _, _, _, _>(total, &keys, &pt);
                // benchmark_queries::<64, _, _, _, _>(total, &keys, &pt);
                // bench_hashers(total, &params, &keys);
            }
        }
        params @ Command::BucketFn { threads, n, .. } => {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global()
                .unwrap();

            let keys = ptr_hash::util::generate_keys(n);

            fn test(params: &Command, keys: &Vec<u64>, bucket_fn: impl BucketFn) {
                eprintln!("\nBenchmarking {bucket_fn:?}");
                let Command::BucketFn {
                    n,
                    lambda,
                    alpha,
                    total,
                    stats,
                    s,
                    keys_per_shard,
                    sharding,
                    threads,
                } = *params
                else {
                    unreachable!()
                };

                let params = PtrHashParams {
                    lambda,
                    alpha,
                    print_stats: stats,
                    slots_per_part: s,
                    keys_per_shard,
                    sharding,
                    bucket_fn,
                    remap: true,
                };

                let pt = PtrHash::<_, _>::new(keys, params);
                benchmark_queries::<32, _, _, _, _, _>(total, &keys, &pt);
            }

            test(&params, &keys, bucket_fn::CubicEps);
            // test(&params, &keys, bucket_fn::Cubic);
            test(&params, &keys, bucket_fn::SquareEps);
            test(&params, &keys, bucket_fn::Square);
            // test(&params, &keys, bucket_fn::Skewed::new(0.6, 0.3));
            // test(&params, &keys, bucket_fn::Linear);
            // test(&params, &keys, bucket_fn::Perfect { eps: 0. });
            // test(&params, &keys, bucket_fn::Perfect { eps: 0.01 });
        }
    }

    Ok(())
}

fn bench_hashers<Key: KeyT, BF: BucketFn>(total: usize, params: &PtrHashParams<BF>, keys: &[Key]) {
    let n = keys.len();
    let loops = total.div_ceil(n);
    fn test<H: Hasher<Key>, Key: KeyT, BF: BucketFn>(
        loops: usize,
        keys: &[Key],
        params: &PtrHashParams<BF>,
    ) {
        type PH<Key, H, BF> = PtrHash<Key, BF, CachelineEfVec, H, Vec<u8>>;
        let pt = PH::<Key, H, BF>::new_random(keys.len(), *params);

        let query = bench_index(loops, keys, |key| pt.index(key));
        eprintln!("  sequential: {query:>4.1}");
        let query = time(loops, keys, || {
            pt.index_stream::<64, false>(keys).sum::<usize>()
        });
        eprintln!(" prefetch 64: {query:>5.2}ns");
    }

    eprintln!("fxhash");
    test::<hash::FxHash, _, _>(loops, keys, params);
    eprintln!("murmur2");
    test::<hash::Murmur2_64, _, _>(loops, keys, params);
    eprintln!("murmur3");
    test::<hash::FastMurmur3_128, _, _>(loops, keys, params);

    eprintln!("highway64");
    test::<hash::Highway64, _, _>(loops, keys, params);
    eprintln!("highway128");
    test::<hash::Highway128, _, _>(loops, keys, params);
    eprintln!("city64");
    test::<hash::City64, _, _>(loops, keys, params);
    eprintln!("city128");
    test::<hash::City128, _, _>(loops, keys, params);
    eprintln!("wy64");
    test::<hash::Wy64, _, _>(loops, keys, params);
    eprintln!("xx64");
    test::<hash::Xx64, _, _>(loops, keys, params);
    eprintln!("xx128");
    test::<hash::Xx128, _, _>(loops, keys, params);
    eprintln!("metro64");
    test::<hash::Metro64, _, _>(loops, keys, params);
    eprintln!("metro128");
    test::<hash::Metro128, _, _>(loops, keys, params);
    eprintln!("spooky64");
    test::<hash::Spooky64, _, _>(loops, keys, params);
    eprintln!("spooky128");
    test::<hash::Spooky128, _, _>(loops, keys, params);
}

fn benchmark_queries<
    const A: usize,
    Key: KeyT,
    H: Hasher<Key>,
    BF: BucketFn,
    T: Packed,
    V: AsRef<[u8]> + Sync,
>(
    total: usize,
    keys: &[Key],
    pt: &PtrHash<Key, BF, T, H, V>,
) {
    let n = keys.len();
    let loops = total.div_ceil(n);

    eprintln!("BENCHMARKING A={A}\t loops {loops}");

    if A == 1 {
        // let query = bench_index(loops, keys, |key| pt.index_minimal(key));
        // eprintln!(" ( 1  /r/ ): {query:>5.2}");
        let query = bench_index(loops, keys, |key| pt.index(key));
        eprintln!(" ( 1  / / )  : {query:>5.2}ns");
    }

    for threads in [1, 3, 6] {
        let query = time(loops, keys, || {
            index_parallel::<A, _, _, _, _, _>(pt, keys, threads, true, false)
        });
        eprintln!(" ({A}t{threads}/r/s)  : {query:>5.2}ns");
        // let query = time(loops, keys, || {
        //     index_parallel::<A, _, _, _, _>(pt, keys, threads, true, true)
        // });
        // eprintln!(" ({A}t{threads}/r/b)  : {query:>5.2}ns");
        let query = time(loops, keys, || {
            index_parallel::<A, _, _, _, _, _>(pt, keys, threads, false, false)
        });
        eprintln!(" ({A:2}t{threads}/ /s)  : {query:>5.2}ns");
        // let query = time(loops, keys, || {
        //     index_parallel::<A, _, _, _, _>(pt, keys, threads, false, true)
        // });
        // eprintln!(" ({A:2}t{threads}/ /b)  : {query:>5.2}ns");
    }
}

#[must_use]
pub fn bench_index<Key: KeyT>(loops: usize, keys: &[Key], index: impl Fn(&Key) -> usize) -> f32 {
    let start = SystemTime::now();
    let mut sum = 0;
    for _ in 0..loops {
        for key in keys {
            sum += index(key);
        }
    }
    black_box(sum);
    start.elapsed().unwrap().as_nanos() as f32 / (loops * keys.len()) as f32
}

#[must_use]
pub fn time<Key: KeyT, F>(loops: usize, keys: &[Key], f: F) -> f32
where
    F: Fn() -> usize,
{
    let start = SystemTime::now();
    let mut sum = 0;
    for _ in 0..loops {
        sum += f();
    }
    black_box(sum);
    start.elapsed().unwrap().as_nanos() as f32 / (loops * keys.len()) as f32
}

/// Wrapper around `index_stream` that runs multiple threads.
fn index_parallel<
    const A: usize,
    Key: KeyT,
    BF: BucketFn,
    T: Packed,
    H: Hasher<Key>,
    V: AsRef<[u8]> + Sync,
>(
    pt: &PtrHash<Key, BF, T, H, V>,
    xs: &[Key],
    threads: usize,
    minimal: bool,
    batch: bool,
) -> usize {
    let chunk_size = xs.len().div_ceil(threads);
    let sum = AtomicUsize::new(0);
    rayon::scope(|scope| {
        let pt = &pt;
        for thread_idx in 0..threads {
            let sum = &sum;
            scope.spawn(move |_| {
                let start_idx = thread_idx * chunk_size;
                let end = min((thread_idx + 1) * chunk_size, xs.len());

                let thread_sum = if batch {
                    if minimal {
                        pt.index_batch_exact::<A, true>(&xs[start_idx..end])
                            .sum::<usize>()
                    } else {
                        pt.index_batch_exact::<A, false>(&xs[start_idx..end])
                            .sum::<usize>()
                    }
                } else {
                    if minimal {
                        pt.index_stream::<A, true>(&xs[start_idx..end])
                            .sum::<usize>()
                    } else {
                        pt.index_stream::<A, false>(&xs[start_idx..end])
                            .sum::<usize>()
                    }
                };

                sum.fetch_add(thread_sum, Ordering::Relaxed);
            });
        }
    });
    sum.load(Ordering::Relaxed)
}
