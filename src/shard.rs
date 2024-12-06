use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    sync::Mutex,
    thread,
    time::Duration,
};

use super::*;

/// Select the sharding method to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, Default)]
pub enum Sharding {
    #[default]
    None,
    Memory,
    Disk,
}

impl<Key: KeyT, BF: BucketFn, F: Packed, Hx: Hasher<Key>> PtrHash<Key, BF, F, Hx> {
    /// Return an iterator over shards.
    /// For each shard, a filtered copy of the ParallelIterator is returned.
    pub(crate) fn shards<'a>(
        &'a self,
        keys: impl ParallelIterator<Item = impl Borrow<Key>> + Clone + 'a,
    ) -> Box<dyn Iterator<Item = Vec<Hx::H>> + 'a> {
        match self.params.sharding {
            Sharding::None => self.no_sharding(keys.clone()),
            Sharding::Memory => self.shard_keys_to_disk(keys.clone()),
            Sharding::Disk => self.shard_keys_in_memory(keys.clone()),
        }
    }

    /// Return an iterator over shards.
    /// For each shard, a filtered copy of the ParallelIterator is returned.
    fn no_sharding<'a>(
        &'a self,
        keys: impl ParallelIterator<Item = impl Borrow<Key>> + Clone + 'a,
    ) -> Box<dyn Iterator<Item = Vec<Hx::H>> + 'a> {
        if self.params.print_stats {
            eprintln!("No sharding: collecting all hashes in memory.");
        }
        let start = std::time::Instant::now();
        let hashes = keys.map(|key| self.hash_key(key.borrow())).collect();
        log_duration("collect hash", start);
        Box::new(std::iter::once(hashes))
    }

    /// Loop over the keys once per shard.
    /// Return an iterator over shards.
    /// For each shard, a filtered copy of the ParallelIterator is returned.
    fn shard_keys_in_memory<'a>(
        &'a self,
        keys: impl ParallelIterator<Item = impl Borrow<Key>> + Clone + 'a,
    ) -> Box<dyn Iterator<Item = Vec<Hx::H>> + 'a> {
        eprintln!("In-memory sharding: iterate keys once per shard.");
        let it = (0..self.shards).map(move |shard| {
            eprintln!("Shard {shard:>3}/{:3}", self.shards);
            let start = std::time::Instant::now();
            let hashes = keys
                .clone()
                .map(|key| self.hash_key(key.borrow()))
                .filter(move |h| self.shard(*h) == shard)
                .collect();

            log_duration("collect shrd", start);
            hashes
        });
        Box::new(it)
    }

    /// Loop over the keys and write each keys hash to the corresponding shard.
    /// Returns an iterator over shards.
    /// Files are written to /tmp by default, but this can be changed using the
    /// TMPDIR environment variable.
    ///
    /// This is based on `SigStore` in `sux-rs`, but simplified for the specific use case here.
    /// https://github.com/vigna/sux-rs/blob/main/src/utils/sig_store.rs
    fn shard_keys_to_disk<'a>(
        &'a self,
        keys: impl ParallelIterator<Item = impl Borrow<Key>> + Clone + 'a,
    ) -> Box<dyn Iterator<Item = Vec<Hx::H>> + 'a> {
        eprintln!("Disk sharding: writing hashes per shard to disk.");
        let temp_dir = tempfile::TempDir::new().unwrap();
        eprintln!("TMP PATH: {:?}", temp_dir.path());

        // Create a file writer and count for each shard.
        let writers = (0..self.shards)
            .map(|shard| {
                Mutex::new((
                    BufWriter::new(
                        File::options()
                            .read(true)
                            .write(true)
                            .create(true)
                            .open(temp_dir.path().join(format!("{}.tmp", shard)))
                            .unwrap(),
                    ),
                    0,
                ))
            })
            .collect_vec();

        // Each thread has a local buffer per shard.
        let init = || writers.iter().map(ThreadLocalBuf::new).collect_vec();
        // Iterate over keys.
        keys.for_each_init(init, |bufs, key| {
            let h = self.hash_key(key.borrow());
            let shard = self.shard(h);
            bufs[shard].push(h);
        });

        eprintln!("Wrote all files. Pausing.");
        for w in &writers {
            let c = w.lock().unwrap().1;
            eprintln!("Count: {c}");
        }
        thread::sleep(Duration::from_secs(10));

        // Convert writers to files.
        let files = writers
            .into_iter()
            .map(|w| {
                let (mut w, cnt) = w.into_inner().unwrap();
                w.flush().unwrap();
                let mut file = w.into_inner().unwrap();
                file.seek(SeekFrom::Start(0)).unwrap();
                (file, cnt)
            })
            .collect_vec();

        let it = files.into_iter().map(move |(f, cnt)| {
            let mut v = vec![Hx::H::default(); cnt];
            let mut f = BufReader::new(f);
            let (pre, data, post) = unsafe { v.align_to_mut::<u8>() };
            assert!(pre.is_empty());
            assert!(post.is_empty());
            f.read_exact(data).unwrap();
            v
        });
        Box::new(it)
    }
}

struct ThreadLocalBuf<'a, H> {
    buf: Vec<H>,
    file: &'a Mutex<(BufWriter<File>, usize)>,
}

impl<'a, H> ThreadLocalBuf<'a, H> {
    fn new(file: &'a Mutex<(BufWriter<File>, usize)>) -> Self {
        Self {
            buf: Vec::with_capacity(1 << 16),
            file,
        }
    }
    fn push(&mut self, h: H) {
        self.buf.push(h);
        // L2 size
        if self.buf.len() == (1 << 16) {
            self.flush();
        }
    }
    fn flush(&mut self) {
        let mut file = self.file.lock().unwrap();
        let (pre, bytes, post) = unsafe { self.buf.align_to::<u8>() };
        assert!(pre.is_empty());
        assert!(post.is_empty());
        file.0.write_all(bytes).unwrap();
        file.1 += self.buf.len();
        self.buf.clear();
    }
}

impl<'a, H> Drop for ThreadLocalBuf<'a, H> {
    fn drop(&mut self) {
        self.flush();
    }
}
