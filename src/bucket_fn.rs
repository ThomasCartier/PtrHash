pub trait BucketFn: Clone + Sync {
    fn call(&self, x: u64) -> u64;
}

#[derive(Clone)]
pub struct Linear;

impl BucketFn for Linear {
    fn call(&self, x: u64) -> u64 {
        x
    }
}
