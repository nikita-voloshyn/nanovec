pub mod brute;
pub mod kdtree;

pub use brute::{BruteForce, DeleteError, SearchResult};
pub use kdtree::{should_use_kdtree, KdTree};
