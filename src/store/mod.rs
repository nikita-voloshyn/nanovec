pub mod record;

/// Error types for VectorStore operations.
#[derive(Debug)]
pub enum StoreError {
    /// The inserted vector has a different dimension than the store expects.
    DimensionMismatch { expected: usize, got: usize },
    /// The given vector index is out of bounds.
    InvalidOffset(usize),
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StoreError::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            StoreError::InvalidOffset(offset) => {
                write!(f, "invalid vector offset: {offset}")
            }
        }
    }
}

impl std::error::Error for StoreError {}

/// Flat SoA vector storage. All vectors are packed contiguously:
/// `[v0_d0, v0_d1, ..., v0_dN, v1_d0, v1_d1, ..., v1_dN, ...]`
pub struct VectorStore {
    vectors: Vec<f32>,
    dimension: usize,
    count: usize,
}

impl VectorStore {
    /// Create a new store for vectors of the given dimension.
    pub fn new(dimension: usize) -> Self {
        Self {
            vectors: Vec::new(),
            dimension,
            count: 0,
        }
    }

    /// Insert a vector. Returns the logical index (0-based vector index).
    /// Returns `Err` if `vector.len() != dimension`.
    pub fn insert(&mut self, vector: &[f32]) -> Result<usize, StoreError> {
        if vector.len() != self.dimension {
            return Err(StoreError::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }
        let index = self.count;
        self.vectors.extend_from_slice(vector);
        self.count += 1;
        Ok(index)
    }

    /// Get the vector slice at the given logical index.
    /// Returns `None` if the index is out of bounds.
    pub fn get(&self, index: usize) -> Option<&[f32]> {
        if index >= self.count {
            return None;
        }
        let start = index * self.dimension;
        Some(&self.vectors[start..start + self.dimension])
    }

    /// Swap-remove: replaces the vector at `index` with the last vector,
    /// then truncates. Returns the index where the moved vector now lives.
    /// If the removed vector was the last one, returns `index` (no swap needed).
    /// Returns `Err` if the index is out of bounds.
    pub fn swap_remove(&mut self, index: usize) -> Result<usize, StoreError> {
        if index >= self.count {
            return Err(StoreError::InvalidOffset(index));
        }
        let last_index = self.count - 1;
        if index != last_index {
            let dst_start = index * self.dimension;
            let src_start = last_index * self.dimension;
            // Copy last vector into the removed slot.
            self.vectors
                .copy_within(src_start..src_start + self.dimension, dst_start);
        }
        self.vectors.truncate(last_index * self.dimension);
        self.count -= 1;
        Ok(index)
    }

    /// Returns the number of stored vectors.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Returns the dimension of each vector.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Yields `(index, slice)` for each stored vector.
    pub fn iter(&self) -> impl Iterator<Item = (usize, &[f32])> {
        let dim = self.dimension;
        self.vectors.chunks_exact(dim).enumerate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_returns_correct_offset() {
        let mut store = VectorStore::new(3);
        let off0 = store.insert(&[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(off0, 0);
        let off1 = store.insert(&[4.0, 5.0, 6.0]).unwrap();
        assert_eq!(off1, 1);
    }

    #[test]
    fn get_after_insert_returns_same_vector() {
        let mut store = VectorStore::new(3);
        store.insert(&[1.0, 2.0, 3.0]).unwrap();
        let v = store.get(0).unwrap();
        assert_eq!(v, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn insert_wrong_dimension_errors() {
        let mut store = VectorStore::new(3);
        let result = store.insert(&[1.0, 2.0]);
        assert!(result.is_err());
        match result.unwrap_err() {
            StoreError::DimensionMismatch { expected, got } => {
                assert_eq!(expected, 3);
                assert_eq!(got, 2);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn count_grows_after_insert() {
        let mut store = VectorStore::new(2);
        assert_eq!(store.count(), 0);
        store.insert(&[1.0, 2.0]).unwrap();
        assert_eq!(store.count(), 1);
        store.insert(&[3.0, 4.0]).unwrap();
        assert_eq!(store.count(), 2);
    }

    #[test]
    fn swap_remove_decreases_count() {
        let mut store = VectorStore::new(2);
        store.insert(&[1.0, 2.0]).unwrap();
        store.insert(&[3.0, 4.0]).unwrap();
        store.insert(&[5.0, 6.0]).unwrap();
        assert_eq!(store.count(), 3);

        // Remove index 0; last vector (index 2) should move to index 0.
        let moved = store.swap_remove(0).unwrap();
        assert_eq!(moved, 0);
        assert_eq!(store.count(), 2);

        // The vector at index 0 should now be [5.0, 6.0].
        assert_eq!(store.get(0).unwrap(), &[5.0, 6.0]);
        // The vector at index 1 should still be [3.0, 4.0].
        assert_eq!(store.get(1).unwrap(), &[3.0, 4.0]);
    }

    #[test]
    fn swap_remove_last_element_no_swap() {
        let mut store = VectorStore::new(2);
        store.insert(&[1.0, 2.0]).unwrap();
        store.insert(&[3.0, 4.0]).unwrap();

        // Remove the last element (index 1); no swap needed.
        let moved = store.swap_remove(1).unwrap();
        assert_eq!(moved, 1);
        assert_eq!(store.count(), 1);
        assert_eq!(store.get(0).unwrap(), &[1.0, 2.0]);
    }

    #[test]
    fn swap_remove_invalid_offset_errors() {
        let mut store = VectorStore::new(2);
        store.insert(&[1.0, 2.0]).unwrap();
        let result = store.swap_remove(5);
        assert!(result.is_err());
        match result.unwrap_err() {
            StoreError::InvalidOffset(off) => assert_eq!(off, 5),
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn iter_yields_all_vectors() {
        let mut store = VectorStore::new(2);
        store.insert(&[1.0, 2.0]).unwrap();
        store.insert(&[3.0, 4.0]).unwrap();
        store.insert(&[5.0, 6.0]).unwrap();

        let collected: Vec<(usize, &[f32])> = store.iter().collect();
        assert_eq!(collected.len(), 3);
        assert_eq!(collected[0], (0, &[1.0, 2.0][..]));
        assert_eq!(collected[1], (1, &[3.0, 4.0][..]));
        assert_eq!(collected[2], (2, &[5.0, 6.0][..]));
    }
}
