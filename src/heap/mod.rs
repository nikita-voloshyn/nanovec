use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A result item with score and ID, ordered by score descending (max-heap).
#[derive(PartialEq)]
struct OrderedItem {
    score: f32,
    id: u64,
}

impl Eq for OrderedItem {}

impl PartialOrd for OrderedItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
            .then(self.id.cmp(&other.id))
    }
}

/// Min-K heap: keeps the K items with the smallest scores.
///
/// Internally uses a max-heap so the largest score is always on top and can
/// be evicted in O(log K) when a better (smaller) score arrives.
pub struct BoundedMaxHeap {
    capacity: usize,
    heap: BinaryHeap<OrderedItem>,
}

impl BoundedMaxHeap {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            heap: BinaryHeap::with_capacity(capacity),
        }
    }

    /// Push a (score, id) pair. Lower score = better (closer match).
    /// If the heap is full and `score` is worse (>=) than the current max, the
    /// item is discarded.
    pub fn push(&mut self, score: f32, id: u64) {
        if self.capacity == 0 {
            return;
        }
        if self.heap.len() < self.capacity {
            self.heap.push(OrderedItem { score, id });
        } else if let Some(top) = self.heap.peek() {
            if score < top.score {
                self.heap.pop();
                self.heap.push(OrderedItem { score, id });
            }
        }
    }

    /// Drain the heap and return items sorted ascending by score (best first).
    pub fn into_sorted_vec(self) -> Vec<(f32, u64)> {
        let mut items: Vec<_> = self.heap.into_iter().collect();
        items.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(Ordering::Equal)
                .then(a.id.cmp(&b.id))
        });
        items
            .into_iter()
            .map(|item| (item.score, item.id))
            .collect()
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Returns the current worst (largest) score in the heap, or `None` if
    /// the heap is empty. O(1).
    ///
    /// Used by KD-Tree search to derive its geometric pruning radius — when
    /// the heap is full of K accepted candidates, the largest score in the
    /// heap is the radius beyond which any new candidate could not improve
    /// the result set.
    pub fn peek_worst(&self) -> Option<f32> {
        self.heap.peek().map(|item| item.score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_more_than_capacity_keeps_k() {
        let mut heap = BoundedMaxHeap::new(3);
        for i in 0..5 {
            heap.push(i as f32, i as u64);
        }
        assert_eq!(heap.len(), 3);
    }

    #[test]
    fn into_sorted_vec_returns_ascending_order() {
        let mut heap = BoundedMaxHeap::new(5);
        heap.push(3.0, 3);
        heap.push(1.0, 1);
        heap.push(4.0, 4);
        heap.push(1.5, 5);
        heap.push(2.0, 2);
        let sorted = heap.into_sorted_vec();
        let scores: Vec<f32> = sorted.iter().map(|(s, _)| *s).collect();
        assert_eq!(scores, vec![1.0, 1.5, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn keeps_top_3_smallest_scores() {
        let mut heap = BoundedMaxHeap::new(3);
        heap.push(10.0, 10);
        heap.push(5.0, 5);
        heap.push(1.0, 1);
        heap.push(3.0, 3);
        heap.push(0.5, 0);
        let sorted = heap.into_sorted_vec();
        let ids: Vec<u64> = sorted.iter().map(|(_, id)| *id).collect();
        assert_eq!(ids, vec![0, 1, 3]);
    }

    #[test]
    fn push_same_score_keeps_k_items() {
        let mut heap = BoundedMaxHeap::new(2);
        heap.push(1.0, 1);
        heap.push(1.0, 2);
        heap.push(1.0, 3);
        assert_eq!(heap.len(), 2);
    }

    #[test]
    fn peek_worst_returns_largest_score_when_full() {
        let mut heap = BoundedMaxHeap::new(3);
        heap.push(1.0, 1);
        heap.push(0.5, 2);
        heap.push(2.0, 3);
        // Worst (largest, since we keep K smallest) of {0.5, 1.0, 2.0} is 2.0.
        assert_eq!(heap.peek_worst(), Some(2.0));
        // Pushing a better (smaller) score evicts 2.0, so worst becomes 1.0.
        heap.push(0.1, 4);
        assert_eq!(heap.peek_worst(), Some(1.0));
    }

    #[test]
    fn peek_worst_empty_heap_returns_none() {
        let heap = BoundedMaxHeap::new(3);
        assert_eq!(heap.peek_worst(), None);
    }

    #[test]
    fn capacity_zero_returns_empty() {
        let mut heap = BoundedMaxHeap::new(0);
        heap.push(1.0, 1);
        assert!(heap.is_empty());
        let sorted = heap.into_sorted_vec();
        assert!(sorted.is_empty());
    }
}
