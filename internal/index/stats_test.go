package index

import (
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestHNSW_GraphStats(t *testing.T) {
	mockStorage := make(map[int64][]float32)
	fetcher := func(offset int64, vec []float32) error {
		if storedVec, ok := mockStorage[offset]; ok {
			copy(vec, storedVec)
			return nil
		}
		return fmt.Errorf("vector not found at offset %d", offset)
	}

	hnsw := NewHNSW(2, 16, 200, 50, Euclidean, fetcher)

	// Insert 100 vectors
	for i := 0; i < 100; i++ {
		id := VectorID(fmt.Sprintf("vec_%d", i))
		vec := []float32{float32(i), float32(i)}
		offset := int64(i)
		mockStorage[offset] = vec
		err := hnsw.Insert(id, vec, offset)
		if err != nil {
			t.Fatalf("Failed to insert: %v", err)
		}
	}

	// Tombstone 30 vectors
	for i := 0; i < 30; i++ {
		id := VectorID(fmt.Sprintf("vec_%d", i))
		hnsw.Delete(id)
	}

	stats := hnsw.GraphStats()
	if stats.TotalNodes != 100 {
		t.Errorf("Expected 100 total nodes, got %d", stats.TotalNodes)
	}
	if stats.TombstonedNodes != 30 {
		t.Errorf("Expected 30 tombstoned nodes, got %d", stats.TombstonedNodes)
	}
	if stats.EntryPointLayer != hnsw.maxLayer {
		t.Errorf("Expected EntryPointLayer %d, got %d", hnsw.maxLayer, stats.EntryPointLayer)
	}
	if len(stats.AvgDegreePerLayer) == 0 {
		t.Errorf("Expected AvgDegreePerLayer to be populated")
	}
}

func TestHNSW_Rebuild(t *testing.T) {
	var mu sync.RWMutex
	mockStorage := make(map[int64][]float32)
	fetcher := func(offset int64, out []float32) error {
		mu.RLock()
		vec, ok := mockStorage[offset]
		mu.RUnlock()
		if ok {
			copy(out, vec)
			return nil
		}
		return fmt.Errorf("vector not found at offset %d", offset)
	}

	hnsw := NewHNSW(2, 16, 200, 50, Euclidean, fetcher)

	// Insert 500 vectors
	vectors := make(map[VectorID][]float32)
	for i := 0; i < 500; i++ {
		id := VectorID(fmt.Sprintf("vec_%d", i))
		vec := []float32{rand.Float32(), rand.Float32()}
		vectors[id] = vec
		offset := int64(i)
		mu.Lock()
		mockStorage[offset] = vec
		mu.Unlock()
		err := hnsw.Insert(id, vec, offset)
		if err != nil {
			t.Fatalf("Failed to insert: %v", err)
		}
	}

	// Delete roughly 200 vectors
	deletedIDs := make(map[VectorID]bool)
	for i := 0; i < 500; i += 2 {
		id := VectorID(fmt.Sprintf("vec_%d", i))
		hnsw.Delete(id)
		deletedIDs[id] = true
	}

	// Verify pre-rebuild stats
	stats := hnsw.GraphStats()
	if stats.TombstonedNodes != 250 {
		t.Fatalf("Expected 250 tombstoned, got %d", stats.TombstonedNodes)
	}

	// Test concurrent access during rebuild
	var wg sync.WaitGroup
	var concurrentOps int32
	stopCh := make(chan struct{})

	// Start a goroutine doing concurrent searches and inserts
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stopCh:
				return
			default:
				// Search
				qVec := []float32{0.5, 0.5}
				_, err := hnsw.Search(qVec, 50, 10, nil)
				if err != nil {
					t.Errorf("Concurrent search failed: %v", err)
				}

				opsCount := atomic.LoadInt32(&concurrentOps)

				// Insert and then immediately delete the same ID to test chronological replay
				flipID := VectorID(fmt.Sprintf("flip_%d", opsCount))

				mu.Lock()
				offsetFlip := int64(1000 + opsCount*2)
				mockStorage[offsetFlip] = qVec
				mu.Unlock()

				hnsw.Insert(flipID, qVec, offsetFlip)
				hnsw.Delete(flipID)

				// Insert a new vector that MUST survive the rebuild
				surviveID := VectorID(fmt.Sprintf("survive_%d", opsCount))

				mu.Lock()
				offsetSurvive := int64(1000 + opsCount*2 + 1)
				mockStorage[offsetSurvive] = qVec
				mu.Unlock()

				err = hnsw.Insert(surviveID, qVec, offsetSurvive)
				if err != nil {
					t.Errorf("Concurrent insert failed: %v", err)
				}

				atomic.AddInt32(&concurrentOps, 1)

				// Small sleep to spread operations across the rebuild window
				time.Sleep(500 * time.Microsecond)
			}
		}
	}()

	// Perform rebuild
	err := hnsw.Rebuild()
	if err != nil {
		t.Fatalf("Rebuild failed: %v", err)
	}

	// Stop concurrent operations
	close(stopCh)
	wg.Wait()

	if atomic.LoadInt32(&concurrentOps) == 0 {
		t.Errorf("Expected concurrent operations to run during rebuild")
	}

	// Verify post-rebuild stats
	statsAfter := hnsw.GraphStats()
	maxExpectedTombstones := int(atomic.LoadInt32(&concurrentOps))
	if statsAfter.TombstonedNodes > maxExpectedTombstones {
		t.Errorf("Expected at most %d tombstoned nodes after rebuild, got %d", maxExpectedTombstones, statsAfter.TombstonedNodes)
	}

	// Total nodes should be roughly (500 - 250) + (inserts during rebuild)
	expectedBase := 250
	if statsAfter.TotalNodes < expectedBase {
		t.Errorf("Expected at least %d nodes after rebuild, got %d", expectedBase, statsAfter.TotalNodes)
	}

	// Verify deleted nodes are actually gone
	hnsw.mu.RLock()
	for id := range deletedIDs {
		if _, exists := hnsw.nodes[id]; exists {
			t.Errorf("Deleted node %s still exists in graph after rebuild", id)
		}
	}
	hnsw.mu.RUnlock()

	// Verify concurrent operations were accurately preserved
	totalConcurrentOps := atomic.LoadInt32(&concurrentOps)
	for i := int32(0); i < totalConcurrentOps; i++ {
		surviveID := VectorID(fmt.Sprintf("survive_%d", i))
		if !hnsw.Contains(surviveID) {
			t.Errorf("Concurrent insert %s was lost during rebuild", surviveID)
		}

		flipID := VectorID(fmt.Sprintf("flip_%d", i))
		if hnsw.Contains(flipID) {
			t.Errorf("Concurrent flip (insert-then-delete) %s survived when it should be deleted", flipID)
		}
	}
}
