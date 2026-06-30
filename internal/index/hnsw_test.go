package index

import (
	"fmt"
	"math/rand"
	"testing"
)

func TestHNSW_InsertSearch(t *testing.T) {
	mockStorage := make(map[int64][]float32)
	fetcher := func(offset int64, vec []float32) error {
		if storedVec, ok := mockStorage[offset]; ok {
			copy(vec, storedVec)
			return nil
		}
		return fmt.Errorf("vector not found at offset %d", offset)
	}

	// m=16, efConstruction=100, efSearch=50, metric=Euclidean
	hnsw := NewHNSW(3, 16, 100, 50, Euclidean, fetcher)

	// Insert 100 random vectors
	for i := 0; i < 100; i++ {
		id := VectorID(fmt.Sprintf("vec_%d", i))
		vec := []float32{rand.Float32(), rand.Float32(), rand.Float32()}
		offset := int64(i)
		mockStorage[offset] = vec
		err := hnsw.Insert(id, vec, offset)
		if err != nil {
			t.Fatalf("Insert failed: %v", err)
		}
	}

	// Search top 5
	query := []float32{0.5, 0.5, 0.5}
	results, err := hnsw.Search(query, 16, 5, nil)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) != 5 {
		t.Fatalf("Expected 5 results, got %d", len(results))
	}

	// Verify they are sorted by distance
	for i := 1; i < len(results); i++ {
		if results[i-1].Distance > results[i].Distance {
			t.Errorf("Results not sorted: result[%d]=%f > result[%d]=%f", i-1, results[i-1].Distance, i, results[i].Distance)
		}
	}
}

func TestHNSW_TombstoneAndCompact(t *testing.T) {
	mockStorage := make(map[int64][]float32)
	fetcher := func(offset int64, vec []float32) error {
		if storedVec, ok := mockStorage[offset]; ok {
			copy(vec, storedVec)
			return nil
		}
		return fmt.Errorf("vector not found at offset %d", offset)
	}

	hnsw := NewHNSW(3, 16, 100, 50, Euclidean, fetcher)

	// Insert 100 random vectors
	for i := 0; i < 100; i++ {
		id := VectorID(fmt.Sprintf("vec_%d", i))
		vec := []float32{float32(i), 0, 0}
		offset := int64(i)
		mockStorage[offset] = vec
		err := hnsw.Insert(id, vec, offset)
		if err != nil {
			t.Fatalf("Insert failed: %v", err)
		}
	}

	// Delete 10 vectors (tombstone them)
	deletedIDs := make(map[VectorID]bool)
	for i := 0; i < 10; i++ {
		id := VectorID(fmt.Sprintf("vec_%d", i))
		hnsw.Delete(id)
		deletedIDs[id] = true
	}

	// Search and verify deleted vectors are excluded
	query := []float32{0, 0, 0}
	results, err := hnsw.Search(query, 16, 5, nil)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	for _, r := range results {
		if deletedIDs[r.ID] {
			t.Errorf("Deleted vector %s was returned in search results", r.ID)
		}
	}

	// Verify Contains and GetVector handle tombstones correctly
	for id := range deletedIDs {
		if hnsw.Contains(id) {
			t.Errorf("Contains returned true for tombstoned vector %s", id)
		}
		if hnsw.GetVector(id) != nil {
			t.Errorf("GetVector returned non-nil for tombstoned vector %s", id)
		}
	}

	// Verify total nodes still contains all 100 elements before compaction
	if len(hnsw.nodes) != 100 {
		t.Errorf("Expected 100 nodes in graph, got %d", len(hnsw.nodes))
	}

	// Call Compact
	compacted, err := hnsw.Compact()
	if err != nil {
		t.Fatalf("Compact failed: %v", err)
	}

	if len(compacted) != 10 {
		t.Errorf("Expected 10 compacted vectors, got %d", len(compacted))
	}

	// Verify total nodes is now 90
	if len(hnsw.nodes) != 90 {
		t.Errorf("Expected 90 nodes in graph after compaction, got %d", len(hnsw.nodes))
	}

	// Verify none of the neighbors reference deleted nodes
	for id, node := range hnsw.nodes {
		for l, neighbors := range node.connections {
			for _, nID := range neighbors {
				if deletedIDs[nID] {
					t.Errorf("Node %s connects to deleted node %s at layer %d after compaction", id, nID, l)
				}
			}
		}
	}
}
