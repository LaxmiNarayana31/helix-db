package storage

import (
	"context"
	"math/rand"
	"path/filepath"
	"reflect"
	"testing"
)

func TestVectorStorage(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "test.db")
	store, err := NewSQLiteStore(dbPath)
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer store.Close()

	ctx := context.Background()
	collection := "default"
	id := "vec_001"
	store.CreateCollection(ctx, collection, 128, nil)

	// Create random 128-dim vector
	vec := make([]float32, 128)
	for i := range vec {
		vec[i] = rand.Float32()
	}

	// Insert
	if err := store.InsertVector(ctx, collection, id, vec); err != nil {
		t.Fatalf("InsertVector failed: %v", err)
	}

	// Get
	fetched, err := store.GetVector(ctx, collection, id)
	if err != nil {
		t.Fatalf("GetVector failed: %v", err)
	}

	// Assert equality
	if len(fetched) != len(vec) {
		t.Fatalf("expected len %d, got %d", len(vec), len(fetched))
	}
	for i := range vec {
		if fetched[i] != vec[i] {
			t.Fatalf("mismatch at %d: expected %f, got %f", i, vec[i], fetched[i])
		}
	}

	// Test metadata
	meta := map[string]interface{}{
		"tenant": "A1",
		"role":   "admin",
	}

	if err := store.InsertMetadata(ctx, collection, id, meta); err != nil {
		t.Fatalf("InsertMetadata failed: %v", err)
	}

	fetchedMeta, err := store.GetMetadata(ctx, collection, id)
	if err != nil {
		t.Fatalf("GetMetadata failed: %v", err)
	}

	if !reflect.DeepEqual(meta, fetchedMeta) {
		t.Fatalf("metadata mismatch: expected %v, got %v", meta, fetchedMeta)
	}
}
