package vectorstore

import (
	"context"
	"helix/internal/index"
	"path/filepath"
	"testing"
)

func TestStore_HybridSearch(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "test_hybrid.db")
	store, err := NewStore(dbPath)
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer store.Close()

	ctx := context.Background()
	config := map[string]interface{}{
		"textField": "text",
	}

	err = store.CreateCollection("hybrid_col", 3, index.Euclidean, config)
	if err != nil {
		t.Fatalf("failed to create collection: %v", err)
	}

	// Insert 3 documents
	docs := []struct {
		id   string
		vec  []float32
		meta map[string]interface{}
	}{
		{
			id:  "v1",
			vec: []float32{1.0, 1.0, 1.0},
			meta: map[string]interface{}{
				"text": "the quick brown fox jumps over the lazy dog",
			},
		},
		{
			id:  "v2",
			vec: []float32{1.0, 1.0, 1.1}, // very close to v1
			meta: map[string]interface{}{
				"text": "the quick brown cat", // missing lazy dog
			},
		},
		{
			id:  "v3",
			vec: []float32{9.0, 9.0, 9.0}, // far away
			meta: map[string]interface{}{
				"text": "a slow yellow fox resting by the lazy dog", // strong text match
			},
		},
	}

	for _, doc := range docs {
		err = store.Add(ctx, "hybrid_col", doc.id, doc.vec, doc.meta)
		if err != nil {
			t.Fatalf("Add failed: %v", err)
		}
	}

	// Wait a brief moment for Bleve to index asynchronously if needed (it should be sync in memory/scorch but just in case)
	// scorch is usually synchronous for Index() method, but let's test directly.

	// 1. Vector Search only (Target v1)
	optsVec := &SearchOptions{
		IncludeVectors: true,
	}
	resVec, err := store.Search(ctx, "hybrid_col", []float32{1.0, 1.0, 1.0}, 2, optsVec)
	if err != nil {
		t.Fatalf("vector search failed: %v", err)
	}
	if len(resVec) != 2 || resVec[0].ID != "v1" || resVec[1].ID != "v2" {
		t.Errorf("expected v1 and v2 for vector search, got: %v", resVec)
	}

	// 2. Text Search only (Target "lazy dog")
	optsText := &SearchOptions{
		TextQuery: "lazy dog",
	}
	resText, err := store.Search(ctx, "hybrid_col", nil, 2, optsText)
	if err != nil {
		t.Fatalf("text search failed: %v", err)
	}
	// "lazy dog" is in v1 and v3.
	if len(resText) != 2 {
		t.Errorf("expected 2 results for text search, got %d", len(resText))
	} else {
		// Just ensure v1 and v3 are present
		foundV1 := false
		foundV3 := false
		for _, r := range resText {
			if r.ID == "v1" {
				foundV1 = true
			}
			if r.ID == "v3" {
				foundV3 = true
			}
		}
		if !foundV1 || !foundV3 {
			t.Errorf("expected v1 and v3 in text search results, got %v", resText)
		}
	}

	// 3. Hybrid Search: Vector query for v1 but text query favors v3
	optsHybrid := &SearchOptions{
		TextQuery: "lazy dog",
	}
	resHybrid, err := store.Search(ctx, "hybrid_col", []float32{9.0, 9.0, 9.0}, 2, optsHybrid)
	if err != nil {
		t.Fatalf("hybrid search failed: %v", err)
	}
	// Vector {9,9,9} favors v3 strongly. Text "lazy dog" favors v3 and v1.
	// Therefore v3 should be the absolute top result, followed by v1.
	if len(resHybrid) != 2 {
		t.Errorf("expected 2 results for hybrid search, got %d", len(resHybrid))
	} else {
		if resHybrid[0].ID != "v3" {
			t.Errorf("expected v3 as top result, got %v", resHybrid[0].ID)
		}
	}
}

func TestStore_TextDelete(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "test_hybrid_del.db")
	store, err := NewStore(dbPath)
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer store.Close()

	ctx := context.Background()
	config := map[string]interface{}{
		"textField": "text",
	}

	err = store.CreateCollection("hybrid_del_col", 3, index.Euclidean, config)
	if err != nil {
		t.Fatalf("failed to create collection: %v", err)
	}

	err = store.Add(ctx, "hybrid_del_col", "v1", []float32{1.0, 1.0, 1.0}, map[string]interface{}{
		"text": "hello world",
	})
	if err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	optsText := &SearchOptions{
		TextQuery: "hello",
	}
	resText, err := store.Search(ctx, "hybrid_del_col", nil, 2, optsText)
	if err != nil {
		t.Fatalf("text search failed: %v", err)
	}
	if len(resText) != 1 || resText[0].ID != "v1" {
		t.Errorf("expected v1 in text search, got %v", resText)
	}

	// Delete
	err = store.Delete(ctx, "hybrid_del_col", "v1")
	if err != nil {
		t.Fatalf("Delete failed: %v", err)
	}

	resTextAfter, err := store.Search(ctx, "hybrid_del_col", nil, 2, optsText)
	if err != nil {
		t.Fatalf("text search after delete failed: %v", err)
	}
	if len(resTextAfter) != 0 {
		t.Errorf("expected 0 results after delete, got %v", resTextAfter)
	}
}
