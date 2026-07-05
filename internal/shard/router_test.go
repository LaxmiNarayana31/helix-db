package shard

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"testing"
	"time"

	"helix/internal/index"
	"helix/internal/vectorstore"
)

func TestShardRouter_Equivalence(t *testing.T) {
	// Create base directory for test
	baseDir, err := os.MkdirTemp("", "shardrouter_test_*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(baseDir)

	// Initialize ShardRouter
	numShards := 4
	router, err := NewShardRouter(numShards, baseDir)
	if err != nil {
		t.Fatalf("Failed to create shard router: %v", err)
	}
	defer router.Close()

	// Initialize single Store for baseline comparison
	singleDir := baseDir + "_single"
	if err := os.MkdirAll(singleDir, 0755); err != nil {
		t.Fatalf("Failed to create single store dir: %v", err)
	}
	singleStore, err := vectorstore.NewStore(singleDir + "/store.db")
	if err != nil {
		t.Fatalf("Failed to create single store: %v", err)
	}
	defer singleStore.Close()

	ctx := context.Background()
	colName := "test_collection"
	dim := 16

	// Create collection in both
	if err := router.CreateCollection(colName, dim, index.Euclidean, nil); err != nil {
		t.Fatalf("Router CreateCollection failed: %v", err)
	}
	if err := singleStore.CreateCollection(colName, dim, index.Euclidean, nil); err != nil {
		t.Fatalf("SingleStore CreateCollection failed: %v", err)
	}

	// Insert vectors
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	numVectors := 1000

	ids := make([]string, numVectors)
	vectors := make([][]float32, numVectors)
	meta := make([]map[string]interface{}, numVectors)

	for i := 0; i < numVectors; i++ {
		ids[i] = fmt.Sprintf("vec-%d", i)
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = r.Float32()
		}
		vectors[i] = vec
		meta[i] = map[string]interface{}{"id": i}
	}

	// Batch insert into both
	if err := router.AddBatch(ctx, colName, ids, vectors, meta); err != nil {
		t.Fatalf("Router AddBatch failed: %v", err)
	}
	if err := singleStore.AddBatch(ctx, colName, ids, vectors, meta); err != nil {
		t.Fatalf("SingleStore AddBatch failed: %v", err)
	}

	// Compare Search results
	numQueries := 10
	k := 5

	for q := 0; q < numQueries; q++ {
		queryVec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			queryVec[j] = r.Float32()
		}

		routerRes, err := router.Search(ctx, colName, queryVec, k, nil)
		if err != nil {
			t.Fatalf("Router Search failed: %v", err)
		}

		singleRes, err := singleStore.Search(ctx, colName, queryVec, k, nil)
		if err != nil {
			t.Fatalf("SingleStore Search failed: %v", err)
		}

		if len(routerRes) != len(singleRes) {
			t.Fatalf("Query %d: length mismatch: router=%d, single=%d", q, len(routerRes), len(singleRes))
		}

		for i := 1; i < len(routerRes); i++ {
			if routerRes[i-1].Score > routerRes[i].Score {
				t.Fatalf("Router results not correctly sorted at index %d", i)
			}
		}

		// Calculate ID overlap between the sharded result and monolithic result.
		// We do not assert 100% exact ID match because Approximate Nearest Neighbor (ANN)
		// traversals depend on graph connectivity. A single monolithic HNSW graph has
		// different edges than 4 isolated subgraphs, meaning their greedy search paths
		// can legitimately diverge and find slightly different local minima.
		// The scatter-gather merge logic itself is exact (merges all local top-Ks and sorts),
		// but the underlying candidate sets from the shards might differ from the monolithic graph.
		matchCount := 0
		for _, rr := range routerRes {
			for _, sr := range singleRes {
				if rr.ID == sr.ID {
					matchCount++
					break
				}
			}
		}
		// At this small scale (1000 vectors, k=5), overlap is typically 100%,
		// but we assert >= 80% (4 out of 5) to account for ANN structural divergence.
		if matchCount < (k * 80 / 100) {
			t.Fatalf("Query %d: overlap too low, expected >= %d matches, got %d", q, (k * 80 / 100), matchCount)
		}
	}

	// Verify routing deterministic
	shardIdx := router.getShardIndex(ids[0])
	res, err := router.shards[shardIdx].Search(ctx, colName, vectors[0], 1, nil)
	if err != nil {
		t.Fatalf("Shard Search failed: %v", err)
	}
	if len(res) == 0 || res[0].ID != index.VectorID(ids[0]) {
		t.Fatalf("Vector not found in expected shard")
	}

	// Test Deletion
	if err := router.Delete(ctx, colName, ids[0]); err != nil {
		t.Fatalf("Router Delete failed: %v", err)
	}

	// Should not be in the shard anymore
	res, _ = router.shards[shardIdx].Search(ctx, colName, vectors[0], 1, nil)
	if len(res) > 0 && res[0].ID == index.VectorID(ids[0]) {
		t.Fatalf("Vector still found in shard after deletion")
	}
}

func TestShardRouter_Health(t *testing.T) {
	baseDir, _ := os.MkdirTemp("", "shardrouter_health_*")
	defer os.RemoveAll(baseDir)

	router, _ := NewShardRouter(2, baseDir)
	defer router.Close()

	ctx := context.Background()
	colName := "health_col"
	router.CreateCollection(colName, 2, index.Euclidean, nil)

	// Add 10 vectors
	for i := 0; i < 10; i++ {
		router.Add(ctx, colName, fmt.Sprintf("vec-%d", i), []float32{float32(i), float32(i)}, nil)
	}

	report, err := router.CheckGraphHealth(ctx, colName)
	if err != nil {
		t.Fatalf("CheckGraphHealth failed: %v", err)
	}

	if report.GraphStats.TotalNodes != 10 {
		t.Fatalf("Expected 10 total nodes, got %d", report.GraphStats.TotalNodes)
	}
	if report.GraphStats.TombstonedNodes != 0 {
		t.Fatalf("Expected 0 tombstoned, got %d", report.GraphStats.TombstonedNodes)
	}

	// Delete 3 vectors
	for i := 0; i < 3; i++ {
		router.Delete(ctx, colName, fmt.Sprintf("vec-%d", i))
	}

	report, _ = router.CheckGraphHealth(ctx, colName)
	if report.GraphStats.TombstonedNodes != 3 {
		t.Fatalf("Expected 3 tombstoned, got %d", report.GraphStats.TombstonedNodes)
	}
	ratio := float64(report.GraphStats.TombstonedNodes) / float64(report.GraphStats.TotalNodes)
	if ratio != 0.3 {
		t.Fatalf("Expected 0.3 ratio, got %f", ratio)
	}
	if !report.RebuildRecommended {
		t.Fatalf("Expected RebuildRecommended = true (0.3 > 0.20)")
	}
}
