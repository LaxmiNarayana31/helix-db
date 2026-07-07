package benchmark

import (
	"fmt"
	"math/rand"
	"sort"
	"testing"

	"helix/internal/index"
)

func randomVector(dim int) []float32 {
	v := make([]float32, dim)
	for i := 0; i < dim; i++ {
		v[i] = rand.Float32()
	}
	return v
}

type bruteForceResult struct {
	id   index.VectorID
	dist float32
}

func TestHNSW_Recall(t *testing.T) {
	// rand.Seed(42) - removed to test variance

	numVectors := 10000
	numQueries := 100
	dim := 128
	topK := 10

	// 1. Generate data
	vectors := make([][]float32, numVectors)
	for i := 0; i < numVectors; i++ {
		vectors[i] = randomVector(dim)
	}

	queries := make([][]float32, numQueries)
	for i := 0; i < numQueries; i++ {
		queries[i] = randomVector(dim)
	}

	mockStorage := make(map[int64][]float32)
	fetcher := func(offset int64, vec []float32) error {
		if storedVec, ok := mockStorage[offset]; ok {
			copy(vec, storedVec)
			return nil
		}
		return fmt.Errorf("vector not found at offset %d", offset)
	}

	// 2. Insert into HNSW
	hnsw := index.NewHNSW(dim, 16, 200, 50, index.Cosine, fetcher)
	for i, vec := range vectors {
		offset := int64(i)
		mockStorage[offset] = vec
		err := hnsw.Insert(index.VectorID(fmt.Sprintf("vec_%d", i)), vec, offset)
		if err != nil {
			t.Fatalf("Insert failed: %v", err)
		}
	}

	// 3. Run queries and compute recall
	totalRecall := 0.0
	minRecall := 1.0
	minRecallIdx := -1

	for qIdx, qVec := range queries {
		// a. Brute force ground truth
		var bfResults []bruteForceResult
		for i, v := range vectors {
			dist, _ := index.CalculateDistance(index.Cosine, qVec, v)
			bfResults = append(bfResults, bruteForceResult{id: index.VectorID(fmt.Sprintf("vec_%d", i)), dist: dist})
		}

		sort.Slice(bfResults, func(i, j int) bool {
			return bfResults[i].dist < bfResults[j].dist
		})

		gtSet := make(map[index.VectorID]bool)
		for i := 0; i < topK; i++ {
			gtSet[bfResults[i].id] = true
		}

		// b. HNSW search
		results, err := hnsw.Search(qVec, 200, topK, nil)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}

		hits := 0
		for _, res := range results {
			if gtSet[res.ID] {
				hits++
			}
		}

		recall := float64(hits) / float64(topK)
		totalRecall += recall
		if recall < minRecall {
			minRecall = recall
			minRecallIdx = qIdx
		}

		if qIdx%20 == 0 {
			t.Logf("Query %d recall: %.2f", qIdx, recall)
		}
	}

	avgRecall := totalRecall / float64(numQueries)
	t.Logf("Average recall@%d: %.4f (%.2f%%)", topK, avgRecall, avgRecall*100)
	t.Logf("Minimum recall@%d: %.2f (query %d)", topK, minRecall, minRecallIdx)

	if avgRecall < 0.90 {
		t.Fatalf("Recall dropped below 90%%! Got %.2f%%", avgRecall*100)
	}
}
