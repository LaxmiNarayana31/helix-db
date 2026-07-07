package benchmark

import (
	"bytes"
	"encoding/json"
	"fmt"
	"helix/internal/api"
	"helix/internal/index"
	"helix/internal/vectorstore"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

func TestStress_ConcurrentInsertAndSearch(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "vectordb_stress_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	dbPath := filepath.Join(tempDir, "stress.db")
	store, err := vectorstore.NewStore(dbPath)
	if err != nil {
		t.Fatalf("Failed to create store: %v", err)
	}
	defer store.Close()

	server := api.NewServer(store)
	ts := httptest.NewServer(server)
	defer ts.Close()

	// Create Collection
	colName := "stress_test_col"
	createReq := api.CreateCollectionRequest{
		Name:      colName,
		Dimension: 128,
		Metric:    index.Cosine,
	}
	body, _ := json.Marshal(createReq)
	resp, err := http.Post(ts.URL+"/collections", "application/json", bytes.NewBuffer(body))
	if err != nil || resp.StatusCode != http.StatusCreated {
		t.Fatalf("Failed to create collection: %v", err)
	}
	resp.Body.Close()

	var wg sync.WaitGroup
	errCh := make(chan error, 1000)
	done := make(chan struct{})

	// Start 10 background Search workers
	transport := &http.Transport{
		MaxIdleConns:        100,
		MaxIdleConnsPerHost: 100,
		MaxConnsPerHost:     100,
	}
	sharedClient := &http.Client{
		Transport: transport,
		Timeout:   60 * time.Second,
	}

	for w := 0; w < 10; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for {
				select {
				case <-done:
					return
				default:
					qVec := randomVector(128)
					searchReq := api.SearchRequest{
						Vector:        qVec,
						TopK:          10,
						IncludeVector: false,
					}
					body, _ := json.Marshal(searchReq)
					resp, err := sharedClient.Post(ts.URL+"/collections/"+colName+"/search", "application/json", bytes.NewBuffer(body))
					if err != nil {
						errCh <- fmt.Errorf("Search err: %v", err)
						return
					}
					if resp.StatusCode != http.StatusOK {
						errCh <- fmt.Errorf("Search failed with status: %d", resp.StatusCode)
						return
					}
					io.Copy(io.Discard, resp.Body)
					resp.Body.Close()
				}
			}
		}(w)
	}

	// Run Batch Inserts concurrently (triggers Remap)
	numVectors := 10000
	batchSize := 500
	for i := 0; i < numVectors; i += batchSize {
		var req api.BatchInsertRequest
		for j := 0; j < batchSize; j++ {
			id := fmt.Sprintf("vec_%d", i+j)
			vec := randomVector(128)
			req.Items = append(req.Items, api.InsertRequest{
				ID:       id,
				Vector:   vec,
				Metadata: map[string]interface{}{"idx": i + j},
			})
		}

		body, _ := json.Marshal(req)
		resp, err := sharedClient.Post(ts.URL+"/collections/"+colName+"/batch", "application/json", bytes.NewBuffer(body))
		if err != nil {
			t.Fatalf("Batch insert failed at %d: err=%v", i, err)
		}
		if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
			t.Fatalf("Batch insert failed at %d: status=%d", i, resp.StatusCode)
		}
		resp.Body.Close()
	}

	// Stop workers
	close(done)
	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Errorf("Worker encountered error: %v", err)
	}
}
