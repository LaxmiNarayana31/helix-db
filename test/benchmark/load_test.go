package benchmark

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"helix/internal/api"

	"helix/internal/index"
	"helix/internal/vectorstore"
)

func TestLoad_ConcurrentSearches(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping load test in short mode.")
	}

	// 1. Setup Store and Server
	dbPath := t.TempDir() + "/load_test.db"
	store, err := vectorstore.NewStore(dbPath)
	if err != nil {
		t.Fatalf("Failed to create store: %v", err)
	}
	defer store.Close()

	server := api.NewServer(store)
	ts := httptest.NewServer(server)
	defer ts.Close()

	// 2. Create Collection
	colName := "load_test_col"
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

	// 3. Insert 50k Vectors using Batch API
	numVectors := 50000
	t.Logf("Inserting %d vectors for load testing...", numVectors)
	batchSize := 1000
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
		resp, err := http.Post(ts.URL+"/collections/"+colName+"/batch", "application/json", bytes.NewBuffer(body))
		if err != nil || (resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated) {
			t.Fatalf("Batch insert failed at %d: err=%v, status=%d", i, err, resp.StatusCode)
		}
		resp.Body.Close()
	}
	t.Log("Insert complete.")

	// 4. Perform 100 concurrent searches
	numWorkers := 100
	numSearchesPerWorker := 10 // Total 1000 searches

	t.Logf("Starting %d concurrent search workers...", numWorkers)
	var wg sync.WaitGroup
	errCh := make(chan error, numWorkers*numSearchesPerWorker)
	latenciesCh := make(chan time.Duration, numWorkers*numSearchesPerWorker)

	transport := &http.Transport{
		MaxIdleConns:        100,
		MaxIdleConnsPerHost: 100,
		MaxConnsPerHost:     100,
	}
	sharedClient := &http.Client{
		Transport: transport,
		Timeout:   10 * time.Second,
	}

	start := time.Now()
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for s := 0; s < numSearchesPerWorker; s++ {
				qVec := randomVector(128)
				searchReq := api.SearchRequest{
					Vector:        qVec,
					TopK:          10,
					IncludeVector: false,
				}
				body, _ := json.Marshal(searchReq)

				reqStart := time.Now()
				resp, err := sharedClient.Post(ts.URL+"/collections/"+colName+"/search", "application/json", bytes.NewBuffer(body))
				if err != nil {
					errCh <- fmt.Errorf("Worker %d search err: %v", workerID, err)
					return
				}

				if resp.StatusCode != http.StatusOK {
					b, _ := io.ReadAll(resp.Body)
					errCh <- fmt.Errorf("Worker %d bad status %d: %s", workerID, resp.StatusCode, string(b))
					resp.Body.Close()
					return
				}
				io.Copy(io.Discard, resp.Body)
				resp.Body.Close()
				latenciesCh <- time.Since(reqStart)
			}
		}(w)
	}

	wg.Wait()
	close(errCh)
	close(latenciesCh)
	duration := time.Since(start)

	var errors []error
	for err := range errCh {
		errors = append(errors, err)
	}

	if len(errors) > 0 {
		t.Fatalf("Load test failed with %d errors. First error: %v", len(errors), errors[0])
	}

	var latencies []time.Duration
	for l := range latenciesCh {
		latencies = append(latencies, l)
	}

	importSort := func(arr []time.Duration) {
		for i := 0; i < len(arr); i++ {
			for j := i + 1; j < len(arr); j++ {
				if arr[i] > arr[j] {
					arr[i], arr[j] = arr[j], arr[i]
				}
			}
		}
	}
	importSort(latencies)

	var p50, p99 time.Duration
	if len(latencies) > 0 {
		p50 = latencies[len(latencies)*50/100]
		p99 = latencies[len(latencies)*99/100]
	}

	totalSearches := numWorkers * numSearchesPerWorker
	throughput := float64(totalSearches) / duration.Seconds()
	t.Logf("Load test complete: %d searches in %v (%.2f requests/sec)", totalSearches, duration, throughput)
	t.Logf("Latencies: p50 = %v, p99 = %v", p50, p99)
}
