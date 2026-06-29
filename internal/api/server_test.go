package api

import (
	"bytes"
	"encoding/json"

	"helix/internal/index"
	"helix/internal/vectorstore"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"
)

func TestAPI_InsertAndSearch(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "api_test.db")
	store, err := vectorstore.NewStore(dbPath)
	if err != nil {
		t.Fatalf("Failed to create store: %v", err)
	}
	defer store.Close()

	server := NewServer(store)
	ts := httptest.NewServer(server)
	defer ts.Close()

	// Initialize collection before inserting
	store.CreateCollection("test", 3, index.Euclidean, nil)

	// 1. Add vector 1
	addReq := InsertRequest{
		ID:     "vec_1",
		Vector: []float32{1.0, 2.0, 3.0},
		Metadata: map[string]interface{}{
			"color": "red",
		},
	}
	body, _ := json.Marshal(addReq)
	resp, err := http.Post(ts.URL+"/collections/test/insert", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("Failed to post: %v", err)
	}
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("Expected 201 Created, got %d", resp.StatusCode)
	}
	resp.Body.Close()

	// 2. Add vector 2
	addReq2 := InsertRequest{
		ID:     "vec_2",
		Vector: []float32{1.1, 2.1, 3.1},
	}
	body2, _ := json.Marshal(addReq2)
	resp2, err := http.Post(ts.URL+"/collections/test/insert", "application/json", bytes.NewReader(body2))
	if err != nil {
		t.Fatalf("Failed to post: %v", err)
	}
	resp2.Body.Close()

	// 3. Search
	searchReq := SearchRequest{
		Vector: []float32{1.0, 2.0, 3.0},
		TopK:   2,
	}
	searchBody, _ := json.Marshal(searchReq)
	respSearch, err := http.Post(ts.URL+"/collections/test/search", "application/json", bytes.NewReader(searchBody))
	if err != nil {
		t.Fatalf("Failed to post search: %v", err)
	}
	defer respSearch.Body.Close()

	if respSearch.StatusCode != http.StatusOK {
		t.Fatalf("Expected 200 OK, got %d", respSearch.StatusCode)
	}

	var searchResp SearchResponse
	if err := json.NewDecoder(respSearch.Body).Decode(&searchResp); err != nil {
		t.Fatalf("Failed to decode response: %v", err)
	}

	if len(searchResp.Results) != 2 {
		t.Fatalf("Expected 2 results, got %d", len(searchResp.Results))
	}

	// First result should be vec_1 since it perfectly matches the query
	if string(searchResp.Results[0].ID) != "vec_1" {
		t.Errorf("Expected first result to be vec_1, got %s", searchResp.Results[0].ID)
	}
	if searchResp.Results[0].Metadata["color"] != "red" {
		t.Errorf("Expected metadata color red, got %v", searchResp.Results[0].Metadata["color"])
	}

	// 4. Search with filter
	filterReq := SearchRequest{
		Vector: []float32{1.1, 2.1, 3.1}, // Closest to vec_2
		TopK:   2,
		Filter: map[string]interface{}{"color": "red"}, // Only vec_1 has this
	}
	filterBody, _ := json.Marshal(filterReq)
	respFilter, err := http.Post(ts.URL+"/collections/test/search", "application/json", bytes.NewReader(filterBody))
	if err != nil {
		t.Fatalf("Failed to post filter search: %v", err)
	}
	defer respFilter.Body.Close()

	var filterResp SearchResponse
	json.NewDecoder(respFilter.Body).Decode(&filterResp)

	// Even though vec_2 is closer to the query, vec_1 should be returned because of the filter
	if len(filterResp.Results) != 1 {
		t.Fatalf("Expected 1 result due to filter, got %d", len(filterResp.Results))
	}
	if string(filterResp.Results[0].ID) != "vec_1" {
		t.Errorf("Expected vec_1 to match filter, got %s", filterResp.Results[0].ID)
	}
}
