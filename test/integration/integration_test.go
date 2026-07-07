package integration

import (
	"bytes"
	"context"
	"encoding/json"
	"helix/internal/api"

	"helix/internal/vectorstore"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"
)

func TestIntegration_HTTP_API(t *testing.T) {
	// Setup Store
	dbPath := filepath.Join(t.TempDir(), "integration.db")
	store, err := vectorstore.NewStore(dbPath)
	if err != nil {
		t.Fatalf("Failed to create store: %v", err)
	}
	defer store.Close()

	server := api.NewServer(store)
	ts := httptest.NewServer(server)
	defer ts.Close()

	client := ts.Client()

	// 1. Create Collection
	colReq := map[string]interface{}{
		"name":      "users",
		"dimension": 3,
		"metric":    0, // Cosine
	}
	colBody, _ := json.Marshal(colReq)
	resp, err := client.Post(ts.URL+"/collections", "application/json", bytes.NewReader(colBody))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("Expected 201 Created, got %v", resp.StatusCode)
	}
	resp.Body.Close()

	// 2. Insert vector
	insReq := map[string]interface{}{
		"id":     "vec1",
		"vector": []float32{1.0, 0.0, 0.0},
		"metadata": map[string]interface{}{
			"role": "admin",
		},
	}
	insBody, _ := json.Marshal(insReq)
	resp, err = client.Post(ts.URL+"/collections/users/insert", "application/json", bytes.NewReader(insBody))
	if err != nil {
		t.Fatalf("Failed to insert vector: %v", err)
	}
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("Expected 201 Created for insert, got %v", resp.StatusCode)
	}
	resp.Body.Close()

	// 2b. Insert Duplicate Vector (Expect 409 Conflict)
	resp, err = client.Post(ts.URL+"/collections/users/insert", "application/json", bytes.NewReader(insBody))
	if err != nil {
		t.Fatalf("Failed to insert duplicate vector: %v", err)
	}
	if resp.StatusCode != http.StatusConflict {
		t.Fatalf("Expected 409 Conflict for duplicate insert, got %v", resp.StatusCode)
	}
	resp.Body.Close()

	// 2c. Upsert Duplicate Vector (Expect 201 Created)
	insReq["vector"] = []float32{0.0, 1.0, 0.0}
	insBody2, _ := json.Marshal(insReq)
	resp, err = client.Post(ts.URL+"/collections/users/upsert", "application/json", bytes.NewReader(insBody2))
	if err != nil {
		t.Fatalf("Failed to upsert vector: %v", err)
	}
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("Expected 201 Created for upsert, got %v", resp.StatusCode)
	}
	resp.Body.Close()

	// 2d. Insert with Dimension Mismatch (Expect 400 Bad Request)
	badInsReq := map[string]interface{}{
		"id":     "vec2",
		"vector": []float32{1.0, 0.0}, // 2 dimensions instead of 3
	}
	badInsBody, _ := json.Marshal(badInsReq)
	resp, err = client.Post(ts.URL+"/collections/users/insert", "application/json", bytes.NewReader(badInsBody))
	if err != nil {
		t.Fatalf("Failed to insert bad vector: %v", err)
	}
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("Expected 400 Bad Request for dimension mismatch, got %v", resp.StatusCode)
	}
	resp.Body.Close()

	// 3. Search Vector
	searchReq := map[string]interface{}{
		"vector":         []float32{0.0, 1.0, 0.0},
		"top_k":          5,
		"include_vector": true,
	}
	searchBody, _ := json.Marshal(searchReq)
	resp, err = client.Post(ts.URL+"/collections/users/search", "application/json", bytes.NewReader(searchBody))
	if err != nil {
		t.Fatalf("Failed to search: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("Expected 200 OK for search, got %v", resp.StatusCode)
	}
	var searchResp api.SearchResponse
	json.NewDecoder(resp.Body).Decode(&searchResp)
	resp.Body.Close()

	if len(searchResp.Results) != 1 {
		t.Fatalf("Expected 1 result, got %d", len(searchResp.Results))
	}
	if searchResp.Results[0].ID != "vec1" {
		t.Fatalf("Expected result ID vec1, got %s", searchResp.Results[0].ID)
	}
	if len(searchResp.Results[0].Vector) != 3 {
		t.Fatalf("Expected vector included in response, got %v", searchResp.Results[0].Vector)
	}

	// 3b. Search non-existent collection (Expect 404 Not Found)
	resp, err = client.Post(ts.URL+"/collections/missing/search", "application/json", bytes.NewReader(searchBody))
	if err != nil {
		t.Fatalf("Failed to search missing collection: %v", err)
	}
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("Expected 404 Not Found for missing collection search, got %v", resp.StatusCode)
	}
	resp.Body.Close()

	// 4. Batch Insert
	batchReq := map[string]interface{}{
		"items": []map[string]interface{}{
			{
				"id":     "b1",
				"vector": []float32{0.1, 0.1, 0.1},
			},
			{
				"id":     "b2",
				"vector": []float32{0.2, 0.2, 0.2},
			},
		},
	}
	batchBody, _ := json.Marshal(batchReq)
	resp, err = client.Post(ts.URL+"/collections/users/batch", "application/json", bytes.NewReader(batchBody))
	if err != nil {
		t.Fatalf("Failed to batch insert: %v", err)
	}
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("Expected 201 Created for batch insert, got %v", resp.StatusCode)
	}
	resp.Body.Close()

	// 5. Delete Vector
	req, _ := http.NewRequest("DELETE", ts.URL+"/collections/users/vectors/vec1", nil)
	resp, err = client.Do(req)
	if err != nil {
		t.Fatalf("Failed to delete vector: %v", err)
	}
	if resp.StatusCode != http.StatusNoContent {
		t.Fatalf("Expected 204 No Content for delete, got %v", resp.StatusCode)
	}
	resp.Body.Close()

	// Verify delete
	resp, _ = client.Post(ts.URL+"/collections/users/search", "application/json", bytes.NewReader(searchBody))
	var searchRespAfter api.SearchResponse
	json.NewDecoder(resp.Body).Decode(&searchRespAfter)
	resp.Body.Close()

	// Because we deleted vec1, only b1 and b2 are left, both with some distance
	if len(searchRespAfter.Results) != 2 {
		t.Fatalf("Expected 2 results after delete, got %d", len(searchRespAfter.Results))
	}
}

func TestCacheConsistency(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "consistency.db")
	store, err := vectorstore.NewStore(dbPath)
	if err != nil {
		t.Fatalf("Failed to create store: %v", err)
	}
	defer store.Close()

	server := api.NewServer(store)
	ts := httptest.NewServer(server)
	defer ts.Close()

	client := ts.Client()

	// Create collection
	colReq := map[string]interface{}{
		"name":      "users",
		"dimension": 3,
		"metric":    0,
	}
	colBody, _ := json.Marshal(colReq)
	client.Post(ts.URL+"/collections", "application/json", bytes.NewReader(colBody))

	// 1. Insert vector A with metadata {"status": "active"}
	insReq := map[string]interface{}{
		"id":     "vecA",
		"vector": []float32{1.0, 0.0, 0.0},
		"metadata": map[string]interface{}{
			"status": "active",
		},
	}
	insBody, _ := json.Marshal(insReq)
	client.Post(ts.URL+"/collections/users/insert", "application/json", bytes.NewReader(insBody))

	// 2. Search for {"status": "active"} and verify A is found
	searchReq := map[string]interface{}{
		"vector": []float32{1.0, 0.0, 0.0},
		"top_k":  5,
		"filter": map[string]interface{}{
			"status": "active",
		},
	}
	searchBody, _ := json.Marshal(searchReq)
	resp, _ := client.Post(ts.URL+"/collections/users/search", "application/json", bytes.NewReader(searchBody))
	var searchResp api.SearchResponse
	json.NewDecoder(resp.Body).Decode(&searchResp)
	resp.Body.Close()

	if len(searchResp.Results) != 1 || searchResp.Results[0].ID != "vecA" {
		t.Fatalf("Expected vecA in active search, got %v", searchResp.Results)
	}

	// 3. Upsert vector A with new metadata {"status": "inactive"}
	upsertReq := map[string]interface{}{
		"id":     "vecA",
		"vector": []float32{1.0, 0.0, 0.0},
		"metadata": map[string]interface{}{
			"status": "inactive",
		},
	}
	upsertBody, _ := json.Marshal(upsertReq)
	client.Post(ts.URL+"/collections/users/upsert", "application/json", bytes.NewReader(upsertBody))

	// 4. Immediately search with filter {"status": "inactive"} -> verify A is found
	searchReq2 := map[string]interface{}{
		"vector": []float32{1.0, 0.0, 0.0},
		"top_k":  5,
		"filter": map[string]interface{}{
			"status": "inactive",
		},
	}
	searchBody2, _ := json.Marshal(searchReq2)
	resp2, _ := client.Post(ts.URL+"/collections/users/search", "application/json", bytes.NewReader(searchBody2))
	var searchResp2 api.SearchResponse
	json.NewDecoder(resp2.Body).Decode(&searchResp2)
	resp2.Body.Close()

	if len(searchResp2.Results) != 1 || searchResp2.Results[0].ID != "vecA" {
		t.Fatalf("Expected vecA in inactive search, got %v", searchResp2.Results)
	}

	// 5. Immediately search with filter {"status": "active"} -> verify A is NOT found
	resp3, _ := client.Post(ts.URL+"/collections/users/search", "application/json", bytes.NewReader(searchBody))
	var searchResp3 api.SearchResponse
	json.NewDecoder(resp3.Body).Decode(&searchResp3)
	resp3.Body.Close()

	if len(searchResp3.Results) != 0 {
		t.Fatalf("Expected 0 results for active search, got %v", searchResp3.Results)
	}
}

func TestIntegration_TombstoneInteraction(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "tombstone.db")
	store, err := vectorstore.NewStore(dbPath)
	if err != nil {
		t.Fatalf("Failed to create store: %v", err)
	}
	defer store.Close()

	server := api.NewServer(store)
	ts := httptest.NewServer(server)
	defer ts.Close()

	client := ts.Client()

	// Create collection
	colReq := map[string]interface{}{
		"name":      "users",
		"dimension": 3,
		"metric":    0,
	}
	colBody, _ := json.Marshal(colReq)
	client.Post(ts.URL+"/collections", "application/json", bytes.NewReader(colBody))

	// 1. Insert vec_match with {"status": "active"}
	insReq1 := map[string]interface{}{
		"id":     "vec_match",
		"vector": []float32{1.0, 0.0, 0.0},
		"metadata": map[string]interface{}{
			"status": "active",
		},
	}
	insBody1, _ := json.Marshal(insReq1)
	client.Post(ts.URL+"/collections/users/insert", "application/json", bytes.NewReader(insBody1))

	// 2. Insert vec_other with {"status": "active"}
	insReq2 := map[string]interface{}{
		"id":     "vec_other",
		"vector": []float32{0.9, 0.1, 0.0},
		"metadata": map[string]interface{}{
			"status": "active",
		},
	}
	insBody2, _ := json.Marshal(insReq2)
	client.Post(ts.URL+"/collections/users/insert", "application/json", bytes.NewReader(insBody2))

	// 3. Delete vec_match (tombstones it in the index, deletes from SQLite/cache)
	req, _ := http.NewRequest("DELETE", ts.URL+"/collections/users/vectors/vec_match", nil)
	respDel, err := client.Do(req)
	if err != nil {
		t.Fatalf("Failed to delete vector: %v", err)
	}
	if respDel.StatusCode != http.StatusNoContent {
		t.Fatalf("Expected 204 No Content for delete, got %v", respDel.StatusCode)
	}
	respDel.Body.Close()

	// 4. Run a filtered search for {"status": "active"}
	searchReq := map[string]interface{}{
		"vector": []float32{1.0, 0.0, 0.0},
		"top_k":  5,
		"filter": map[string]interface{}{
			"status": "active",
		},
	}
	searchBody, _ := json.Marshal(searchReq)

	resp, _ := client.Post(ts.URL+"/collections/users/search", "application/json", bytes.NewReader(searchBody))
	var searchResp api.SearchResponse
	json.NewDecoder(resp.Body).Decode(&searchResp)
	resp.Body.Close()

	// 5. Assert vec_other is found, but vec_match is not
	foundOther := false
	foundMatch := false
	for _, r := range searchResp.Results {
		if r.ID == "vec_other" {
			foundOther = true
		}
		if r.ID == "vec_match" {
			foundMatch = true
		}
	}

	if !foundOther {
		t.Errorf("Expected vec_other to be found, but it was not")
	}
	if foundMatch {
		t.Errorf("Expected vec_match to be excluded (tombstoned), but it was found")
	}

	// 6. Compact the store manually
	if err := store.Compact(context.Background()); err != nil {
		t.Fatalf("Store compaction failed: %v", err)
	}

	// 7. Re-run search and assert same results (vec_match is still excluded after physical deletion)
	resp2, _ := client.Post(ts.URL+"/collections/users/search", "application/json", bytes.NewReader(searchBody))
	var searchResp2 api.SearchResponse
	json.NewDecoder(resp2.Body).Decode(&searchResp2)
	resp2.Body.Close()

	foundOther2 := false
	foundMatch2 := false
	for _, r := range searchResp2.Results {
		if r.ID == "vec_other" {
			foundOther2 = true
		}
		if r.ID == "vec_match" {
			foundMatch2 = true
		}
	}

	if !foundOther2 {
		t.Errorf("Expected vec_other to be found after compaction, but it was not")
	}
	if foundMatch2 {
		t.Errorf("Expected vec_match to be excluded after compaction, but it was found")
	}
}

func TestIntegration_HybridSearch(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "hybrid.db")
	store, err := vectorstore.NewStore(dbPath)
	if err != nil {
		t.Fatalf("Failed to create store: %v", err)
	}
	defer store.Close()

	server := api.NewServer(store)
	ts := httptest.NewServer(server)
	defer ts.Close()

	client := ts.Client()

	// Create a collection with textField so the bleve index is active.
	colReq := map[string]interface{}{
		"name":      "docs",
		"dimension": 3,
		"metric":    0, // Cosine
		"config": map[string]interface{}{
			"textField": "content",
		},
	}
	colBody, _ := json.Marshal(colReq)
	resp, err := client.Post(ts.URL+"/collections", "application/json", bytes.NewReader(colBody))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("Expected 201 for CreateCollection, got %d", resp.StatusCode)
	}
	resp.Body.Close()

	// Insert doc A: semantically close to query, keyword: "golang vector"
	insA := map[string]interface{}{
		"id":     "docA",
		"vector": []float32{1.0, 0.0, 0.0},
		"metadata": map[string]interface{}{
			"content": "golang vector database search",
		},
	}
	bodyA, _ := json.Marshal(insA)
	resp, _ = client.Post(ts.URL+"/collections/docs/insert", "application/json", bytes.NewReader(bodyA))
	resp.Body.Close()

	// Insert doc B: keyword match "golang" but vector far from query
	insB := map[string]interface{}{
		"id":     "docB",
		"vector": []float32{0.0, 1.0, 0.0},
		"metadata": map[string]interface{}{
			"content": "golang programming language tutorial",
		},
	}
	bodyB, _ := json.Marshal(insB)
	resp, _ = client.Post(ts.URL+"/collections/docs/insert", "application/json", bytes.NewReader(bodyB))
	resp.Body.Close()

	// Insert doc C: no keyword match, semantically medium
	insC := map[string]interface{}{
		"id":     "docC",
		"vector": []float32{0.7, 0.7, 0.0},
		"metadata": map[string]interface{}{
			"content": "database storage engine",
		},
	}
	bodyC, _ := json.Marshal(insC)
	resp, _ = client.Post(ts.URL+"/collections/docs/insert", "application/json", bytes.NewReader(bodyC))
	resp.Body.Close()

	// --- Test 1: Hybrid search (vector + text_query) returns 200 with results ---
	hybridReq := map[string]interface{}{
		"text_query": "golang",
		"vector":     []float32{1.0, 0.0, 0.0},
		"top_k":      3,
	}
	hybridBody, _ := json.Marshal(hybridReq)
	resp, err = client.Post(ts.URL+"/collections/docs/hybrid-search", "application/json", bytes.NewReader(hybridBody))
	if err != nil {
		t.Fatalf("Hybrid search request failed: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("Expected 200 OK for hybrid search, got %d", resp.StatusCode)
	}
	var hybridResp api.SearchResponse
	json.NewDecoder(resp.Body).Decode(&hybridResp)
	resp.Body.Close()

	if len(hybridResp.Results) == 0 {
		t.Fatal("Expected at least one result from hybrid search, got 0")
	}

	// docA should appear: it matches both the vector (closest to [1,0,0]) and
	// the keyword "golang" — highest RRF score expected.
	foundA := false
	for _, r := range hybridResp.Results {
		if r.ID == "docA" {
			foundA = true
		}
	}
	if !foundA {
		t.Errorf("Expected docA in hybrid results (keyword+vector match), got: %v", hybridResp.Results)
	}

	// --- Test 2: Text-only mode (no vector) returns 200 ---
	textOnlyReq := map[string]interface{}{
		"text_query": "golang",
		"top_k":      3,
	}
	textOnlyBody, _ := json.Marshal(textOnlyReq)
	resp, err = client.Post(ts.URL+"/collections/docs/hybrid-search", "application/json", bytes.NewReader(textOnlyBody))
	if err != nil {
		t.Fatalf("Text-only hybrid search request failed: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("Expected 200 OK for text-only hybrid search, got %d", resp.StatusCode)
	}
	var textOnlyResp api.SearchResponse
	json.NewDecoder(resp.Body).Decode(&textOnlyResp)
	resp.Body.Close()

	if len(textOnlyResp.Results) == 0 {
		t.Fatal("Expected results from text-only hybrid search, got 0")
	}

	// --- Test 3: Missing text_query returns 400 ---
	badReq := map[string]interface{}{
		"vector": []float32{1.0, 0.0, 0.0},
		"top_k":  3,
	}
	badBody, _ := json.Marshal(badReq)
	resp, err = client.Post(ts.URL+"/collections/docs/hybrid-search", "application/json", bytes.NewReader(badBody))
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("Expected 400 for missing text_query, got %d", resp.StatusCode)
	}
	resp.Body.Close()

	// --- Test 4: Non-existent collection returns 404 ---
	resp, err = client.Post(ts.URL+"/collections/missing/hybrid-search", "application/json", bytes.NewReader(hybridBody))
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("Expected 404 for unknown collection, got %d", resp.StatusCode)
	}
	resp.Body.Close()
}
