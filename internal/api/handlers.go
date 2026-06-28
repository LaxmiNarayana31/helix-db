package api

import (
	"encoding/json"
	"helix/internal/filter"
	"helix/internal/vectorstore"
	"net/http"
	"strings"
)

func (s *Server) handleCreateCollection(w http.ResponseWriter, r *http.Request) {
	var req CreateCollectionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if req.Name == "" || req.Dimension <= 0 {
		http.Error(w, "name and positive dimension are required", http.StatusBadRequest)
		return
	}

	if err := s.store.CreateCollection(req.Name, req.Dimension, req.Metric, req.Config); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(req)
}

func (s *Server) handleListCollections(w http.ResponseWriter, r *http.Request) {
	cols := s.store.ListCollections()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(cols)
}

func (s *Server) handleDropCollection(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	if err := s.store.DropCollection(r.Context(), name); err != nil {
		if strings.Contains(err.Error(), "not found") {
			http.Error(w, err.Error(), http.StatusNotFound)
		} else {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) handleInsertVector(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var req InsertRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if err := s.store.Add(r.Context(), name, req.ID, req.Vector, req.Metadata); err != nil {
		if strings.Contains(err.Error(), "dimension mismatch") {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if strings.Contains(err.Error(), "UNIQUE constraint failed") {
			http.Error(w, err.Error(), http.StatusConflict)
			return
		}
		if strings.Contains(err.Error(), "not found") {
			http.Error(w, err.Error(), http.StatusNotFound)
			return
		}
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	w.Write([]byte(`{"id":"` + req.ID + `"}`))
}

func (s *Server) handleUpsertVector(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var req InsertRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if err := s.store.Upsert(r.Context(), name, req.ID, req.Vector, req.Metadata); err != nil {
		if strings.Contains(err.Error(), "dimension mismatch") {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if strings.Contains(err.Error(), "not found") {
			http.Error(w, err.Error(), http.StatusNotFound)
			return
		}
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	w.Write([]byte(`{"id":"` + req.ID + `"}`))
}

func (s *Server) handleBatchInsert(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var req BatchInsertRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	ids := make([]string, len(req.Items))
	vectors := make([][]float32, len(req.Items))
	metas := make([]map[string]interface{}, len(req.Items))

	for i, item := range req.Items {
		ids[i] = item.ID
		vectors[i] = item.Vector
		metas[i] = item.Metadata
	}

	if err := s.store.AddBatch(r.Context(), name, ids, vectors, metas); err != nil {
		if strings.Contains(err.Error(), "dimension mismatch") {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if strings.Contains(err.Error(), "not found") {
			http.Error(w, err.Error(), http.StatusNotFound)
			return
		}
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(map[string]int{"inserted": len(req.Items)})
}

func (s *Server) handleSearch(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var req SearchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if len(req.Vector) == 0 && req.TextQuery == "" {
		http.Error(w, "vector or text_query is required", http.StatusBadRequest)
		return
	}

	k := req.TopK
	if k <= 0 {
		k = 5 // default
	}

	opts := &vectorstore.SearchOptions{
		IncludeVectors: req.IncludeVector,
		TextQuery:      req.TextQuery,
	}

	if req.Filter != nil {
		opts.Filter = filter.Filter(req.Filter)
	}

	results, err := s.store.Search(r.Context(), name, req.Vector, k, opts)

	if err != nil {
		if strings.Contains(err.Error(), "not found") {
			http.Error(w, err.Error(), http.StatusNotFound)
			return
		}
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	if results == nil {
		results = []vectorstore.SearchResult{}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(SearchResponse{Results: results})
}

func (s *Server) handleDeleteVector(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	id := r.PathValue("id")

	if err := s.store.Delete(r.Context(), name, id); err != nil {
		if strings.Contains(err.Error(), "not found") {
			http.Error(w, err.Error(), http.StatusNotFound)
			return
		}
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) handleGraphHealth(w http.ResponseWriter, r *http.Request) {
	colName := r.PathValue("name")
	if colName == "" {
		http.Error(w, "collection name required", http.StatusBadRequest)
		return
	}

	report, err := s.store.CheckGraphHealth(r.Context(), colName)
	if err != nil {
		if strings.Contains(err.Error(), "not found") {
			http.Error(w, err.Error(), http.StatusNotFound)
			return
		}
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	resp := HealthResponse{
		TotalNodes:         report.TotalNodes,
		TombstonedNodes:    report.TombstonedNodes,
		EntryPointLayer:    report.EntryPointLayer,
		AvgDegreePerLayer:  report.AvgDegreePerLayer,
		RebuildRecommended: report.RebuildRecommended,
		Reason:             report.Reason,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *Server) handleRebuildIndex(w http.ResponseWriter, r *http.Request) {
	colName := r.PathValue("name")
	if colName == "" {
		http.Error(w, "collection name required", http.StatusBadRequest)
		return
	}

	err := s.store.RebuildIndex(r.Context(), colName)
	if err != nil {
		if strings.Contains(err.Error(), "not found") {
			http.Error(w, err.Error(), http.StatusNotFound)
			return
		}
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(RebuildResponse{Status: "success"})
}

// handleHybridSearch serves POST /collections/{name}/hybrid-search.
// text_query is required. vector is optional: if provided, results are merged
// via Reciprocal Rank Fusion (k=60); if omitted, pure BM25 text results are returned.
func (s *Server) handleHybridSearch(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var req HybridSearchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if req.TextQuery == "" {
		http.Error(w, "text_query is required for hybrid search", http.StatusBadRequest)
		return
	}

	k := req.TopK
	if k <= 0 {
		k = 5
	}

	opts := &vectorstore.SearchOptions{
		IncludeVectors: req.IncludeVector,
		TextQuery:      req.TextQuery,
	}
	if req.Filter != nil {
		opts.Filter = filter.Filter(req.Filter)
	}

	results, err := s.store.Search(r.Context(), name, req.Vector, k, opts)
	if err != nil {
		if strings.Contains(err.Error(), "not found") {
			http.Error(w, err.Error(), http.StatusNotFound)
			return
		}
		if strings.Contains(err.Error(), "hybrid search not enabled") {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	if results == nil {
		results = []vectorstore.SearchResult{}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(SearchResponse{Results: results})
}

func (s *Server) handleRoot(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write([]byte(`{"status":"running","message":"Helix is running"}`))
}

