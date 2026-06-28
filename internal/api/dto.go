package api

import (
	"helix/internal/index"
	"helix/internal/vectorstore"
)

type CreateCollectionRequest struct {
	Name      string                 `json:"name"`
	Dimension int                    `json:"dimension"`
	Metric    index.DistanceMetric   `json:"metric"`
	Config    map[string]interface{} `json:"config,omitempty"`
}

type InsertRequest struct {
	ID       string                 `json:"id"`
	Vector   []float32              `json:"vector"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

type BatchInsertRequest struct {
	Items []InsertRequest `json:"items"`
}

type SearchRequest struct {
	Vector        []float32              `json:"vector,omitempty"`
	TextQuery     string                 `json:"text_query,omitempty"`
	TopK          int                    `json:"top_k"`
	Filter        map[string]interface{} `json:"filter,omitempty"`
	IncludeVector bool                   `json:"include_vector,omitempty"`
}

type SearchResponse struct {
	Results []vectorstore.SearchResult `json:"results"`
}

// HybridSearchRequest is the request body for POST /collections/{name}/hybrid-search.
// text_query is mandatory; vector is optional (omit for text-only BM25 mode).
type HybridSearchRequest struct {
	TextQuery     string                 `json:"text_query"`
	Vector        []float32              `json:"vector,omitempty"`
	TopK          int                    `json:"top_k"`
	Filter        map[string]interface{} `json:"filter,omitempty"`
	IncludeVector bool                   `json:"include_vector,omitempty"`
}

type HealthResponse struct {
	TotalNodes         int             `json:"total_nodes"`
	TombstonedNodes    int             `json:"tombstoned_nodes"`
	EntryPointLayer    int             `json:"entry_point_layer"`
	AvgDegreePerLayer  map[int]float64 `json:"avg_degree_per_layer"`
	RebuildRecommended bool            `json:"rebuild_recommended"`
	Reason             string          `json:"reason,omitempty"`
}

type RebuildResponse struct {
	Status string `json:"status"`
}
