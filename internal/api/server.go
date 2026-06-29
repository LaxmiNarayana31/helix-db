package api

import (
	"context"
	"helix/internal/collection"
	"helix/internal/index"
	"helix/internal/vectorstore"
	"net/http"
)

type VectorDB interface {
	CreateCollection(name string, dim int, metric index.DistanceMetric, config map[string]interface{}) error
	ListCollections() []collection.CollectionInfo
	DropCollection(ctx context.Context, name string) error
	Add(ctx context.Context, collectionName string, id string, vec []float32, metadata map[string]interface{}) error
	Upsert(ctx context.Context, collectionName string, id string, vec []float32, metadata map[string]interface{}) error
	AddBatch(ctx context.Context, collectionName string, ids []string, vectors [][]float32, metadataList []map[string]interface{}) error
	Delete(ctx context.Context, collectionName string, id string) error
	Search(ctx context.Context, collectionName string, query []float32, k int, opts *vectorstore.SearchOptions) ([]vectorstore.SearchResult, error)
	CheckGraphHealth(ctx context.Context, collectionName string) (vectorstore.HealthReport, error)
	RebuildIndex(ctx context.Context, collectionName string) error
}

type Server struct {
	store VectorDB

	mux     *http.ServeMux
	handler http.Handler
}

func NewServer(store VectorDB) *Server {
	s := &Server{
		store: store,
		mux:   http.NewServeMux(),
	}
	s.routes()
	s.handler = MiddlewareChain(s.mux)
	return s
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.handler.ServeHTTP(w, r)
}

func (s *Server) routes() {
	s.mux.HandleFunc("GET /", s.handleRoot)
	s.mux.HandleFunc("POST /collections", s.handleCreateCollection)
	s.mux.HandleFunc("GET /collections", s.handleListCollections)
	s.mux.HandleFunc("DELETE /collections/{name}", s.handleDropCollection)


	s.mux.HandleFunc("POST /collections/{name}/insert", s.handleInsertVector)
	s.mux.HandleFunc("POST /collections/{name}/upsert", s.handleUpsertVector)
	s.mux.HandleFunc("POST /collections/{name}/batch", s.handleBatchInsert)
	s.mux.HandleFunc("POST /collections/{name}/search", s.handleSearch)
	s.mux.HandleFunc("DELETE /collections/{name}/vectors/{id}", s.handleDeleteVector)

	s.mux.HandleFunc("GET /collections/{name}/health", s.handleGraphHealth)
	s.mux.HandleFunc("POST /collections/{name}/rebuild", s.handleRebuildIndex)
	s.mux.HandleFunc("POST /collections/{name}/hybrid-search", s.handleHybridSearch)
}
