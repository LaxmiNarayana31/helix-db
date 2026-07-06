package textindex

import (
	"fmt"
	"helix/internal/index"
	"os"

	"github.com/blevesearch/bleve/v2"
)

type SearchResult struct {
	ID    index.VectorID
	Score float32
}

type TextIndex interface {
	Index(id index.VectorID, text string) error
	Delete(id index.VectorID) error
	Search(query string, k int) ([]SearchResult, error)
	Close() error
}

type BleveIndex struct {
	idx bleve.Index
}

// Ensure BleveIndex implements TextIndex
var _ TextIndex = (*BleveIndex)(nil)

func NewBleveIndex(path string) (*BleveIndex, error) {
	var idx bleve.Index
	var err error

	if _, errStat := os.Stat(path); os.IsNotExist(errStat) {
		// Create a new mapping
		mapping := bleve.NewIndexMapping()
		// We only care about a single "text" field, let's use default mapping
		idx, err = bleve.New(path, mapping)
		if err != nil {
			return nil, fmt.Errorf("failed to create bleve index: %w", err)
		}
	} else {
		idx, err = bleve.Open(path)
		if err != nil {
			return nil, fmt.Errorf("failed to open bleve index: %w", err)
		}
	}

	return &BleveIndex{idx: idx}, nil
}

type Document struct {
	Text string `json:"text"`
}

func (b *BleveIndex) Index(id index.VectorID, text string) error {
	doc := Document{Text: text}
	return b.idx.Index(string(id), doc)
}

func (b *BleveIndex) Delete(id index.VectorID) error {
	return b.idx.Delete(string(id))
}

func (b *BleveIndex) Search(query string, k int) ([]SearchResult, error) {
	q := bleve.NewMatchQuery(query)
	searchReq := bleve.NewSearchRequest(q)
	searchReq.Size = k

	searchRes, err := b.idx.Search(searchReq)
	if err != nil {
		return nil, fmt.Errorf("bleve search failed: %w", err)
	}

	var results []SearchResult
	for _, hit := range searchRes.Hits {
		results = append(results, SearchResult{
			ID:    index.VectorID(hit.ID),
			Score: float32(hit.Score),
		})
	}
	return results, nil
}

func (b *BleveIndex) Close() error {
	if b.idx != nil {
		return b.idx.Close()
	}
	return nil
}
