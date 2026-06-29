package collection

import (
	"context"
	"errors"
	"helix/internal/index"
	"helix/internal/storage"
	"sync"
)

type Collection struct {
	Name            string
	Dimension       int
	Metric          index.DistanceMetric
	Index           *index.HNSW
	MmapStore       *storage.MmapVectorStore
	Config          map[string]interface{}
	MetadataCacheMu sync.RWMutex
	MetadataCache   map[index.VectorID]map[string]interface{}
}

type CollectionInfo struct {
	Name      string
	Dimension int
	Metric    index.DistanceMetric
	Config    map[string]interface{}
}

type Manager struct {
	mu          sync.RWMutex
	collections map[string]*Collection
	db          *storage.SQLiteStore
}

func NewManager(db *storage.SQLiteStore) *Manager {
	return &Manager{
		collections: make(map[string]*Collection),
		db:          db,
	}
}

func (m *Manager) RegisterCollection(name string, dim int, metric index.DistanceMetric, idx *index.HNSW, mmapStore *storage.MmapVectorStore, config map[string]interface{}) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, ok := m.collections[name]; ok {
		return errors.New("collection already exists")
	}

	col := &Collection{
		Name:          name,
		Dimension:     dim,
		Metric:        metric,
		Index:         idx,
		MmapStore:     mmapStore,
		Config:        config,
		MetadataCache: make(map[index.VectorID]map[string]interface{}),
	}
	m.collections[name] = col
	return nil
}

func (m *Manager) GetCollection(name string) (*Collection, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	col, ok := m.collections[name]
	if !ok {
		return nil, errors.New("collection not found")
	}
	return col, nil
}

func (m *Manager) ListCollections() []CollectionInfo {
	m.mu.RLock()
	defer m.mu.RUnlock()
	var list []CollectionInfo
	for _, c := range m.collections {
		list = append(list, CollectionInfo{
			Name:      c.Name,
			Dimension: c.Dimension,
			Metric:    c.Metric,
			Config:    c.Config,
		})
	}
	return list
}

func (m *Manager) DropCollection(ctx context.Context, name string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, ok := m.collections[name]; !ok {
		return errors.New("collection not found")
	}

	if err := m.db.DeleteCollection(ctx, name); err != nil {
		return err
	}

	delete(m.collections, name)
	return nil
}

func (m *Manager) CompactAll() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, col := range m.collections {
		compactedIDs, err := col.Index.Compact()
		if err != nil {
			return err
		}
		for _, id := range compactedIDs {
			col.DeleteMetadata(id)
		}
	}
	return nil
}

func (c *Collection) GetMetadata(id index.VectorID) map[string]interface{} {
	c.MetadataCacheMu.RLock()
	defer c.MetadataCacheMu.RUnlock()
	return c.MetadataCache[id]
}

func (c *Collection) SetMetadata(id index.VectorID, meta map[string]interface{}) {
	c.MetadataCacheMu.Lock()
	defer c.MetadataCacheMu.Unlock()
	c.MetadataCache[id] = meta
}

func (c *Collection) DeleteMetadata(id index.VectorID) {
	c.MetadataCacheMu.Lock()
	defer c.MetadataCacheMu.Unlock()
	delete(c.MetadataCache, id)
}
