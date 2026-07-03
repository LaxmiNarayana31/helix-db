package index

import (
	"encoding/gob"
	"os"
)

type NodeSnapshot struct {
	ID           VectorID
	VectorOffset int64
	Layer        int
	Connections  [][]VectorID
	Tombstoned   bool
}

type HNSWSnapshot struct {
	Nodes          map[VectorID]*NodeSnapshot
	EntryPoint     VectorID
	MaxLayer       int
	M              int
	M0             int
	EfConstruction int
	EfSearch       int
	Metric         DistanceMetric
	ML             float64
}

func (h *HNSW) SaveSnapshot(path string) error {
	h.mu.RLock()
	defer h.mu.RUnlock()

	snap := HNSWSnapshot{
		Nodes:          make(map[VectorID]*NodeSnapshot, len(h.nodes)),
		EntryPoint:     h.entryPoint,
		MaxLayer:       h.maxLayer,
		M:              h.m,
		M0:             h.m0,
		EfConstruction: h.efConstruction,
		EfSearch:       h.efSearch,
		Metric:         h.metric,
		ML:             h.mL,
	}

	for id, n := range h.nodes {
		snap.Nodes[id] = &NodeSnapshot{
			ID:           n.id,
			VectorOffset: n.vectorOffset,
			Layer:        n.layer,
			Connections:  n.connections,
			Tombstoned:   n.tombstoned,
		}
	}

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	return gob.NewEncoder(f).Encode(snap)
}

func LoadSnapshot(path string, dim int, fetcher VectorFetcher) (*HNSW, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var snap HNSWSnapshot
	if err := gob.NewDecoder(f).Decode(&snap); err != nil {
		return nil, err
	}

	h := &HNSW{
		nodes:          make(map[VectorID]*node, len(snap.Nodes)),
		entryPoint:     snap.EntryPoint,
		maxLayer:       snap.MaxLayer,
		m:              snap.M,
		m0:             snap.M0,
		efConstruction: snap.EfConstruction,
		efSearch:       snap.EfSearch,
		metric:         snap.Metric,
		mL:             snap.ML,
		dim:            dim,
		fetcher:        fetcher,
	}

	for id, ns := range snap.Nodes {
		h.nodes[id] = &node{
			id:           ns.ID,
			vectorOffset: ns.VectorOffset,
			layer:        ns.Layer,
			connections:  ns.Connections,
			tombstoned:   ns.Tombstoned,
		}
	}

	return h, nil
}
