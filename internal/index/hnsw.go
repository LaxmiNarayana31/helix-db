package index

import (
	"container/heap"
	"math"
	"math/rand"
	"sync"
)

type SearchResult struct {
	ID       VectorID
	Distance float32
}

// MinHeap for exploring closest nodes (smallest distance at the top)
type MinHeap []SearchResult

func (h MinHeap) Len() int            { return len(h) }
func (h MinHeap) Less(i, j int) bool  { return h[i].Distance < h[j].Distance }
func (h MinHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *MinHeap) Push(x interface{}) { *h = append(*h, x.(SearchResult)) }
func (h *MinHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// MaxHeap for keeping the closest 'ef' nodes (largest distance at the top)
type MaxHeap []SearchResult

func (h MaxHeap) Len() int            { return len(h) }
func (h MaxHeap) Less(i, j int) bool  { return h[i].Distance > h[j].Distance }
func (h MaxHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *MaxHeap) Push(x interface{}) { *h = append(*h, x.(SearchResult)) }
func (h *MaxHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

type HNSW struct {
	nodes          map[VectorID]*node
	entryPoint     VectorID
	maxLayer       int
	m              int
	m0             int
	efConstruction int
	efSearch       int
	metric         DistanceMetric
	mu             sync.RWMutex
	mL             float64
	dim            int

	isRebuilding bool
	pendingOps   []rebuildOp

	fetcher VectorFetcher
}

type VectorFetcher func(offset int64, vec []float32) error

type rebuildOp struct {
	isInsert bool
	id       VectorID
	offset   int64
	vec      []float32 // Still held for replay locally if needed, but not stored in graph. Or we can just drop vec. Wait, if we drop vec, we rely solely on offset to fetch it.

}

func NewHNSW(dim, m, efConstruction, efSearch int, metric DistanceMetric, fetcher VectorFetcher) *HNSW {
	return &HNSW{
		nodes:          make(map[VectorID]*node),
		entryPoint:     "",
		maxLayer:       -1,
		m:              m,
		m0:             m * 2,
		efConstruction: efConstruction,
		efSearch:       efSearch,
		mL:             1.0 / math.Log(float64(m)),
		dim:            dim,
		metric:         metric,
		fetcher:        fetcher,
	}
}

func (h *HNSW) Metric() DistanceMetric {
	return h.metric
}

func (h *HNSW) searchLayer(q []float32, eps []VectorID, ef int, layer int, filter func(VectorID) bool) (*MaxHeap, error) {
	fetchBuf := make([]float32, h.dim)
	var W MaxHeap
	var C MinHeap
	visited := make(map[VectorID]bool, ef*30)

	heap.Init(&W)
	heap.Init(&C)

	for _, ep := range eps {
		epNode := h.nodes[ep]
		err := h.fetcher(epNode.vectorOffset, fetchBuf)
		if err != nil {
			return nil, err
		}
		d, err := CalculateDistance(h.metric, q, fetchBuf)
		if err != nil {
			return nil, err
		}
		heap.Push(&C, SearchResult{ID: ep, Distance: d})
		visited[ep] = true

		if (filter == nil || filter(ep)) && !epNode.tombstoned {
			heap.Push(&W, SearchResult{ID: ep, Distance: d})
			if W.Len() > ef {
				heap.Pop(&W)
			}
		}
	}

	// We need a way to track the worst distance in W for pruning.
	// If W is empty (all entry points filtered out), we don't have a bound from W.
	// We use math.MaxFloat32 as worst distance if W is not full.

	for C.Len() > 0 {
		c := heap.Pop(&C).(SearchResult)

		var worstW float32 = math.MaxFloat32
		if W.Len() == ef {
			worstW = W[0].Distance
		}

		if c.Distance > worstW {
			break
		}

		cNode := h.nodes[c.ID]
		if layer >= len(cNode.connections) {
			continue
		}

		for _, eID := range cNode.connections[layer] {
			if !visited[eID] {
				visited[eID] = true
				eNode := h.nodes[eID]

				err := h.fetcher(eNode.vectorOffset, fetchBuf)
				if err != nil {
					return nil, err
				}
				eDist, err := CalculateDistance(h.metric, q, fetchBuf)
				if err != nil {
					return nil, err
				}

				var worstW float32 = math.MaxFloat32
				if W.Len() == ef {
					worstW = W[0].Distance
				}

				if eDist < worstW || W.Len() < ef {
					heap.Push(&C, SearchResult{ID: eID, Distance: eDist})
					if (filter == nil || filter(eID)) && !eNode.tombstoned {
						heap.Push(&W, SearchResult{ID: eID, Distance: eDist})
						if W.Len() > ef {
							heap.Pop(&W)
						}
					}
				}
			}
		}
	}
	return &W, nil
}

func (h *HNSW) selectNeighborsSimple(candidates *MaxHeap, m int) []VectorID {
	if candidates.Len() <= m {
		res := make([]VectorID, 0, candidates.Len())
		for _, item := range *candidates {
			res = append(res, item.ID)
		}
		return res
	}

	// We clone the heap to avoid mutating the caller's candidate set
	clone := make(MaxHeap, candidates.Len())
	copy(clone, *candidates)

	// Pop from MaxHeap until we have at most m elements left (which will be the m closest)
	for clone.Len() > m {
		heap.Pop(&clone)
	}

	res := make([]VectorID, 0, clone.Len())
	for _, item := range clone {
		res = append(res, item.ID)
	}
	return res
}

func (h *HNSW) Insert(id VectorID, vec []float32, offset int64) error {
	fetchBuf := make([]float32, len(vec))
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.isRebuilding {
		h.pendingOps = append(h.pendingOps, rebuildOp{isInsert: true, id: id, offset: offset})
	}

	// random layer
	f := rand.Float64()
	if f == 0.0 {
		f = 1e-6
	}
	l := int(math.Floor(-math.Log(f) * h.mL))

	newNode := &node{
		id:           id,
		vectorOffset: offset,
		layer:        l,
		connections:  make([][]VectorID, l+1),
	}
	h.nodes[id] = newNode

	if h.entryPoint == "" {
		h.entryPoint = id
		h.maxLayer = l
		return nil
	}

	ep := h.entryPoint
	epLayer := h.maxLayer

	// Phase 1: greedy search from epLayer down to l+1
	for lc := epLayer; lc > l; lc-- {
		changed := true
		for changed {
			changed = false
			epNode := h.nodes[ep]
			err := h.fetcher(epNode.vectorOffset, fetchBuf)
			if err != nil {
				return err
			}
			epDist, err := CalculateDistance(h.metric, vec, fetchBuf)
			if err != nil {
				return err
			}

			if lc >= len(epNode.connections) {
				continue
			}

			for _, eID := range epNode.connections[lc] {
				eNode := h.nodes[eID]
				err := h.fetcher(eNode.vectorOffset, fetchBuf)
				if err != nil {
					return err
				}
				eDist, err := CalculateDistance(h.metric, vec, fetchBuf)
				if err != nil {
					return err
				}
				if eDist < epDist {
					ep = eID
					epDist = eDist
					changed = true
				}
			}
		}
	}

	// Phase 2: from min(l, epLayer) down to 0
	startLayer := l
	if epLayer < startLayer {
		startLayer = epLayer
	}

	eps := []VectorID{ep}

	for lc := startLayer; lc >= 0; lc-- {
		W, err := h.searchLayer(vec, eps, h.efConstruction, lc, nil)
		if err != nil {
			return err
		}

		m := h.m
		if lc == 0 {
			m = h.m0
		}

		neighbors := h.selectNeighborsSimple(W, m)
		newNode.connections[lc] = neighbors

		// Bidirectional connections
		for _, nID := range neighbors {
			nNode := h.nodes[nID]
			// extend connections slice if needed
			if len(nNode.connections) <= lc {
				newConns := make([][]VectorID, lc+1)
				copy(newConns, nNode.connections)
				nNode.connections = newConns
			}
			nNode.connections[lc] = append(nNode.connections[lc], id)

			// Prune if exceeds M
			if len(nNode.connections[lc]) > m {
				var nCandidates MaxHeap
				heap.Init(&nCandidates)

				err := h.fetcher(nNode.vectorOffset, fetchBuf)
				if err != nil {
					return err
				}
				nVecCopy := make([]float32, len(fetchBuf))
				copy(nVecCopy, fetchBuf)

				for _, nnID := range nNode.connections[lc] {
					nnNode := h.nodes[nnID]
					err := h.fetcher(nnNode.vectorOffset, fetchBuf)
					if err == nil {
						nnDist, err := CalculateDistance(h.metric, nVecCopy, fetchBuf)
						if err == nil {
							heap.Push(&nCandidates, SearchResult{ID: nnID, Distance: nnDist})
						}
					}
				}
				nNode.connections[lc] = h.selectNeighborsSimple(&nCandidates, m)
			}
		}

		// eps becomes all elements from W to pass to the next layer
		eps = make([]VectorID, 0, W.Len())
		for _, item := range *W {
			eps = append(eps, item.ID)
		}
	}

	if l > h.maxLayer {
		h.maxLayer = l
		h.entryPoint = id
	}

	return nil
}

func (h *HNSW) Contains(id VectorID) bool {
	h.mu.RLock()
	defer h.mu.RUnlock()
	n, ok := h.nodes[id]
	return ok && !n.tombstoned
}

func (h *HNSW) GetVector(id VectorID) []float32 {
	h.mu.RLock()
	defer h.mu.RUnlock()
	if n, ok := h.nodes[id]; ok && !n.tombstoned {
		vec := make([]float32, h.dim)
		err := h.fetcher(n.vectorOffset, vec)
		if err == nil {
			return vec
		}
	}
	return nil
}

func (h *HNSW) Delete(id VectorID) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.isRebuilding {
		h.pendingOps = append(h.pendingOps, rebuildOp{isInsert: false, id: id})
	}

	if n, ok := h.nodes[id]; ok {
		n.tombstoned = true
	}
	return nil
}

func (h *HNSW) Compact() ([]VectorID, error) {
	h.mu.Lock()
	defer h.mu.Unlock()

	var compactedIDs []VectorID

	// Loop through all tombstoned nodes
	for id, nNode := range h.nodes {
		if nNode.tombstoned {
			// Remove from neighbors
			for l, neighbors := range nNode.connections {
				for _, nID := range neighbors {
					neighborNode, nOk := h.nodes[nID]
					if !nOk {
						continue
					}
					if l >= len(neighborNode.connections) {
						continue
					}
					filtered := make([]VectorID, 0, len(neighborNode.connections[l]))
					for _, currentID := range neighborNode.connections[l] {
						if currentID != id {
							filtered = append(filtered, currentID)
						}
					}
					neighborNode.connections[l] = filtered
				}
			}
			// If it was the entry point, handle it
			if h.entryPoint == id {
				h.entryPoint = ""
				h.maxLayer = -1
			}
			compactedIDs = append(compactedIDs, id)
			delete(h.nodes, id)
		}
	}

	// Re-elect entry point if it was deleted
	if h.entryPoint == "" && len(h.nodes) > 0 {
		newMaxLayer := -1
		var newEp VectorID
		for nID, nNode := range h.nodes {
			if nNode.layer > newMaxLayer {
				newMaxLayer = nNode.layer
				newEp = nID
			}
		}
		h.entryPoint = newEp
		h.maxLayer = newMaxLayer
	}

	return compactedIDs, nil
}

func (h *HNSW) Search(queryVec []float32, ef int, k int, filter func(VectorID) bool) ([]SearchResult, error) {
	fetchBuf := make([]float32, h.dim)
	h.mu.RLock()
	defer h.mu.RUnlock()

	if h.entryPoint == "" {
		return nil, nil
	}

	if ef < k {
		ef = k
	}

	ep := h.entryPoint
	epLayer := h.maxLayer

	// Greedily traverse down to layer 1
	for lc := epLayer; lc > 0; lc-- {
		changed := true
		for changed {
			changed = false
			epNode := h.nodes[ep]
			err := h.fetcher(epNode.vectorOffset, fetchBuf)
			if err != nil {
				return nil, err
			}
			epDist, err := CalculateDistance(h.metric, queryVec, fetchBuf)
			if err != nil {
				return nil, err
			}

			if lc >= len(epNode.connections) {
				continue
			}

			for _, eID := range epNode.connections[lc] {
				eNode := h.nodes[eID]
				err := h.fetcher(eNode.vectorOffset, fetchBuf)
				if err != nil {
					return nil, err
				}
				eDist, err := CalculateDistance(h.metric, queryVec, fetchBuf)
				if err != nil {
					return nil, err
				}
				if eDist < epDist {
					ep = eID
					epDist = eDist
					changed = true
				}
			}
		}
	}

	// Layer 0 search
	W, err := h.searchLayer(queryVec, []VectorID{ep}, ef, 0, filter)
	if err != nil {
		return nil, err
	}

	items := make([]SearchResult, W.Len())
	copy(items, *W)

	// Sort ascending
	for i := 0; i < len(items); i++ {
		for j := i + 1; j < len(items); j++ {
			if items[j].Distance < items[i].Distance {
				items[i], items[j] = items[j], items[i]
			}
		}
	}

	if k > len(items) {
		k = len(items)
	}
	return items[:k], nil
}

// Rebuild extracts all non-tombstoned vectors, builds a brand new HNSW graph,
// and then atomically swaps the internal state. This ensures that the live graph
// remains available for concurrent queries and updates during the rebuild process.
func (h *HNSW) Rebuild() error {
	type vectorEntry struct {
		id     VectorID
		offset int64
	}

	// 1. Extract live vectors and begin buffering mutations under a write lock.
	h.mu.Lock()
	h.isRebuilding = true
	h.pendingOps = nil // Ensure buffer is empty

	liveVectors := make([]vectorEntry, 0, len(h.nodes))
	for id, n := range h.nodes {
		if !n.tombstoned {
			liveVectors = append(liveVectors, vectorEntry{id: id, offset: n.vectorOffset})
		}
	}
	m := h.m
	efConstruction := h.efConstruction
	efSearch := h.efSearch
	metric := h.metric
	dim := h.dim
	h.mu.Unlock()

	// 2. Build the new graph without holding any locks on the live graph.
	newHnsw := NewHNSW(dim, m, efConstruction, efSearch, metric, h.fetcher)
	fetchBuf := make([]float32, dim)
	for _, entry := range liveVectors {
		err := h.fetcher(entry.offset, fetchBuf)
		if err != nil {
			h.mu.Lock()
			h.isRebuilding = false
			h.pendingOps = nil
			h.mu.Unlock()
			return err
		}
		if err := newHnsw.Insert(entry.id, fetchBuf, entry.offset); err != nil {
			// If rebuild fails, we must reset the isRebuilding flag.
			h.mu.Lock()
			h.isRebuilding = false
			h.pendingOps = nil
			h.mu.Unlock()
			return err
		}
	}

	// 3. Replay buffered mutations and atomically swap the state.
	h.mu.Lock()
	defer h.mu.Unlock()

	for _, op := range h.pendingOps {
		if op.isInsert {
			// Ensure it's not already tombstoned or duplicated in newHnsw
			if newHnsw.Contains(op.id) {
				newHnsw.Delete(op.id)
			}
			err := h.fetcher(op.offset, fetchBuf)
			if err == nil {
				newHnsw.Insert(op.id, fetchBuf, op.offset)
			}
		} else {
			newHnsw.Delete(op.id)
		}
	}

	h.isRebuilding = false
	h.pendingOps = nil

	h.nodes = newHnsw.nodes
	h.entryPoint = newHnsw.entryPoint
	h.maxLayer = newHnsw.maxLayer

	return nil
}
