package index

type GraphStats struct {
	TotalNodes        int
	TombstonedNodes   int
	EntryPointLayer   int
	AvgDegreePerLayer map[int]float64
}

// GraphStats calculates and returns current health metrics of the HNSW graph.
// It acquires a read lock to safely traverse the nodes.
func (h *HNSW) GraphStats() GraphStats {
	h.mu.RLock()
	defer h.mu.RUnlock()

	stats := GraphStats{
		TotalNodes:        len(h.nodes),
		EntryPointLayer:   h.maxLayer,
		AvgDegreePerLayer: make(map[int]float64),
	}

	layerEdges := make(map[int]int)
	layerNodes := make(map[int]int)

	for _, n := range h.nodes {
		if n.tombstoned {
			stats.TombstonedNodes++
		}

		for l := 0; l <= n.layer; l++ {
			layerNodes[l]++
			if l < len(n.connections) {
				layerEdges[l] += len(n.connections[l])
			}
		}
	}

	for l, nodesCount := range layerNodes {
		if nodesCount > 0 {
			stats.AvgDegreePerLayer[l] = float64(layerEdges[l]) / float64(nodesCount)
		}
	}

	return stats
}
