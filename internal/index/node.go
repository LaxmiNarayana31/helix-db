package index

type VectorID string

type DistanceMetric int

const (
	Cosine DistanceMetric = iota
	DotProduct
	Euclidean
)

type node struct {
	id           VectorID
	vectorOffset int64
	layer        int
	connections  [][]VectorID // layer -> list of neighbors
	tombstoned   bool
}
