package index

import (
	"errors"
	"math"
)

var ErrDimensionMismatch = errors.New("vector dimensions do not match")

// CalculateDistance computes the distance between two vectors using the specified metric.
// For all metrics returned here, a smaller value means the vectors are "closer" or more similar.
func CalculateDistance(metric DistanceMetric, a, b []float32) (float32, error) {
	if len(a) != len(b) {
		return 0, ErrDimensionMismatch
	}
	if len(a) == 0 {
		return 0, nil
	}

	switch metric {
	case Euclidean:
		var sum float32
		_ = b[len(a)-1] // bounds check elimination
		for i := 0; i < len(a); i++ {
			diff := a[i] - b[i]
			sum += diff * diff
		}
		return float32(math.Sqrt(float64(sum))), nil

	case DotProduct:
		// For DotProduct as a "distance" metric (where smaller is closer),
		// we return the negative dot product.
		var dot float32
		_ = b[len(a)-1] // bounds check elimination
		for i := 0; i < len(a); i++ {
			dot += a[i] * b[i]
		}
		return -dot, nil

	case Cosine:
		var dot, normA, normB float32
		_ = b[len(a)-1] // bounds check elimination
		for i := 0; i < len(a); i++ {
			vA := a[i]
			vB := b[i]
			dot += vA * vB
			normA += vA * vA
			normB += vB * vB
		}
		if normA == 0 || normB == 0 {
			return 1.0, nil // Cosine similarity undefined for zero vector; distance = 1.0
		}
		similarity := float64(dot) / (math.Sqrt(float64(normA)) * math.Sqrt(float64(normB)))
		// Clamp similarity to [-1, 1] to avoid floating point inaccuracies causing distance < 0
		if similarity > 1.0 {
			similarity = 1.0
		} else if similarity < -1.0 {
			similarity = -1.0
		}
		return float32(1.0 - similarity), nil

	default:
		return 0, errors.New("unknown distance metric")
	}
}
