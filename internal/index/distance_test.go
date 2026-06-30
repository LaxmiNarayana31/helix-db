package index

import (
	"math"
	"testing"
)

func TestDistanceMetrics(t *testing.T) {
	tests := []struct {
		name     string
		metric   DistanceMetric
		a        []float32
		b        []float32
		expected float32
	}{
		{
			name:     "Euclidean (0,0) to (3,4)",
			metric:   Euclidean,
			a:        []float32{0, 0},
			b:        []float32{3, 4},
			expected: 5.0,
		},
		{
			name:     "Cosine [1,0] to [0,1]",
			metric:   Cosine,
			a:        []float32{1, 0},
			b:        []float32{0, 1},
			expected: 1.0, // similarity is 0, distance is 1.0
		},
		{
			name:     "Cosine identical vectors",
			metric:   Cosine,
			a:        []float32{1, 2, 3},
			b:        []float32{1, 2, 3},
			expected: 0.0, // similarity is 1.0, distance is 0.0
		},
		{
			name:     "Cosine opposite vectors",
			metric:   Cosine,
			a:        []float32{1, 2, 3},
			b:        []float32{-1, -2, -3},
			expected: 2.0, // similarity is -1.0, distance is 2.0
		},
		{
			name:     "DotProduct distance",
			metric:   DotProduct,
			a:        []float32{1, 2},
			b:        []float32{3, 4},
			expected: -11.0, // 1*3 + 2*4 = 11. Negated for distance = -11.0
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			dist, err := CalculateDistance(tc.metric, tc.a, tc.b)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			// Float comparison with tolerance
			if math.Abs(float64(dist-tc.expected)) > 1e-6 {
				t.Errorf("expected %f, got %f", tc.expected, dist)
			}
		})
	}
}

func TestDistanceMismatch(t *testing.T) {
	_, err := CalculateDistance(Euclidean, []float32{1, 2}, []float32{1})
	if err != ErrDimensionMismatch {
		t.Errorf("expected ErrDimensionMismatch, got %v", err)
	}
}

var GlobalDistance float32

func BenchmarkDistance(b *testing.B) {
	vec1 := make([]float32, 128)
	vec2 := make([]float32, 128)
	for i := 0; i < 128; i++ {
		vec1[i] = float32(i) * 0.1
		vec2[i] = float32(128-i) * 0.1
	}

	b.Run("Euclidean", func(b *testing.B) {
		var dist float32
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			dist, _ = CalculateDistance(Euclidean, vec1, vec2)
		}
		GlobalDistance = dist
	})

	b.Run("DotProduct", func(b *testing.B) {
		var dist float32
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			dist, _ = CalculateDistance(DotProduct, vec1, vec2)
		}
		GlobalDistance = dist
	})

	b.Run("Cosine", func(b *testing.B) {
		var dist float32
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			dist, _ = CalculateDistance(Cosine, vec1, vec2)
		}
		GlobalDistance = dist
	})
}
