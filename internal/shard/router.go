package shard

import (
	"context"
	"fmt"
	"hash/fnv"
	"os"
	"path/filepath"
	"sort"

	"helix/internal/collection"
	"helix/internal/index"
	"helix/internal/vectorstore"

	"golang.org/x/sync/errgroup"
)

type ShardRouter struct {
	shards []*vectorstore.Store
}

func NewShardRouter(numShards int, baseDir string) (*ShardRouter, error) {
	if numShards <= 0 {
		return nil, fmt.Errorf("numShards must be > 0")
	}

	shards := make([]*vectorstore.Store, numShards)
	for i := 0; i < numShards; i++ {
		shardDir := filepath.Join(baseDir, fmt.Sprintf("shard-%d", i))
		if err := os.MkdirAll(shardDir, 0755); err != nil {
			return nil, fmt.Errorf("failed to create shard directory %s: %w", shardDir, err)
		}
		shardDbPath := filepath.Join(shardDir, "store.db")
		store, err := vectorstore.NewStore(shardDbPath)
		if err != nil {
			return nil, fmt.Errorf("failed to initialize shard %d: %w", i, err)
		}
		shards[i] = store
	}

	return &ShardRouter{shards: shards}, nil
}

func (r *ShardRouter) Close() error {
	var firstErr error
	for _, shard := range r.shards {
		if err := shard.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

func (r *ShardRouter) getShardIndex(id string) int {
	h := fnv.New64a()
	h.Write([]byte(id))
	return int(h.Sum64() % uint64(len(r.shards)))
}

func (r *ShardRouter) CreateCollection(name string, dim int, metric index.DistanceMetric, config map[string]interface{}) error {
	var g errgroup.Group
	for _, shard := range r.shards {
		shard := shard
		g.Go(func() error {
			return shard.CreateCollection(name, dim, metric, config)
		})
	}
	return g.Wait()
}

func (r *ShardRouter) ListCollections() []collection.CollectionInfo {
	if len(r.shards) == 0 {
		return nil
	}
	// We assume collections exist globally across all shards.
	// Returning from shard-0 is sufficient since metadata is mirrored.
	return r.shards[0].ListCollections()
}

func (r *ShardRouter) DropCollection(ctx context.Context, name string) error {
	var g errgroup.Group
	for _, shard := range r.shards {
		shard := shard
		g.Go(func() error {
			return shard.DropCollection(ctx, name)
		})
	}
	return g.Wait()
}

func (r *ShardRouter) Add(ctx context.Context, collectionName string, id string, vec []float32, metadata map[string]interface{}) error {
	shardIdx := r.getShardIndex(id)
	return r.shards[shardIdx].Add(ctx, collectionName, id, vec, metadata)
}

func (r *ShardRouter) Upsert(ctx context.Context, collectionName string, id string, vec []float32, metadata map[string]interface{}) error {
	shardIdx := r.getShardIndex(id)
	return r.shards[shardIdx].Upsert(ctx, collectionName, id, vec, metadata)
}

func (r *ShardRouter) AddBatch(ctx context.Context, collectionName string, ids []string, vectors [][]float32, metadataList []map[string]interface{}) error {
	shardReqs := make(map[int]struct {
		ids          []string
		vectors      [][]float32
		metadataList []map[string]interface{}
	})

	for i, id := range ids {
		shardIdx := r.getShardIndex(id)
		req := shardReqs[shardIdx]
		req.ids = append(req.ids, id)
		req.vectors = append(req.vectors, vectors[i])
		if len(metadataList) > i {
			req.metadataList = append(req.metadataList, metadataList[i])
		}
		shardReqs[shardIdx] = req
	}

	g, ctx := errgroup.WithContext(ctx)
	for i, req := range shardReqs {
		i, req := i, req
		g.Go(func() error {
			return r.shards[i].AddBatch(ctx, collectionName, req.ids, req.vectors, req.metadataList)
		})
	}
	return g.Wait()
}

func (r *ShardRouter) Delete(ctx context.Context, collectionName string, id string) error {
	shardIdx := r.getShardIndex(id)
	return r.shards[shardIdx].Delete(ctx, collectionName, id)
}

func (r *ShardRouter) Search(ctx context.Context, collectionName string, query []float32, k int, opts *vectorstore.SearchOptions) ([]vectorstore.SearchResult, error) {
	var g errgroup.Group
	resultsCh := make(chan []vectorstore.SearchResult, len(r.shards))

	for _, shard := range r.shards {
		shard := shard
		g.Go(func() error {
			res, err := shard.Search(ctx, collectionName, query, k, opts)
			if err != nil {
				return err
			}
			resultsCh <- res
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return nil, err
	}
	close(resultsCh)

	var allResults []vectorstore.SearchResult
	for res := range resultsCh {
		allResults = append(allResults, res...)
	}

	// Sort ascending by Score (distance)
	sort.Slice(allResults, func(i, j int) bool {
		return allResults[i].Score < allResults[j].Score
	})

	if len(allResults) > k {
		allResults = allResults[:k]
	}

	return allResults, nil
}

func (r *ShardRouter) CheckGraphHealth(ctx context.Context, collectionName string) (vectorstore.HealthReport, error) {
	var g errgroup.Group
	reports := make(chan vectorstore.HealthReport, len(r.shards))

	for _, shard := range r.shards {
		shard := shard
		g.Go(func() error {
			report, err := shard.CheckGraphHealth(ctx, collectionName)
			if err != nil {
				return err
			}
			reports <- report
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return vectorstore.HealthReport{}, err
	}
	close(reports)

	var totalNodes, tombstonedNodes int
	for report := range reports {
		totalNodes += report.GraphStats.TotalNodes
		tombstonedNodes += report.GraphStats.TombstonedNodes
	}

	ratio := float64(0)
	if totalNodes > 0 {
		ratio = float64(tombstonedNodes) / float64(totalNodes)
	}

	reason := ""
	if ratio > 0.20 {
		reason = fmt.Sprintf("High tombstone ratio globally (%.2f%%)", ratio*100)
	}
	return vectorstore.HealthReport{
		GraphStats: index.GraphStats{
			TotalNodes:      totalNodes,
			TombstonedNodes: tombstonedNodes,
		},
		RebuildRecommended: ratio > 0.20,
		Reason:             reason,
	}, nil
}

func (r *ShardRouter) RebuildIndex(ctx context.Context, collectionName string) error {
	var g errgroup.Group
	for _, shard := range r.shards {
		shard := shard
		g.Go(func() error {
			return shard.RebuildIndex(ctx, collectionName)
		})
	}
	return g.Wait()
}
