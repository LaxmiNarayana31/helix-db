package storage

import (
	"context"
	"database/sql"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"time"

	_ "modernc.org/sqlite"
)

type SQLiteStore struct {
	db *sql.DB
}

func NewSQLiteStore(dbPath string) (*SQLiteStore, error) {
	// Use WAL mode for better concurrency and performance with modernc.org/sqlite
	dsn := fmt.Sprintf("file:%s?_pragma=journal_mode(WAL)&_pragma=busy_timeout(5000)&_pragma=synchronous(NORMAL)", dbPath)

	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, err
	}

	// Enable foreign key constraints
	if _, err := db.Exec("PRAGMA foreign_keys = ON;"); err != nil {
		return nil, err
	}

	query := `
	CREATE TABLE IF NOT EXISTS collections (
		name TEXT PRIMARY KEY,
		dim INTEGER NOT NULL,
		config TEXT NOT NULL DEFAULT '{}'
	);
	
	CREATE TABLE IF NOT EXISTS vectors (
		id          TEXT NOT NULL,
		collection  TEXT NOT NULL DEFAULT 'default',
		dim         INTEGER NOT NULL,
		data        BLOB NOT NULL,
		created_at  INTEGER NOT NULL,
		PRIMARY KEY(collection, id),
		FOREIGN KEY(collection) REFERENCES collections(name) ON DELETE CASCADE
	);
	CREATE INDEX IF NOT EXISTS idx_vectors_collection ON vectors(collection);

	CREATE TABLE IF NOT EXISTS metadata (
		vector_id   TEXT NOT NULL,
		collection  TEXT NOT NULL,
		payload     TEXT NOT NULL,
		PRIMARY KEY (collection, vector_id),
		FOREIGN KEY (collection, vector_id) REFERENCES vectors(collection, id) ON DELETE CASCADE
	);
	`
	if _, err := db.Exec(query); err != nil {
		return nil, err
	}

	// Data migration: insert existing collections from vectors table into collections table if they don't exist
	migQuery := `INSERT OR IGNORE INTO collections (name, dim, config) SELECT collection, dim, '{}' FROM vectors GROUP BY collection;`
	if _, err := db.Exec(migQuery); err != nil {
		return nil, err
	}

	if err := db.Ping(); err != nil {
		return nil, err
	}

	// Configure connection pool for concurrent readers
	db.SetMaxOpenConns(100)
	db.SetMaxIdleConns(100)

	return &SQLiteStore{db: db}, nil
}

func (s *SQLiteStore) Close() error {
	if s.db != nil {
		return s.db.Close()
	}
	return nil
}

func (s *SQLiteStore) InsertBatch(ctx context.Context, collection string, ids []string, vectors [][]float32, metadataList []map[string]interface{}) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()

	// Prepare statements
	stmtVec, err := tx.PrepareContext(ctx, "INSERT OR REPLACE INTO vectors (id, collection, dim, data, created_at) VALUES (?, ?, ?, ?, ?)")
	if err != nil {
		return err
	}
	defer stmtVec.Close()

	stmtMetaDel, err := tx.PrepareContext(ctx, "DELETE FROM metadata WHERE collection = ? AND vector_id = ?")
	if err != nil {
		return err
	}
	defer stmtMetaDel.Close()

	stmtMetaIns, err := tx.PrepareContext(ctx, "INSERT INTO metadata (vector_id, collection, payload) VALUES (?, ?, ?)")
	if err != nil {
		return err
	}
	defer stmtMetaIns.Close()

	now := time.Now().UnixNano()

	for i := range ids {
		id := ids[i]
		vec := vectors[i]
		meta := metadataList[i]

		data := encodeVector(vec)
		if _, err := stmtVec.ExecContext(ctx, id, collection, len(vec), data, now); err != nil {
			return err
		}

		if meta != nil {
			// Delete existing meta just in case it's an upsert
			if _, err := stmtMetaDel.ExecContext(ctx, collection, id); err != nil {
				return err
			}
			metaBytes, err := json.Marshal(meta)
			if err != nil {
				return err
			}
			if _, err := stmtMetaIns.ExecContext(ctx, id, collection, string(metaBytes)); err != nil {
				return err
			}
		} else {
			if _, err := stmtMetaDel.ExecContext(ctx, collection, id); err != nil {
				return err
			}
		}
	}

	return tx.Commit()
}

func encodeVector(vec []float32) []byte {
	buf := make([]byte, len(vec)*4)
	for i, v := range vec {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	return buf
}

func decodeVector(data []byte) []float32 {
	if len(data)%4 != 0 {
		return nil
	}
	vec := make([]float32, len(data)/4)
	for i := 0; i < len(vec); i++ {
		vec[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return vec
}

func (s *SQLiteStore) InsertVector(ctx context.Context, collection string, id string, vec []float32) error {
	data := encodeVector(vec)
	createdAt := time.Now().Unix()

	query := `INSERT INTO vectors (id, collection, dim, data, created_at) VALUES (?, ?, ?, ?, ?)`
	_, err := s.db.ExecContext(ctx, query, id, collection, len(vec), data, createdAt)
	return err
}

func (s *SQLiteStore) UpsertVector(ctx context.Context, collection string, id string, vec []float32) error {
	data := encodeVector(vec)
	createdAt := time.Now().Unix()

	query := `INSERT INTO vectors (id, collection, dim, data, created_at) VALUES (?, ?, ?, ?, ?)
	ON CONFLICT(collection, id) DO UPDATE SET data=excluded.data, dim=excluded.dim`
	_, err := s.db.ExecContext(ctx, query, id, collection, len(vec), data, createdAt)
	return err
}

func (s *SQLiteStore) DeleteVector(ctx context.Context, collection string, id string) error {
	query := `DELETE FROM vectors WHERE collection = ? AND id = ?`
	_, err := s.db.ExecContext(ctx, query, collection, id)
	return err
}

func (s *SQLiteStore) GetVector(ctx context.Context, collection string, id string) ([]float32, error) {
	query := `SELECT data FROM vectors WHERE collection = ? AND id = ?`
	row := s.db.QueryRowContext(ctx, query, collection, id)

	var data []byte
	if err := row.Scan(&data); err != nil {
		return nil, err
	}

	return decodeVector(data), nil
}

func (s *SQLiteStore) InsertMetadata(ctx context.Context, collection string, id string, meta map[string]interface{}) error {
	payload, err := json.Marshal(meta)
	if err != nil {
		return err
	}
	query := `INSERT INTO metadata (vector_id, collection, payload) VALUES (?, ?, ?)
	ON CONFLICT(collection, vector_id) DO UPDATE SET payload=excluded.payload`
	_, err = s.db.ExecContext(ctx, query, id, collection, string(payload))
	return err
}

func (s *SQLiteStore) DeleteMetadata(ctx context.Context, collection string, id string) error {
	query := `DELETE FROM metadata WHERE collection = ? AND vector_id = ?`
	_, err := s.db.ExecContext(ctx, query, collection, id)
	return err
}

func (s *SQLiteStore) GetMetadata(ctx context.Context, collection string, id string) (map[string]interface{}, error) {
	query := `SELECT payload FROM metadata WHERE collection = ? AND vector_id = ?`
	row := s.db.QueryRowContext(ctx, query, collection, id)

	var payload string
	if err := row.Scan(&payload); err != nil {
		return nil, err
	}

	var meta map[string]interface{}
	if err := json.Unmarshal([]byte(payload), &meta); err != nil {
		return nil, err
	}

	return meta, nil
}

type Payload struct {
	Metadata map[string]interface{}
	Vector   []float32
}

func (s *SQLiteStore) GetPayloadsBatch(ctx context.Context, collection string, ids []string, includeVectors bool) (map[string]Payload, error) {
	if len(ids) == 0 {
		return make(map[string]Payload), nil
	}

	// Build the IN clause
	placeholders := make([]string, len(ids))
	args := make([]interface{}, 0, len(ids)+1)
	args = append(args, collection)
	for i, id := range ids {
		placeholders[i] = "?"
		args = append(args, id)
	}

	query := "SELECT vector_id, payload FROM metadata WHERE collection = ? AND vector_id IN (" + strings.Join(placeholders, ",") + ")"

	// If includeVectors is true, we should probably do a LEFT JOIN with vectors table, or just fetch from vectors table.
	// Actually, doing a LEFT JOIN guarantees we get the vector even if metadata is missing.
	if includeVectors {
		query = "SELECT v.id, m.payload, v.data FROM vectors v LEFT JOIN metadata m ON v.collection = m.collection AND v.id = m.vector_id WHERE v.collection = ? AND v.id IN (" + strings.Join(placeholders, ",") + ")"
	}

	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	res := make(map[string]Payload, len(ids))
	for rows.Next() {
		var id string
		var payload sql.NullString
		var data []byte

		if includeVectors {
			if err := rows.Scan(&id, &payload, &data); err != nil {
				return nil, err
			}
		} else {
			if err := rows.Scan(&id, &payload); err != nil {
				return nil, err
			}
		}

		p := Payload{}
		if payload.Valid {
			var m map[string]interface{}
			if err := json.Unmarshal([]byte(payload.String), &m); err != nil {
				return nil, err
			}
			p.Metadata = m
		}

		if includeVectors && len(data) > 0 {
			p.Vector = decodeVector(data)
		}

		res[id] = p
	}

	return res, rows.Err()
}

func (s *SQLiteStore) GetMaxCreatedAt(ctx context.Context, collection string) (int64, error) {
	var max sql.NullInt64
	err := s.db.QueryRowContext(ctx, "SELECT MAX(created_at) FROM vectors WHERE collection = ?", collection).Scan(&max)
	if err != nil {
		return 0, err
	}
	if max.Valid {
		return max.Int64, nil
	}
	return 0, nil
}

// IterateVectors returns all vectors for a collection to rebuild the index.
func (s *SQLiteStore) IterateVectors(ctx context.Context, collection string, cb func(id string, vec []float32) error) error {
	rows, err := s.db.QueryContext(ctx, "SELECT id, data FROM vectors WHERE collection = ?", collection)
	if err != nil {
		return err
	}
	defer rows.Close()

	for rows.Next() {
		var id string
		var data []byte
		if err := rows.Scan(&id, &data); err != nil {
			return err
		}
		vec := decodeVector(data)
		if err := cb(id, vec); err != nil {
			return err
		}
	}
	return rows.Err()
}

// IterateMetadata returns all metadata for a collection to rebuild the cache.
func (s *SQLiteStore) IterateMetadata(ctx context.Context, collection string, cb func(id string, meta map[string]interface{}) error) error {
	rows, err := s.db.QueryContext(ctx, "SELECT vector_id, payload FROM metadata WHERE collection = ?", collection)
	if err != nil {
		return err
	}
	defer rows.Close()

	for rows.Next() {
		var id string
		var payload string
		if err := rows.Scan(&id, &payload); err != nil {
			return err
		}
		var meta map[string]interface{}
		if err := json.Unmarshal([]byte(payload), &meta); err != nil {
			return err
		}
		if err := cb(id, meta); err != nil {
			return err
		}
	}
	return rows.Err()
}

// ListCollections returns distinct collections and their dimensions.
func (s *SQLiteStore) CreateCollection(ctx context.Context, name string, dim int, config map[string]interface{}) error {
	configBytes, err := json.Marshal(config)
	if err != nil {
		return err
	}
	if configBytes == nil || string(configBytes) == "null" {
		configBytes = []byte("{}")
	}
	query := `INSERT INTO collections (name, dim, config) VALUES (?, ?, ?)`
	_, err = s.db.ExecContext(ctx, query, name, dim, string(configBytes))
	return err
}

type CollectionMeta struct {
	Name   string
	Dim    int
	Config map[string]interface{}
}

func (s *SQLiteStore) ListCollections(ctx context.Context) (map[string]CollectionMeta, error) {
	rows, err := s.db.QueryContext(ctx, "SELECT name, dim, config FROM collections")
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	cols := make(map[string]CollectionMeta)
	for rows.Next() {
		var name string
		var dim int
		var configStr string
		if err := rows.Scan(&name, &dim, &configStr); err != nil {
			return nil, err
		}
		var config map[string]interface{}
		if err := json.Unmarshal([]byte(configStr), &config); err != nil {
			// fallback
			config = make(map[string]interface{})
		}
		cols[name] = CollectionMeta{
			Name:   name,
			Dim:    dim,
			Config: config,
		}
	}
	return cols, rows.Err()
}

func (s *SQLiteStore) DeleteCollection(ctx context.Context, collection string) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()

	_, err = tx.ExecContext(ctx, "DELETE FROM metadata WHERE collection = ?", collection)
	if err != nil {
		return err
	}

	_, err = tx.ExecContext(ctx, "DELETE FROM vectors WHERE collection = ?", collection)
	if err != nil {
		return err
	}

	return tx.Commit()
}
