package storage

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestMmapVectorStore_Basic(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "mmapstore_test_*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	dbPath := filepath.Join(tempDir, "test.vecs")
	dim := 4

	store, err := OpenMmapVectorStore(dbPath, dim)
	if err != nil {
		t.Fatalf("Failed to open mmap store: %v", err)
	}
	defer store.Close()

	vec1 := []float32{1.0, 2.0, 3.0, 4.0}
	vec2 := []float32{5.0, 6.0, 7.0, 8.0}

	offset1, err := store.Append(vec1)
	if err != nil {
		t.Fatalf("Failed to append vec1: %v", err)
	}
	if offset1 != 0 {
		t.Fatalf("Expected offset 0 for first append, got %d", offset1)
	}

	offset2, err := store.Append(vec2)
	if err != nil {
		t.Fatalf("Failed to append vec2: %v", err)
	}
	if offset2 != int64(dim*4) {
		t.Fatalf("Expected offset %d for second append, got %d", dim*4, offset2)
	}

	// Must remap to see appended data in reader
	if err := store.Remap(); err != nil {
		t.Fatalf("Failed to remap: %v", err)
	}

	readVec1 := make([]float32, dim)
	err = store.Read(offset1, readVec1)
	if err != nil {
		t.Fatalf("Failed to read vec1: %v", err)
	}
	if !reflect.DeepEqual(readVec1, vec1) {
		t.Fatalf("Read vec1 mismatch: expected %v, got %v", vec1, readVec1)
	}

	readVec2 := make([]float32, dim)
	err = store.Read(offset2, readVec2)
	if err != nil {
		t.Fatalf("Failed to read vec2: %v", err)
	}
	if !reflect.DeepEqual(readVec2, vec2) {
		t.Fatalf("Read vec2 mismatch: expected %v, got %v", vec2, readVec2)
	}

	// Test append dimension mismatch
	_, err = store.Append([]float32{1.0})
	if err == nil {
		t.Fatalf("Expected error when appending wrong dimension vector")
	}

	// Reopen existing store
	store.Close()
	store2, err := OpenMmapVectorStore(dbPath, dim)
	if err != nil {
		t.Fatalf("Failed to reopen store: %v", err)
	}
	defer store2.Close()

	readVec2Reopen := make([]float32, dim)
	err = store2.Read(offset2, readVec2Reopen)
	if err != nil {
		t.Fatalf("Failed to read vec2 after reopen: %v", err)
	}
	if !reflect.DeepEqual(readVec2Reopen, vec2) {
		t.Fatalf("Read vec2 mismatch after reopen: expected %v, got %v", vec2, readVec2Reopen)
	}
}
