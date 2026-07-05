package storage

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"sync"

	"golang.org/x/exp/mmap"
)

type MmapVectorStore struct {
	path   string
	dim    int
	file   *os.File
	reader *mmap.ReaderAt
	mu     sync.RWMutex
	size   int64
}

func OpenMmapVectorStore(path string, dim int) (*MmapVectorStore, error) {
	file, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0644)
	if err != nil {
		return nil, fmt.Errorf("failed to open vector file: %w", err)
	}

	info, err := file.Stat()
	if err != nil {
		file.Close()
		return nil, fmt.Errorf("failed to stat vector file: %w", err)
	}

	store := &MmapVectorStore{
		path: path,
		dim:  dim,
		file: file,
		size: info.Size(),
	}

	// Only mmap if file has content
	if store.size > 0 {
		reader, err := mmap.Open(path)
		if err != nil {
			file.Close()
			return nil, fmt.Errorf("failed to mmap vector file: %w", err)
		}
		store.reader = reader
	}

	return store, nil
}

func (s *MmapVectorStore) Dim() int {
	return s.dim
}

func (s *MmapVectorStore) Append(vec []float32) (int64, error) {
	if len(vec) != s.dim {
		return 0, fmt.Errorf("vector dimension mismatch: expected %d, got %d", s.dim, len(vec))
	}

	buf := make([]byte, s.dim*4)
	for i, v := range vec {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	offset := s.size
	n, err := s.file.Write(buf)
	if err != nil {
		return 0, fmt.Errorf("failed to append vector: %w", err)
	}
	if n != len(buf) {
		return 0, fmt.Errorf("short write when appending vector")
	}

	s.size += int64(n)
	return offset, nil
}

var mmapByteBufPool = sync.Pool{
	New: func() interface{} {
		b := make([]byte, 0, 1024)
		return &b
	},
}

func (s *MmapVectorStore) Read(offset int64, vec []float32) error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	reader := s.reader

	if reader == nil {
		return fmt.Errorf("cannot read from empty or unmapped store at offset %d", offset)
	}
	if len(vec) != s.dim {
		return fmt.Errorf("vec length %d does not match dim %d", len(vec), s.dim)
	}

	bufPtr := mmapByteBufPool.Get().(*[]byte)
	buf := *bufPtr
	if cap(buf) < s.dim*4 {
		buf = make([]byte, s.dim*4)
	} else {
		buf = buf[:s.dim*4]
	}
	defer func() {
		*bufPtr = buf
		mmapByteBufPool.Put(bufPtr)
	}()

	n, err := reader.ReadAt(buf, offset)
	if err != nil {
		return fmt.Errorf("failed to read vector at offset %d: %w", offset, err)
	}
	if n != len(buf) {
		return fmt.Errorf("short read at offset %d", offset)
	}

	for i := 0; i < s.dim; i++ {
		bits := binary.LittleEndian.Uint32(buf[i*4:])
		vec[i] = math.Float32frombits(bits)
	}
	return nil
}

func (s *MmapVectorStore) Remap() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.reader != nil {
		if err := s.reader.Close(); err != nil {
			return fmt.Errorf("failed to close existing mmap: %w", err)
		}
		s.reader = nil
	}

	// We don't call s.file.Sync() here because it's extremely slow on some platforms
	// and isn't required for mmap to see newly appended data.

	info, err := s.file.Stat()
	if err != nil {
		return fmt.Errorf("failed to stat file before remap: %w", err)
	}
	s.size = info.Size()

	if s.size > 0 {
		reader, err := mmap.Open(s.path)
		if err != nil {
			return fmt.Errorf("failed to remap vector file: %w", err)
		}
		s.reader = reader
	}

	return nil
}

func (s *MmapVectorStore) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	var firstErr error
	if s.reader != nil {
		if err := s.reader.Close(); err != nil {
			firstErr = fmt.Errorf("failed to close mmap: %w", err)
		}
		s.reader = nil
	}
	if s.file != nil {
		s.file.Sync()
		if err := s.file.Close(); err != nil && firstErr == nil {
			firstErr = fmt.Errorf("failed to close file: %w", err)
		}
		s.file = nil
	}
	return firstErr
}
