package storage

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"encoding/gob"
	"io"
	"os"
	"sync"
)

type WALEntry struct {
	Op         string // "insert" | "delete"
	Collection string // collection name
	VectorID   string
	Vector     []float32 // nil for delete ops
	Timestamp  int64     // UnixNano
}

type WAL struct {
	f  *os.File
	w  *bufio.Writer
	mu sync.Mutex
}

func OpenWAL(path string) (*WAL, error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_RDWR, 0644)
	if err != nil {
		return nil, err
	}
	w := bufio.NewWriter(f)
	return &WAL{
		f: f,
		w: w,
	}, nil
}

func (w *WAL) Append(entry WALEntry) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	if err := enc.Encode(entry); err != nil {
		return err
	}

	data := buf.Bytes()
	length := uint32(len(data))

	if err := binary.Write(w.w, binary.BigEndian, length); err != nil {
		return err
	}
	if _, err := w.w.Write(data); err != nil {
		return err
	}

	return w.w.Flush()
}

func (w *WAL) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if err := w.w.Flush(); err != nil {
		w.f.Close()
		return err
	}
	return w.f.Close()
}

func ReadWALEntries(path string) ([]WALEntry, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()

	var entries []WALEntry
	r := bufio.NewReader(f)

	for {
		var length uint32
		if err := binary.Read(r, binary.BigEndian, &length); err != nil {
			if err == io.EOF {
				break
			}
			// Unexpected EOF or partial write, stop reading but return what we have
			break
		}

		data := make([]byte, length)
		if _, err := io.ReadFull(r, data); err != nil {
			// Partial write, stop reading
			break
		}

		var entry WALEntry
		dec := gob.NewDecoder(bytes.NewReader(data))
		if err := dec.Decode(&entry); err != nil {
			// Corrupt entry, stop reading
			break
		}

		entries = append(entries, entry)
	}

	return entries, nil
}

func (w *WAL) Truncate() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if err := w.w.Flush(); err != nil {
		return err
	}

	// Close the file first to avoid Windows access denied errors on Truncate
	w.f.Close()

	path := w.f.Name()

	// Truncate it
	if err := os.Truncate(path, 0); err != nil {
		return err
	}

	// Reopen it
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_RDWR, 0644)
	if err != nil {
		return err
	}

	w.f = f
	w.w.Reset(w.f)
	return nil
}
