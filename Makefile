.PHONY: build test run bench

build:
	go build ./...

test:
	go test ./...

run:
	go run ./cmd/govectordb

bench:
	go test -bench=. -benchmem ./...
