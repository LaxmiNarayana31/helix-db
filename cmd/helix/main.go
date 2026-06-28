package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"path/filepath"

	"helix/internal/api"
	"helix/internal/shard"
	"helix/internal/vectorstore"
)

func main() {
	serveCmd := flag.NewFlagSet("serve", flag.ExitOnError)
	port := serveCmd.String("port", "8080", "Port to serve on")
	dbPath := serveCmd.String("db", "helix.db", "Path to SQLite database file")
	shards := serveCmd.Int("shards", 1, "Number of shards (1 for non-sharded)")

	if len(os.Args) < 2 {
		fmt.Println("Expected 'serve' subcommand")
		fmt.Println("Usage: helix serve [--port 8080] [--db helix.db]")
		os.Exit(1)
	}

	switch os.Args[1] {
	case "serve":
		serveCmd.Parse(os.Args[2:])
		runServer(*dbPath, *port, *shards)
	default:
		fmt.Println("Expected 'serve' subcommand")
		fmt.Println("Usage: helix serve [--port 8080] [--db helix.db]")
		os.Exit(1)
	}
}

func runServer(dbPath, port string, shards int) {
	var store api.VectorDB
	var closeFunc func() error

	if shards > 1 {
		baseDir := dbPath
		if filepath.Ext(dbPath) == ".db" {
			baseDir = filepath.Dir(dbPath)
		}
		router, routerErr := shard.NewShardRouter(shards, baseDir)
		if routerErr != nil {
			log.Fatalf("Failed to initialize shard router: %v", routerErr)
		}
		store = router
		closeFunc = router.Close
	} else {
		singleStore, storeErr := vectorstore.NewStore(dbPath)
		if storeErr != nil {
			log.Fatalf("Failed to initialize store: %v", storeErr)
		}
		store = singleStore
		closeFunc = singleStore.Close
	}

	defer func() {
		fmt.Println("\nShutdown signal received. Saving index snapshots...")
		if err := closeFunc(); err != nil {
			log.Fatalf("Error during shutdown: %v", err)
		}
		fmt.Println("Shutdown complete.")
	}()

	// Initialize API server
	apiServer := api.NewServer(store)
	addr := ":" + port

	srv := &http.Server{
		Addr:    addr,
		Handler: apiServer,
	}

	go func() {
		fmt.Printf("Helix started. Using database: %s\n", dbPath)
		fmt.Printf("Listening on %s\n", addr)
		fmt.Println("Press Ctrl+C to gracefully shutdown and save snapshots...")
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Server error: %v", err)
		}
	}()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
	<-sigCh

	// Give the server a few seconds to drain requests
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	srv.Shutdown(ctx)
}
