#!/bin/bash

# Configuration
SEARCH_URL=$SEARCH_URL
MAX_ATTEMPTS=30
RETRY_DELAY=10
SAVE_PATH_RETRIEVER=/change/to/your/path # the path to save the retrieval files

# Function to check if server is responding
check_server() {
    local url=$1
    curl -s -X POST "$url" -H "Content-Type: application/json" -d '{}' > /dev/null 2>&1
    return $?
}

# Function to wait for server to be ready with retries
wait_for_server() {
    local url=$1
    local attempt=1
    
    echo "Waiting for server at $url to be ready..."
    
    while [ $attempt -le $MAX_ATTEMPTS ]; do
        if check_server "$url"; then
            echo "Server is ready!"
            return 0
        fi
        
        echo "Attempt $attempt/$MAX_ATTEMPTS: Server not ready, waiting ${RETRY_DELAY} seconds..."
        sleep $RETRY_DELAY
        ((attempt++))
    done
    
    echo "Error: Server failed to start after $MAX_ATTEMPTS attempts"
    return 1
}

# Function to cleanup server process
cleanup_server() {
    local pid=$1
    if [ -n "$pid" ]; then
        echo "Cleaning up server process (PID: $pid)..."
        kill $pid 2>/dev/null
        wait $pid 2>/dev/null
    fi
}

# Main execution
echo "=== Starting Local E5 Server ==="
echo "Starting local E5 server..."

# Server configuration
index_file=$SAVE_PATH_RETRIEVER/e5_HNSW64.index
corpus_file=$SAVE_PATH_RETRIEVER/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2
num_workers=1

export MOSEC_TIMEOUT=10000
python gem/tools/search_engine/retrieval_server.py  --index_path $index_file \
                                                    --corpus_path $corpus_file \
                                                    --topk 3 \
                                                    --retriever_name $retriever_name \
                                                    --retriever_model $retriever_path \
                                                    --num_workers $num_workers &

server_pid=$!
echo "Server started with PID: $server_pid"

# Wait for server to be ready
if wait_for_server "$SEARCH_URL"; then
    echo "=== Server is ready and running ==="
    exit 0
else
    echo "=== Failed to start server ==="
    cleanup_server $server_pid
    exit 1
fi
