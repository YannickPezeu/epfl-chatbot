#!/bin/bash

# Create a log directory if it doesn't exist
mkdir -p /var/log/app

# Start formatting service in background and log to both file and stdout
python runai_code/formatting_service.py 2>&1 | tee /var/log/app/formatting.log &
formatting_pid=$!

# Start LLM service and log to both file and stdout
python runai_code/llm_service_titanX.py 2>&1 | tee /var/log/app/llm.log &
llm_pid=$!

# Function to check if LLM service is ready
check_llm_service() {
    # Check the log file for a specific message indicating service is ready
    grep -q "Model loaded successfully" /var/log/app/llm.log 2>/dev/null || \
    grep -q "Server started" /var/log/app/llm.log 2>/dev/null
}

echo "Waiting for LLM service to start..."
start_time=$(date +%s)
timeout=900  # 15 minutes timeout

while ! check_llm_service; do
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))

    # Check if services are still running
    if ! kill -0 $llm_pid 2>/dev/null; then
        echo "LLM service crashed. Check logs at /var/log/app/llm.log" | tee -a /var/log/app/error.log
        exit 1
    fi

    # Check timeout
    if [ $elapsed -gt $timeout ]; then
        echo "Timeout waiting for LLM service to start" | tee -a /var/log/app/error.log
        echo "Last few lines of LLM log:" | tee -a /var/log/app/error.log
        tail -n 50 /var/log/app/llm.log | tee -a /var/log/app/error.log
        exit 1
    fi

    sleep 5
    echo "Still waiting for LLM service... ($elapsed seconds elapsed)" | tee -a /var/log/app/status.log
done

echo "LLM service is ready, starting main.py" | tee -a /var/log/app/status.log

# Start main application and log to both file and stdout
python main.py 2>&1 | tee /var/log/app/main.log