#!/bin/bash

"""
This file is used to run evaluation after training some 
model immediately after the trainig code exits.
Note: you must ensure the inference config weights_path
aligns with the model you are training.
"""

# Process IDs to wait for
PID1=437544
PID2=437545

# Function to check if a process is running
is_running() {
    kill -0 "$1" 2>/dev/null
}

echo "Waiting for processes $PID1 and $PID2 to exit..."

# Wait for both processes to finish
while is_running "$PID1" || is_running "$PID2"; do
    sleep 60
done

echo "Both processes have exited."
sleep 3

echo "Running Inference."

python3 main.py