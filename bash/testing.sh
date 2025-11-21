#!/bin/bash
# A simple script to test basic bash functionality
echo "Starting bash functionality tests..."

pytest tests --cov=src --cov-report=term-missing
