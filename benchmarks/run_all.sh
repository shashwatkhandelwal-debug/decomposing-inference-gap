#!/usr/bin/env bash
set -e

echo "Running Full Test Suite..."
pytest tests/ -v

echo "Running Master Benchmarks Across Phases 1-7..."
python benchmarks/run_all.py
