$ErrorActionPreference = "Stop"

Write-Host "Running Full Test Suite..." -ForegroundColor Cyan
pytest tests/ -v

Write-Host "Running Master Benchmarks Across Phases 1-7..." -ForegroundColor Cyan
python benchmarks/run_all.py
