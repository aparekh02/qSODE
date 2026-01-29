#!/usr/bin/env python3
"""
Quick Start Script
==================

Run this script to execute the urban wave simulation example.

Usage:
    python run_example.py

This will:
1. Create an urban environment
2. Run the qODE simulation
3. Generate visualizations and video
4. Save results to the examples folder
"""

import sys
import os

# Ensure the package is importable
sys.path.insert(0, os.path.dirname(__file__))

# Run the example
from qode_framework.examples.urban_wave_simulation import main

if __name__ == "__main__":
    main()
