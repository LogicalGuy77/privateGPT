#!/usr/bin/env python3
"""Test script for Private-GPT application."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from private_gpt_app.main import main

if __name__ == '__main__':
    main()
