"""
Narameter — Narrative Shift Detection Pipeline

Deteksi Perubahan Narasi Publik Menggunakan Distribusi Embedding Teks
dan Wasserstein Distance.
"""

__version__ = "0.1.0"
__author__ = "narameter"

import os
from dotenv import load_dotenv

import warnings
# Suppress urllib3 LibreSSL warning before it can be imported by any dependencies
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")

# Auto-load variables from a .env file if it exists
load_dotenv()
