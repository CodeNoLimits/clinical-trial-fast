"""
Database Module - Vector Database & RAG Implementation

This module handles:
- ChromaDB vector database management
- Medical embeddings (MedEmbed/BioBERT)
- Hybrid search (BM25 + Dense + RRF Fusion)
- Document ingestion for trial protocols
"""

from .chromadb_client import ChromaDBClient, get_chromadb_client
from .embeddings import EmbeddingManager, get_embeddings
from .retrieval import HybridRetriever
from .ingest_trials import TrialIngester

__all__ = [
    "ChromaDBClient",
    "get_chromadb_client",
    "EmbeddingManager",
    "get_embeddings",
    "HybridRetriever",
    "TrialIngester",
]
