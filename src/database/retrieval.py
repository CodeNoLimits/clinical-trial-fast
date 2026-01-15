"""
Hybrid Retrieval Module - RAG Implementation

Implements hybrid search combining:
- BM25 (lexical search) for exact medical terminology
- Dense retrieval (embeddings) for semantic similarity
- RRF (Reciprocal Rank Fusion) for result merging
- Optional re-ranking for improved precision

Follows best practices from medical RAG research (2025).
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from rank_bm25 import BM25Okapi

from .chromadb_client import ChromaDBClient
from .embeddings import EmbeddingManager


@dataclass
class RetrievalResult:
    """Single retrieval result with metadata."""
    document: str
    score: float
    source: str  # 'bm25', 'dense', or 'hybrid'
    metadata: Dict[str, Any]


class HybridRetriever:
    """
    Hybrid retrieval system combining BM25 and dense search.

    Uses Reciprocal Rank Fusion (RRF) to merge results from
    both retrieval methods for optimal recall and precision.

    Architecture follows medical RAG best practices:
    - BM25 for exact term matching (medical codes, drug names)
    - Dense for semantic similarity (symptom descriptions)
    - RRF fusion with configurable weights
    """

    # RRF constant (higher = more weight to lower ranks)
    RRF_K = 60

    def __init__(
        self,
        db_client: Optional[ChromaDBClient] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
        bm25_weight: float = 0.4,
        dense_weight: float = 0.6
    ):
        """
        Initialize the hybrid retriever.

        Args:
            db_client: ChromaDB client (created if not provided)
            embedding_manager: Embedding manager (created if not provided)
            bm25_weight: Weight for BM25 results (0-1)
            dense_weight: Weight for dense results (0-1)
        """
        self.db = db_client or ChromaDBClient()
        self.embeddings = embedding_manager or EmbeddingManager()
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

        # BM25 index (built on first query or explicit call)
        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_documents: List[str] = []
        self._bm25_metadata: List[Dict[str, Any]] = []

    def build_bm25_index(self, collection: str = "clinical_trials") -> None:
        """
        Build BM25 index from ChromaDB collection.

        Args:
            collection: Collection name to index
        """
        # Get all documents from collection
        if collection == "clinical_trials":
            results = self.db.trials.get(include=["documents", "metadatas"])
        elif collection == "medical_knowledge":
            results = self.db.knowledge.get(include=["documents", "metadatas"])
        else:
            raise ValueError(f"Unknown collection: {collection}")

        if not results["documents"]:
            print(f"Warning: No documents in {collection} collection")
            return

        self._bm25_documents = results["documents"]
        self._bm25_metadata = results["metadatas"] or [{}] * len(results["documents"])

        # Tokenize documents for BM25
        tokenized = [doc.lower().split() for doc in self._bm25_documents]
        self._bm25_index = BM25Okapi(tokenized)

        print(f"Built BM25 index with {len(self._bm25_documents)} documents")

    def search_bm25(
        self,
        query: str,
        top_k: int = 10
    ) -> List[RetrievalResult]:
        """
        Search using BM25 (lexical search).

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of retrieval results
        """
        if self._bm25_index is None:
            self.build_bm25_index()

        if self._bm25_index is None or len(self._bm25_documents) == 0:
            return []

        # Tokenize query
        tokenized_query = query.lower().split()

        # Get BM25 scores
        scores = self._bm25_index.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include positive scores
                results.append(RetrievalResult(
                    document=self._bm25_documents[idx],
                    score=float(scores[idx]),
                    source="bm25",
                    metadata=self._bm25_metadata[idx] if idx < len(self._bm25_metadata) else {}
                ))

        return results

    def search_dense(
        self,
        query: str,
        top_k: int = 10,
        collection: str = "clinical_trials"
    ) -> List[RetrievalResult]:
        """
        Search using dense embeddings.

        Args:
            query: Search query
            top_k: Number of results to return
            collection: Collection to search

        Returns:
            List of retrieval results
        """
        # Generate query embedding
        query_embedding = self.embeddings.embed_text(query)

        # Query ChromaDB
        if collection == "clinical_trials":
            results = self.db.query_trials(query_embedding, n_results=top_k)
        else:
            results = self.db.query_knowledge(query_embedding, n_results=top_k)

        retrieval_results = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0
                # Convert distance to similarity score
                similarity = 1 - distance  # For cosine distance

                metadata = results["metadatas"][0][i] if results["metadatas"] else {}

                retrieval_results.append(RetrievalResult(
                    document=doc,
                    score=float(similarity),
                    source="dense",
                    metadata=metadata
                ))

        return retrieval_results

    def search_hybrid(
        self,
        query: str,
        top_k: int = 10,
        collection: str = "clinical_trials"
    ) -> List[RetrievalResult]:
        """
        Hybrid search combining BM25 and dense retrieval with RRF fusion.

        Args:
            query: Search query
            top_k: Number of results to return
            collection: Collection to search

        Returns:
            Merged and re-ranked results
        """
        # Get results from both methods
        bm25_results = self.search_bm25(query, top_k=top_k * 2)
        dense_results = self.search_dense(query, top_k=top_k * 2, collection=collection)

        # Apply RRF fusion
        fused_results = self._rrf_fusion(bm25_results, dense_results)

        return fused_results[:top_k]

    def _rrf_fusion(
        self,
        bm25_results: List[RetrievalResult],
        dense_results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Apply Reciprocal Rank Fusion to merge result lists.

        RRF score = sum(1 / (k + rank)) for each list where document appears

        Args:
            bm25_results: Results from BM25
            dense_results: Results from dense search

        Returns:
            Merged results sorted by RRF score
        """
        # Create document -> score mapping
        doc_scores: Dict[str, Tuple[float, Dict[str, Any]]] = {}

        # Process BM25 results
        for rank, result in enumerate(bm25_results):
            doc_key = result.document[:100]  # Use first 100 chars as key
            rrf_score = self.bm25_weight / (self.RRF_K + rank + 1)

            if doc_key in doc_scores:
                current_score, metadata = doc_scores[doc_key]
                doc_scores[doc_key] = (current_score + rrf_score, metadata)
            else:
                doc_scores[doc_key] = (rrf_score, result.metadata)

        # Process dense results
        for rank, result in enumerate(dense_results):
            doc_key = result.document[:100]
            rrf_score = self.dense_weight / (self.RRF_K + rank + 1)

            if doc_key in doc_scores:
                current_score, metadata = doc_scores[doc_key]
                doc_scores[doc_key] = (current_score + rrf_score, metadata)
            else:
                doc_scores[doc_key] = (rrf_score, result.metadata)

        # Create final results
        # Need to map back to full documents
        all_docs = {r.document[:100]: r.document for r in bm25_results + dense_results}

        fused_results = []
        for doc_key, (score, metadata) in doc_scores.items():
            full_doc = all_docs.get(doc_key, doc_key)
            fused_results.append(RetrievalResult(
                document=full_doc,
                score=score,
                source="hybrid",
                metadata=metadata
            ))

        # Sort by score descending
        fused_results.sort(key=lambda x: x.score, reverse=True)

        return fused_results

    async def search(
        self,
        query: str,
        top_k: int = 5,
        method: str = "hybrid"
    ) -> List[Dict[str, Any]]:
        """
        Async search interface for agent integration.

        Args:
            query: Search query
            top_k: Number of results
            method: "bm25", "dense", or "hybrid"

        Returns:
            List of result dictionaries
        """
        if method == "bm25":
            results = self.search_bm25(query, top_k)
        elif method == "dense":
            results = self.search_dense(query, top_k)
        else:
            results = self.search_hybrid(query, top_k)

        return [
            {
                "document": r.document,
                "score": r.score,
                "source": r.source,
                "metadata": r.metadata
            }
            for r in results
        ]


class ReRanker:
    """
    Cross-encoder re-ranker for improved precision.

    Optional component that re-ranks top results from hybrid search
    using a cross-encoder model for more accurate relevance scoring.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the re-ranker.

        Args:
            model_name: Cross-encoder model to use
        """
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
            self.available = True
        except ImportError:
            print("Warning: CrossEncoder not available, re-ranking disabled")
            self.available = False

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Re-rank results using cross-encoder.

        Args:
            query: Original query
            results: Results to re-rank
            top_k: Number of results to return

        Returns:
            Re-ranked results
        """
        if not self.available or not results:
            return results[:top_k]

        # Prepare pairs for cross-encoder
        pairs = [(query, r.document) for r in results]

        # Get cross-encoder scores
        scores = self.model.predict(pairs)

        # Update scores and sort
        for i, result in enumerate(results):
            result.score = float(scores[i])

        results.sort(key=lambda x: x.score, reverse=True)

        return results[:top_k]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def test_retrieval():
    """Test the hybrid retrieval system."""
    retriever = HybridRetriever()

    # Add some test documents first
    from .embeddings import create_trial_embeddings

    test_protocol = """
    INCLUSION CRITERIA:
    1. Age 18-75 years
    2. Diagnosis of Type 2 Diabetes
    3. HbA1c 7.0-10.0%

    EXCLUSION CRITERIA:
    1. Type 1 Diabetes
    2. Pregnancy
    """

    docs, embeddings, metadatas = create_trial_embeddings(test_protocol, "TEST001")
    retriever.db.add_trial("TEST001", docs, embeddings, metadatas)

    # Test search
    query = "diabetes HbA1c criteria"
    results = retriever.search_hybrid(query, top_k=3)

    print(f"\nQuery: {query}")
    print(f"Results: {len(results)}")
    for r in results:
        print(f"  - Score: {r.score:.4f} | Source: {r.source}")
        print(f"    Text: {r.document[:100]}...")


if __name__ == "__main__":
    test_retrieval()
