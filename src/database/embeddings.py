"""
Embeddings Module - Medical Text Embedding Management

Handles embedding generation using medical-specific models:
- MedEmbed (recommended for production)
- BioBERT
- sentence-transformers (fallback)

Implements chunking strategies optimized for medical documents.
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer


# Singleton embedding model
_embedding_model: Optional[SentenceTransformer] = None


def get_embeddings() -> SentenceTransformer:
    """
    Get or create the embedding model singleton.

    Uses the model specified in EMBEDDING_MODEL env var,
    defaults to MiniLM for fast development.

    Returns:
        SentenceTransformer: Loaded embedding model
    """
    global _embedding_model

    if _embedding_model is None:
        model_name = os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        _embedding_model = SentenceTransformer(model_name)
        print(f"Loaded embedding model: {model_name}")

    return _embedding_model


class EmbeddingManager:
    """
    Manages text embedding generation and medical document chunking.

    Implements CLEAR-style entity-aware chunking for clinical documents
    and semantic chunking for protocol documents.
    """

    # Chunking parameters
    DEFAULT_CHUNK_SIZE = 512  # tokens
    DEFAULT_OVERLAP = 50  # tokens
    MAX_CHUNK_SIZE = 1024

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedding manager.

        Args:
            model_name: Optional model name override
        """
        if model_name:
            self.model = SentenceTransformer(model_name)
        else:
            self.model = get_embeddings()

        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector as list of floats
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts
            batch_size: Batch size for encoding

        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        return embeddings.tolist()

    def chunk_protocol(
        self,
        protocol_text: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP
    ) -> List[Dict[str, Any]]:
        """
        Chunk a clinical trial protocol document.

        Uses section-aware chunking that preserves:
        - Inclusion criteria as separate chunks
        - Exclusion criteria as separate chunks
        - Study design information
        - Safety information

        Args:
            protocol_text: Full protocol document text
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks

        Returns:
            List of chunks with metadata
        """
        chunks = []

        # Split by major sections
        sections = self._split_by_sections(protocol_text)

        for section_name, section_text in sections.items():
            # Further split large sections
            if len(section_text) > chunk_size:
                sub_chunks = self._split_text(section_text, chunk_size, overlap)
                for i, chunk in enumerate(sub_chunks):
                    chunks.append({
                        "text": chunk,
                        "section": section_name,
                        "chunk_index": i,
                        "total_chunks": len(sub_chunks)
                    })
            else:
                chunks.append({
                    "text": section_text,
                    "section": section_name,
                    "chunk_index": 0,
                    "total_chunks": 1
                })

        return chunks

    def _split_by_sections(self, text: str) -> Dict[str, str]:
        """
        Split protocol text by major sections.

        Args:
            text: Protocol text

        Returns:
            Dictionary mapping section names to content
        """
        sections = {}

        # Common section headers in clinical trial protocols
        section_markers = [
            ("INCLUSION CRITERIA", "inclusion_criteria"),
            ("EXCLUSION CRITERIA", "exclusion_criteria"),
            ("STUDY DESIGN", "study_design"),
            ("OBJECTIVES", "objectives"),
            ("ENDPOINTS", "endpoints"),
            ("SAFETY", "safety"),
            ("PROCEDURES", "procedures"),
        ]

        current_section = "general"
        current_text = []

        lines = text.split("\n")
        for line in lines:
            line_upper = line.upper().strip()

            # Check if this line starts a new section
            new_section = None
            for marker, section_name in section_markers:
                if marker in line_upper:
                    new_section = section_name
                    break

            if new_section:
                # Save previous section
                if current_text:
                    sections[current_section] = "\n".join(current_text)
                current_section = new_section
                current_text = [line]
            else:
                current_text.append(line)

        # Save last section
        if current_text:
            sections[current_section] = "\n".join(current_text)

        return sections

    def _split_text(
        self,
        text: str,
        chunk_size: int,
        overlap: int
    ) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to split
            chunk_size: Target chunk size
            overlap: Overlap size

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for period, question mark, or exclamation
                for punct in [". ", "? ", "! "]:
                    punct_pos = text.rfind(punct, start, end)
                    if punct_pos > start:
                        end = punct_pos + 1
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap

        return chunks

    def chunk_eligibility_criteria(
        self,
        criteria_text: str,
        criteria_type: str = "inclusion"
    ) -> List[Dict[str, Any]]:
        """
        Chunk eligibility criteria, one criterion per chunk.

        Args:
            criteria_text: Text containing eligibility criteria
            criteria_type: "inclusion" or "exclusion"

        Returns:
            List of individual criteria with metadata
        """
        chunks = []

        # Split by numbered items or bullet points
        import re
        pattern = r'(?:^|\n)\s*(?:\d+[\.\)]\s*|[-•]\s*)'
        items = re.split(pattern, criteria_text)

        for i, item in enumerate(items):
            item = item.strip()
            if item and len(item) > 10:  # Skip empty or very short items
                chunks.append({
                    "text": item,
                    "section": f"{criteria_type}_criteria",
                    "criterion_number": i + 1,
                    "criterion_type": criteria_type
                })

        return chunks

    def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0-1)
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_trial_embeddings(
    protocol_text: str,
    trial_id: str
) -> tuple[List[str], List[List[float]], List[Dict[str, Any]]]:
    """
    Create embeddings for a clinical trial protocol.

    Convenience function that chunks the protocol and generates embeddings.

    Args:
        protocol_text: Full protocol document
        trial_id: Trial identifier

    Returns:
        Tuple of (documents, embeddings, metadatas)
    """
    manager = EmbeddingManager()

    # Chunk the protocol
    chunks = manager.chunk_protocol(protocol_text)

    # Extract texts and prepare metadata
    documents = [c["text"] for c in chunks]
    metadatas = [{
        "trial_id": trial_id,
        "section": c["section"],
        "chunk_index": c["chunk_index"]
    } for c in chunks]

    # Generate embeddings
    embeddings = manager.embed_texts(documents)

    return documents, embeddings, metadatas


if __name__ == "__main__":
    # Test the embedding manager
    manager = EmbeddingManager()
    print(f"Embedding dimension: {manager.embedding_dim}")

    test_text = "Type 2 Diabetes Mellitus with HbA1c between 7.0% and 10.0%"
    embedding = manager.embed_text(test_text)
    print(f"Test embedding shape: {len(embedding)}")
