"""
ChromaDB Client - Vector Database Management

Handles connection and operations for ChromaDB vector database,
used to store clinical trial protocols and medical knowledge.
"""

import os
from typing import List, Dict, Any, Optional, Union
import chromadb


# Singleton client instance
_client: Optional[Union[chromadb.Client, chromadb.PersistentClient]] = None


def get_chromadb_client() -> Union[chromadb.Client, chromadb.PersistentClient]:
    """
    Get or create the ChromaDB client singleton.

    Uses PersistentClient for data persistence, or ephemeral Client for testing.

    Returns:
        chromadb.Client or chromadb.PersistentClient: Configured ChromaDB client
    """
    global _client

    if _client is None:
        persist_dir = os.getenv("CHROMADB_PERSIST_DIR", "./data/chromadb")
        use_persistent = os.getenv("CHROMADB_PERSISTENT", "true").lower() == "true"

        if use_persistent:
            # Create directory if it doesn't exist
            os.makedirs(persist_dir, exist_ok=True)
            _client = chromadb.PersistentClient(path=persist_dir)
        else:
            # Ephemeral client for testing
            _client = chromadb.Client()

    return _client


class ChromaDBClient:
    """
    ChromaDB client wrapper for clinical trial vector operations.

    Collections:
    - clinical_trials: Trial protocols and eligibility criteria
    - clinical_notes: Patient records (anonymized)
    - medical_knowledge: Guidelines and reference material
    """

    # Collection names
    TRIALS_COLLECTION = "clinical_trials"
    NOTES_COLLECTION = "clinical_notes"
    KNOWLEDGE_COLLECTION = "medical_knowledge"

    def __init__(self):
        self.client = get_chromadb_client()
        self._init_collections()

    def _init_collections(self):
        """Initialize all required collections."""
        # Clinical trials collection
        self.trials = self.client.get_or_create_collection(
            name=self.TRIALS_COLLECTION,
            metadata={
                "description": "Clinical trial protocols and eligibility criteria",
                "hnsw:space": "cosine"  # Cosine similarity for medical embeddings
            }
        )

        # Clinical notes collection
        self.notes = self.client.get_or_create_collection(
            name=self.NOTES_COLLECTION,
            metadata={
                "description": "Anonymized patient clinical notes",
                "hnsw:space": "cosine"
            }
        )

        # Medical knowledge collection
        self.knowledge = self.client.get_or_create_collection(
            name=self.KNOWLEDGE_COLLECTION,
            metadata={
                "description": "Medical guidelines and reference material",
                "hnsw:space": "cosine"
            }
        )

    def add_trial(
        self,
        trial_id: str,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]]
    ) -> None:
        """
        Add trial documents to the vector database.

        Args:
            trial_id: Unique trial identifier (e.g., NCT number)
            documents: List of document chunks
            embeddings: Pre-computed embeddings for each chunk
            metadatas: Metadata for each chunk (section, criterion_type, etc.)
        """
        ids = [f"{trial_id}_{i}" for i in range(len(documents))]

        self.trials.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def add_knowledge(
        self,
        source_id: str,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]]
    ) -> None:
        """
        Add medical knowledge documents to the vector database.

        Args:
            source_id: Unique source identifier
            documents: List of document chunks
            embeddings: Pre-computed embeddings
            metadatas: Metadata for each chunk
        """
        ids = [f"{source_id}_{i}" for i in range(len(documents))]

        self.knowledge.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def query_trials(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query trial documents by embedding similarity.

        Args:
            query_embedding: Query vector
            n_results: Number of results to return
            where: Optional filter conditions

        Returns:
            Query results with documents, distances, and metadata
        """
        results = self.trials.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        return results

    def query_knowledge(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query medical knowledge by embedding similarity.

        Args:
            query_embedding: Query vector
            n_results: Number of results to return
            where: Optional filter conditions

        Returns:
            Query results with documents, distances, and metadata
        """
        results = self.knowledge.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        return results

    def get_trial_by_id(self, trial_id: str) -> Dict[str, Any]:
        """
        Get all documents for a specific trial.

        Args:
            trial_id: Trial identifier (NCT number)

        Returns:
            All documents and metadata for the trial
        """
        results = self.trials.get(
            where={"trial_id": trial_id},
            include=["documents", "metadatas"]
        )

        return results

    def delete_trial(self, trial_id: str) -> None:
        """
        Delete all documents for a trial.

        Args:
            trial_id: Trial identifier to delete
        """
        # Get all IDs for this trial
        results = self.trials.get(
            where={"trial_id": trial_id}
        )

        if results["ids"]:
            self.trials.delete(ids=results["ids"])

    def get_collection_stats(self) -> Dict[str, int]:
        """
        Get statistics for all collections.

        Returns:
            Dictionary with counts for each collection
        """
        return {
            "trials": self.trials.count(),
            "notes": self.notes.count(),
            "knowledge": self.knowledge.count()
        }

    def persist(self) -> None:
        """Persist the database to disk (no-op with PersistentClient)."""
        # PersistentClient automatically persists data
        # This method is kept for backward compatibility
        pass


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def init_database() -> ChromaDBClient:
    """
    Initialize and return the database client.

    Creates collections if they don't exist.

    Returns:
        ChromaDBClient: Initialized client
    """
    client = ChromaDBClient()
    print(f"Database initialized. Stats: {client.get_collection_stats()}")
    return client


if __name__ == "__main__":
    # Test the client
    client = init_database()
    print("ChromaDB client initialized successfully!")
    print(f"Collection stats: {client.get_collection_stats()}")
