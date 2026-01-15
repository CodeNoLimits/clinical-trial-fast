"""
Trial Ingestion Module - Protocol Document Processing

Handles the ingestion of clinical trial protocols into the vector database.
Supports multiple formats: Markdown, PDF, JSON.

Features:
- Section-aware chunking for protocols
- Metadata extraction (NCT number, phase, condition)
- Batch processing for multiple trials
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from .chromadb_client import ChromaDBClient
from .embeddings import EmbeddingManager


class TrialIngester:
    """
    Ingests clinical trial protocols into the vector database.

    Supports protocols from:
    - Local markdown files (Meléa's format)
    - ClinicalTrials.gov JSON
    - FHIR ResearchStudy format
    """

    def __init__(
        self,
        db_client: Optional[ChromaDBClient] = None,
        embedding_manager: Optional[EmbeddingManager] = None
    ):
        """
        Initialize the trial ingester.

        Args:
            db_client: ChromaDB client
            embedding_manager: Embedding manager
        """
        self.db = db_client or ChromaDBClient()
        self.embeddings = embedding_manager or EmbeddingManager()

    def ingest_markdown_file(
        self,
        file_path: str,
        trial_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ingest a clinical trial protocol from a markdown file.

        Args:
            file_path: Path to the markdown file
            trial_id: Optional trial ID (extracted from filename if not provided)

        Returns:
            Ingestion result with document count and trial ID
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Extract trial ID from filename if not provided
        if trial_id is None:
            trial_id = path.stem  # Filename without extension

        # Read the file
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract metadata from content
        metadata = self._extract_metadata_from_markdown(content, trial_id)

        # Chunk the protocol
        chunks = self.embeddings.chunk_protocol(content)

        # Prepare documents and metadata
        documents = [c["text"] for c in chunks]
        metadatas = []
        for c in chunks:
            chunk_meta = {
                "trial_id": trial_id,
                "section": c["section"],
                "chunk_index": c["chunk_index"],
                "source_file": str(path.name),
                "ingested_at": datetime.now().isoformat(),
                **metadata
            }
            metadatas.append(chunk_meta)

        # Generate embeddings
        embeddings = self.embeddings.embed_texts(documents)

        # Store in database
        self.db.add_trial(trial_id, documents, embeddings, metadatas)

        return {
            "trial_id": trial_id,
            "documents_ingested": len(documents),
            "source_file": str(path.name),
            "metadata": metadata
        }

    def _extract_metadata_from_markdown(
        self,
        content: str,
        trial_id: str
    ) -> Dict[str, Any]:
        """
        Extract metadata from markdown content.

        Args:
            content: Markdown content
            trial_id: Trial ID

        Returns:
            Extracted metadata
        """
        metadata = {
            "trial_id": trial_id,
            "format": "markdown"
        }

        # Try to extract NCT number
        import re
        nct_match = re.search(r'NCT\d{8}', content)
        if nct_match:
            metadata["nct_number"] = nct_match.group()

        # Try to extract title
        lines = content.split("\n")
        for line in lines[:10]:
            if line.startswith("# "):
                metadata["title"] = line[2:].strip()
                break
            elif line.startswith("CLINICAL TRIAL:"):
                metadata["title"] = line.replace("CLINICAL TRIAL:", "").strip()
                break

        # Try to extract condition
        condition_patterns = [
            r"(?:condition|disease|indication):\s*(.+)",
            r"(?:type \d+ diabetes|diabetes mellitus|hypertension|cancer)",
        ]
        for pattern in condition_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                metadata["condition"] = match.group(1) if match.lastindex else match.group()
                break

        return metadata

    def ingest_json_trial(
        self,
        json_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Ingest a clinical trial from JSON format (ClinicalTrials.gov style).

        Args:
            json_data: Trial data in JSON format

        Returns:
            Ingestion result
        """
        # Extract trial ID
        trial_id = json_data.get("nct_id") or json_data.get("trial_id") or "UNKNOWN"

        # Build protocol text from JSON
        protocol_text = self._json_to_protocol_text(json_data)

        # Extract metadata
        metadata = {
            "trial_id": trial_id,
            "format": "json",
            "title": json_data.get("title", ""),
            "condition": json_data.get("condition", ""),
            "phase": json_data.get("phase", ""),
            "status": json_data.get("status", ""),
        }

        # Chunk the protocol
        chunks = self.embeddings.chunk_protocol(protocol_text)

        # Prepare documents and metadata
        documents = [c["text"] for c in chunks]
        metadatas = []
        for c in chunks:
            chunk_meta = {
                **metadata,
                "section": c["section"],
                "chunk_index": c["chunk_index"],
                "ingested_at": datetime.now().isoformat(),
            }
            metadatas.append(chunk_meta)

        # Generate embeddings
        embeddings = self.embeddings.embed_texts(documents)

        # Store in database
        self.db.add_trial(trial_id, documents, embeddings, metadatas)

        return {
            "trial_id": trial_id,
            "documents_ingested": len(documents),
            "metadata": metadata
        }

    def _json_to_protocol_text(self, json_data: Dict[str, Any]) -> str:
        """
        Convert JSON trial data to protocol text format.

        Args:
            json_data: Trial data in JSON

        Returns:
            Protocol text
        """
        sections = []

        # Title
        if json_data.get("title"):
            sections.append(f"# {json_data['title']}")

        # Trial ID
        if json_data.get("nct_id"):
            sections.append(f"NCT Number: {json_data['nct_id']}")

        # Condition
        if json_data.get("condition"):
            sections.append(f"Condition: {json_data['condition']}")

        # Description
        if json_data.get("description"):
            sections.append(f"\n## STUDY DESCRIPTION\n{json_data['description']}")

        # Eligibility
        if json_data.get("eligibility"):
            elig = json_data["eligibility"]

            if elig.get("criteria"):
                sections.append(f"\n## ELIGIBILITY CRITERIA\n{elig['criteria']}")

            if elig.get("inclusion_criteria"):
                sections.append("\n## INCLUSION CRITERIA")
                for i, criterion in enumerate(elig["inclusion_criteria"], 1):
                    sections.append(f"{i}. {criterion}")

            if elig.get("exclusion_criteria"):
                sections.append("\n## EXCLUSION CRITERIA")
                for i, criterion in enumerate(elig["exclusion_criteria"], 1):
                    sections.append(f"{i}. {criterion}")

            # Age
            if elig.get("minimum_age"):
                sections.append(f"Minimum Age: {elig['minimum_age']}")
            if elig.get("maximum_age"):
                sections.append(f"Maximum Age: {elig['maximum_age']}")

        return "\n".join(sections)

    def ingest_directory(
        self,
        directory_path: str,
        pattern: str = "*.md"
    ) -> List[Dict[str, Any]]:
        """
        Ingest all trial files from a directory.

        Args:
            directory_path: Path to directory containing trial files
            pattern: File pattern to match (default: *.md)

        Returns:
            List of ingestion results
        """
        path = Path(directory_path)

        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        results = []
        files = list(path.glob(pattern))

        print(f"Found {len(files)} files to ingest")

        for file_path in files:
            try:
                result = self.ingest_markdown_file(str(file_path))
                results.append(result)
                print(f"  ✓ Ingested: {result['trial_id']} ({result['documents_ingested']} chunks)")
            except Exception as e:
                print(f"  ✗ Error ingesting {file_path}: {str(e)}")
                results.append({
                    "trial_id": file_path.stem,
                    "error": str(e)
                })

        return results

    def ingest_melea_trials(self) -> List[Dict[str, Any]]:
        """
        Ingest trials from Meléa's original project structure.

        Looks for trial files in data/trials/ directory.

        Returns:
            List of ingestion results
        """
        trials_dir = Path(__file__).parent.parent.parent / "data" / "trials"

        if not trials_dir.exists():
            print(f"Trials directory not found: {trials_dir}")
            return []

        return self.ingest_directory(str(trials_dir))


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """CLI entry point for trial ingestion."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest clinical trial protocols into vector database"
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        help="Single file to ingest"
    )
    parser.add_argument(
        "--directory",
        "-d",
        type=str,
        help="Directory to ingest all trials from"
    )
    parser.add_argument(
        "--pattern",
        "-p",
        type=str,
        default="*.md",
        help="File pattern for directory ingestion (default: *.md)"
    )
    parser.add_argument(
        "--melea",
        action="store_true",
        help="Ingest Meléa's trial files from data/trials/"
    )

    args = parser.parse_args()

    ingester = TrialIngester()

    if args.file:
        result = ingester.ingest_markdown_file(args.file)
        print(f"Ingested trial: {result['trial_id']}")
        print(f"Documents: {result['documents_ingested']}")

    elif args.directory:
        results = ingester.ingest_directory(args.directory, args.pattern)
        print(f"\nIngested {len(results)} trials")

    elif args.melea:
        results = ingester.ingest_melea_trials()
        print(f"\nIngested {len(results)} Meléa trials")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
