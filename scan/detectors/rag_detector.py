"""Detect RAG-related code patterns in repository files."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, List, Optional

from scan.repo_scanner import FileObject


@dataclass(frozen=True)
class RAGPipeline:
    file_path: str
    vector_store_type: Optional[str]
    embedding_model: Optional[str]
    data_source_hint: Optional[str]


VECTOR_STORE_PATTERNS = [
    {
        "type": "FAISS",
        "patterns": [
            r"\bFAISS\b",
            r"\bfaiss\.Index\b",
            r"\bfaiss\.write_index\b",
            r"\blangchain\.vectorstores\.faiss\b",
        ],
    },
    {
        "type": "Pinecone",
        "patterns": [
            r"\bPinecone\b",
            r"\bpinecone\.Index\b",
            r"\bpinecone\.init\b",
            r"\blangchain\.vectorstores\.pinecone\b",
        ],
    },
    {
        "type": "Weaviate",
        "patterns": [
            r"\bWeaviate\b",
            r"\bweaviate\.Client\b",
            r"\blangchain\.vectorstores\.weaviate\b",
        ],
    },
    {
        "type": "Chroma",
        "patterns": [
            r"\bChroma\b",
            r"\bchromadb\b",
            r"\blangchain\.vectorstores\.chroma\b",
        ],
    },
]


EMBEDDING_MODEL_PATTERNS = [
    re.compile(r"\bembedding_model\s*[:=]\s*[\"']([^\"']+)[\"']", re.IGNORECASE),
    re.compile(r"\bembedding_model_name\s*[:=]\s*[\"']([^\"']+)[\"']", re.IGNORECASE),
    re.compile(r"\bmodel\s*[:=]\s*[\"'](text-embedding[^\"']+)[\"']", re.IGNORECASE),
    re.compile(r"\bmodel_name\s*[:=]\s*[\"'](text-embedding[^\"']+)[\"']", re.IGNORECASE),
    re.compile(r"\bSentenceTransformer\(\s*[\"']([^\"']+)[\"']\s*\)"),
    re.compile(r"\bOpenAIEmbeddings\([^)]*model\s*=\s*[\"']([^\"']+)[\"']"),
    re.compile(r"\bHuggingFaceEmbeddings\([^)]*model_name\s*=\s*[\"']([^\"']+)[\"']"),
]


EMBEDDING_CLASS_PATTERNS = [
    re.compile(r"\bOpenAIEmbeddings\b"),
    re.compile(r"\bAzureOpenAIEmbeddings\b"),
    re.compile(r"\bHuggingFaceEmbeddings\b"),
    re.compile(r"\bHuggingFaceBgeEmbeddings\b"),
    re.compile(r"\bSentenceTransformer\b"),
    re.compile(r"\bCohereEmbeddings\b"),
    re.compile(r"\bBedrockEmbeddings\b"),
    re.compile(r"\bVertexAIEmbeddings\b"),
]


EMBEDDING_SIGNAL_PATTERNS = [
    re.compile(r"\bembeddings?\b", re.IGNORECASE),
    re.compile(r"\bembed_documents?\b", re.IGNORECASE),
    re.compile(r"\bembed_query\b", re.IGNORECASE),
    re.compile(r"\bembeddings\.create\b", re.IGNORECASE),
]


RETRIEVAL_PATTERNS = [
    re.compile(r"\bsimilarity_search\b", re.IGNORECASE),
    re.compile(r"\bsimilarity_search_with_score\b", re.IGNORECASE),
    re.compile(r"\bget_relevant_documents\b", re.IGNORECASE),
    re.compile(r"\bas_retriever\b", re.IGNORECASE),
    re.compile(r"\bretriever\b", re.IGNORECASE),
    re.compile(r"\bretrieve\b", re.IGNORECASE),
    re.compile(r"\bvectorstore\.search\b", re.IGNORECASE),
    re.compile(r"\bknn_search\b", re.IGNORECASE),
]


CHUNKING_PATTERNS = [
    re.compile(r"\bTextSplitter\b"),
    re.compile(r"\bRecursiveCharacterTextSplitter\b"),
    re.compile(r"\bCharacterTextSplitter\b"),
    re.compile(r"\bTokenTextSplitter\b"),
    re.compile(r"\bchunk_size\b", re.IGNORECASE),
    re.compile(r"\bchunk_overlap\b", re.IGNORECASE),
    re.compile(r"\bsplit_documents\b", re.IGNORECASE),
    re.compile(r"\bsplit_text\b", re.IGNORECASE),
]


DATA_SOURCE_PATTERNS = [
    re.compile(r"(s3://[^\s\"']+)", re.IGNORECASE),
    re.compile(r"(gs://[^\s\"']+)", re.IGNORECASE),
    re.compile(r"(https?://[^\s\"']+)", re.IGNORECASE),
    re.compile(r"[\"']([^\"']+\.(?:pdf|csv|txt|md|docx|json|html|xlsx))[\"']", re.IGNORECASE),
]

LOADER_HINTS = [
    re.compile(r"\bPDFLoader\b"),
    re.compile(r"\bCSVLoader\b"),
    re.compile(r"\bDirectoryLoader\b"),
    re.compile(r"\bWebBaseLoader\b"),
    re.compile(r"\bUnstructuredFileLoader\b"),
    re.compile(r"\bS3Loader\b"),
    re.compile(r"\bGCSFileLoader\b"),
    re.compile(r"\bGitLoader\b"),
]


def detect_rag_pipelines(file_obj: FileObject) -> List[RAGPipeline]:
    """Detect RAG-related pipelines from a FileObject."""
    vector_store_types = _find_vector_store_types(file_obj.content)
    embedding_model = _extract_embedding_model(file_obj.content)
    data_source_hint = _extract_data_source_hint(file_obj.content)

    embedding_detected = _matches_any(EMBEDDING_SIGNAL_PATTERNS, file_obj.content)
    retrieval_detected = _matches_any(RETRIEVAL_PATTERNS, file_obj.content)
    chunking_detected = _matches_any(CHUNKING_PATTERNS, file_obj.content)

    if not (vector_store_types or embedding_detected or retrieval_detected or chunking_detected):
        return []

    if not vector_store_types:
        return [
            RAGPipeline(
                file_path=file_obj.path,
                vector_store_type=None,
                embedding_model=embedding_model,
                data_source_hint=data_source_hint,
            )
        ]

    return [
        RAGPipeline(
            file_path=file_obj.path,
            vector_store_type=vector_store_type,
            embedding_model=embedding_model,
            data_source_hint=data_source_hint,
        )
        for vector_store_type in sorted(vector_store_types)
    ]


def _find_vector_store_types(content: str) -> List[str]:
    found = []
    for entry in VECTOR_STORE_PATTERNS:
        if _matches_any(_compile_patterns(entry["patterns"]), content):
            found.append(entry["type"])
    return found


def _extract_embedding_model(content: str) -> Optional[str]:
    for pattern in EMBEDDING_MODEL_PATTERNS:
        match = pattern.search(content)
        if match:
            return match.group(1).strip()

    for pattern in EMBEDDING_CLASS_PATTERNS:
        match = pattern.search(content)
        if match:
            return match.group(0)

    return None


def _extract_data_source_hint(content: str) -> Optional[str]:
    for pattern in DATA_SOURCE_PATTERNS:
        match = pattern.search(content)
        if match:
            return match.group(1)

    for pattern in LOADER_HINTS:
        match = pattern.search(content)
        if match:
            return match.group(0)

    return None


def _compile_patterns(patterns: Iterable[str]) -> List[re.Pattern]:
    return [re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in patterns]


def _matches_any(patterns: Iterable[re.Pattern], text: str) -> bool:
    return any(pattern.search(text) for pattern in patterns)
