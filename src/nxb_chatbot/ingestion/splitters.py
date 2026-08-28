import re
import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from nxb_chatbot.core.config import settings

logger = logging.getLogger(__name__)

# A valid markdown table must have a separator row like |---|---|
TABLE_SEPARATOR_RE = re.compile(r"\|[\s]*:?-{2,}:?[\s]*\|")


def _is_table_block(text: str) -> bool:
    """Check if a block contains a valid markdown table."""
    return bool(TABLE_SEPARATOR_RE.search(text))


def _extract_segments(text: str) -> list[dict]:
    """
    Walk through page content line by line and split into
    alternating text and table segments.

    Returns:
        list of {"content": str, "is_table": bool}
    """
    segments = []
    current_lines: list[str] = []
    in_table = False

    for line in text.split("\n"):
        line_has_pipe = "|" in line

        if line_has_pipe and not in_table:
            # Flush accumulated text before entering table
            if current_lines:
                content = "\n".join(current_lines).strip()
                if content:
                    segments.append({"content": content, "is_table": False})
                current_lines = []
            in_table = True
            current_lines.append(line)

        elif not line_has_pipe and in_table:
            # Flush accumulated table block
            content = "\n".join(current_lines).strip()
            if content:
                segments.append(
                    {
                        "content": content,
                        "is_table": _is_table_block(content),
                    }
                )
            current_lines = []
            in_table = False
            current_lines.append(line)

        else:
            current_lines.append(line)

    # Flush whatever remains
    if current_lines:
        content = "\n".join(current_lines).strip()
        if content:
            segments.append(
                {
                    "content": content,
                    "is_table": in_table and _is_table_block(content),
                }
            )

    return segments


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Split a list of Documents into chunks:
    - Table blocks → single atomic chunk (never split), has_table=True
    - Text blocks  → RecursiveCharacterTextSplitter, has_table=False

    Original metadata is preserved on every chunk.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    all_chunks: list[Document] = []

    for doc in documents:
        segments = _extract_segments(doc.page_content)

        for segment in segments:
            base_metadata = {**doc.metadata, "has_table": segment["is_table"]}

            if segment["is_table"]:
                all_chunks.append(
                    Document(
                        page_content=segment["content"],
                        metadata=base_metadata,
                    )
                )
            else:
                chunks = text_splitter.create_documents(
                    texts=[segment["content"]],
                    metadatas=[base_metadata],
                )
                all_chunks.extend(chunks)

    table_count = sum(1 for c in all_chunks if c.metadata.get("has_table"))
    logger.info(
        f"Split {len(documents)} pages → {len(all_chunks)} chunks "
        f"({table_count} table chunks, {len(all_chunks) - table_count} text chunks)"
    )

    return all_chunks
