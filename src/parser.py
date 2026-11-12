import os
import re
from pathlib import Path
from typing import List, Dict, Any

class MarkdownParser:
    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path)

    def parse_vault(self) -> List[Dict[str, Any]]:
        """
        Scans the directory, parses markdown files, 
        and splits them into semantic chunks based on headers.
        """
        documents = []
        if not self.vault_path.exists():
            return documents

        for file_path in self.vault_path.glob("**/*.md"):
            if ".obsidian" in str(file_path):
                continue
            
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            chunks = self._semantic_chunk(content, file_path.name)
            documents.extend(chunks)

        return documents

    def _semantic_chunk(self, content: str, file_name: str, max_chunk_size: int = 500) -> List[Dict[str, Any]]:
        """
        Splits markdown text by headers to preserve contextual boundaries.
        """
        # Strip YAML frontmatter if present
        content = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)
        
        # Split by markdown headers (#, ##, ###)
        sections = re.split(r'\n(?=#{1,3}\s+)', content)
        chunks = []

        for idx, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue

            # Extract header title if present
            header_match = re.match(r'^(#{1,3})\s+(.+)', section)
            header_title = header_match.group(2) if header_match else "General"

            # If section is too long, sub-chunk it by paragraphs
            if len(section) > max_chunk_size:
                paragraphs = section.split('\n\n')
                current_chunk = ""
                for p in paragraphs:
                    if len(current_chunk) + len(p) < max_chunk_size:
                        current_chunk += "\n\n" + p
                    else:
                        if current_chunk.strip():
                            chunks.append({
                                "file_name": file_name,
                                "header": header_title,
                                "text": current_chunk.strip()
                            })
                        current_chunk = p
                if current_chunk.strip():
                    chunks.append({
                        "file_name": file_name,
                        "header": header_title,
                        "text": current_chunk.strip()
                    })
            else:
                chunks.append({
                    "file_name": file_name,
                    "header": header_title,
                    "text": section
                })

        return chunks