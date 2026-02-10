import os
from typing import List, Dict, Any
from pypdf import PdfReader
import docx
from src.core.agent import BaseAgent

class InputReaderAgent(BaseAgent):
    def process(self, input_path: str) -> str:
        """
        Reads a file and returns its text content.
        Supported formats: .txt, .pdf, .docx
        """
        self.logger.info(f"Reading input file: {input_path}")
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        ext = os.path.splitext(input_path)[1].lower()

        if ext == '.txt':
            return self._read_text(input_path)
        elif ext == '.pdf':
            return self._read_pdf(input_path)
        elif ext == '.docx':
            return self._read_docx(input_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _read_text(self, path: str) -> str:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    def _read_pdf(self, path: str) -> str:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    def _read_docx(self, path: str) -> str:
        doc = docx.Document(path)
        return "\n".join([para.text for para in doc.paragraphs])

    def chunk_text(self, text: str, max_words: int = 2000) -> List[str]:
        """
        Splits text into manageable chunks.
        For Phase 2 enhancement, this now attempts to respect paragraph boundaries and potentially
        uses an LLM to find scene breaks if the text is very long.
        """
        # Simple paragraph-based chunking for now to avoid breaking sentences
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_count = 0
        
        for para in paragraphs:
            para_words = len(para.split())
            if current_count + para_words > max_words:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_count = para_words
            else:
                current_chunk.append(para)
                current_count += para_words
        
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
            
        return chunks

    def detect_scenes(self, text_chunk: str, llm_interface) -> List[str]:
        """
        Uses an LLM to identify distinct scenes within a text chunk.
        """
        # This would correspond to the 'Smart Chunking' feature
        # For now, implemented as a placeholder for the advanced logic
        self.logger.info("Detecting scenes in text chunk...")
        # In a full implementation, we'd ask the LLM to return split indices or text segments
        # For now, we return the chunk as a single scene if no clear breaks found
        return [text_chunk]
