import os
from pathlib import Path
import fitz  # PyMuPDF

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from langchain_docling import DoclingLoader
from langchain_core.documents import Document


# ============================================================
# DOCLING SUPPORTED FORMATS
# Everything else falls back to plain text read
# ============================================================
DOCLING_SUPPORTED = {".pdf", ".docx", ".doc"}
PLAINTEXT_FORMATS = {".txt"}


# -----------------------------
# TEXT LOADER
# Routes to Docling or plain text depending on file type
# -----------------------------
class TextLoader:
    def __init__(self):
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.do_ocr = False
        self.pipeline_options.do_table_structure = True

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.pipeline_options
                )
            }
        )

    def load(self, file_path: str) -> list:
        suffix = Path(file_path).suffix.lower()

        if suffix in PLAINTEXT_FORMATS:
            return self._load_plain_text(file_path)

        if suffix in DOCLING_SUPPORTED:
            return self._load_docling(file_path)

        raise ValueError(
            f"Unsupported format for RAG ingestion: {suffix}. "
            f"Supported: {', '.join(DOCLING_SUPPORTED | PLAINTEXT_FORMATS)}"
        )

    def _load_docling(self, file_path: str) -> list:
        """Use Docling for PDF, DOCX, DOC — handles structure and tables."""
        loader = DoclingLoader(
            file_path=file_path,
            converter=self.converter
        )
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"]    = Path(file_path).name
            doc.metadata["file_path"] = file_path
            doc.metadata["page"]      = doc.metadata.get("page_number", None)

        return docs

    def _load_plain_text(self, file_path: str) -> list:
        """
        Read .txt files directly — no Docling needed.
        Splits into chunks of ~1000 chars to keep embedding size manageable.
        """
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            full_text = f.read()

        # Split into chunks so large txt files don't become a single giant embedding
        chunk_size = 1000
        overlap    = 100
        chunks     = []
        start      = 0

        while start < len(full_text):
            end   = min(start + chunk_size, len(full_text))
            chunk = full_text[start:end].strip()
            if chunk:
                chunks.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source":    Path(file_path).name,
                            "file_path": file_path,
                            "page":      None,
                            "images":    []
                        }
                    )
                )
            start += chunk_size - overlap

        return chunks if chunks else [
            Document(
                page_content="(empty file)",
                metadata={"source": Path(file_path).name, "file_path": file_path, "page": None, "images": []}
            )
        ]


# -----------------------------
# IMAGE LOADER
# -----------------------------
class ImageLoader:
    def extract(self, pdf_path: str, output_dir: str) -> list:
        """Extract images from PDF and save uniquely per file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        doc    = fitz.open(pdf_path)
        images = []
        file_name = Path(pdf_path).stem

        for page_index in range(len(doc)):
            page       = doc[page_index]
            image_list = page.get_images(full=True)

            for img_index, img in enumerate(image_list):
                xref       = img[0]
                base_image = doc.extract_image(xref)

                image_bytes = base_image["image"]
                ext         = base_image["ext"]

                filename = f"{file_name}_p{page_index+1}_img{img_index+1}.{ext}"
                filepath = output_path / filename

                with open(filepath, "wb") as f:
                    f.write(image_bytes)

                images.append({
                    "image_path": str(filepath),
                    "page":       page_index + 1
                })

        return images


# -----------------------------
# MULTIMODAL LOADER (MAIN)
# -----------------------------
class MultimodalLoader:
    def __init__(self, image_output_dir="data/extracted_images"):
        self.text_loader      = TextLoader()
        self.image_loader     = ImageLoader()
        self.image_output_dir = image_output_dir

    def load(self, file_path: str) -> list:
        """Load file → extract text + images (PDF only) → link them."""
        suffix = Path(file_path).suffix.lower()

        # Text extraction (routes internally by format)
        text_docs = self.text_loader.load(file_path)

        # Image extraction — PDF only
        images = []
        if suffix == ".pdf":
            try:
                images = self.image_loader.extract(file_path, self.image_output_dir)
            except Exception:
                images = []  # non-fatal — continue without images

        # Page → image mapping
        page_to_images = {}
        for img in images:
            page_to_images.setdefault(img["page"], []).append(img["image_path"])

        # Link text docs with their images
        multimodal_docs = []
        for doc in text_docs:
            page         = doc.metadata.get("page")
            linked_images = page_to_images.get(page, [])

            multimodal_docs.append(
                Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "images": linked_images}
                )
            )

        return multimodal_docs


# -----------------------------
# PREPARE DOCUMENTS (FOR EMBEDDING)
# -----------------------------
def prepare_documents(multimodal_docs: list) -> list:
    """Merge text + image metadata into embedding-ready Documents."""
    processed_docs = []

    for doc in multimodal_docs:
        images = doc.metadata.get("images", [])

        image_text = ""
        if images:
            image_text = "\n".join([
                f"Image reference: {Path(img).name}" for img in images
            ])

        final_text = f"{doc.page_content}\n\n{image_text}".strip()

        processed_docs.append(
            Document(
                page_content=final_text,
                metadata=doc.metadata
            )
        )

    return processed_docs