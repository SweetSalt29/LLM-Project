import os
from pathlib import Path
import fitz  # PyMuPDF

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from langchain_docling import DoclingLoader
from langchain_core.documents import Document


# ============================================================
# FORMAT ROUTING
# ============================================================
DOCLING_SUPPORTED  = {".pdf", ".docx", ".doc"}
PLAINTEXT_FORMATS  = {".txt"}
CHM_FORMATS        = {".chm"}
MSG_FORMATS        = {".msg"}

ALL_SUPPORTED = DOCLING_SUPPORTED | PLAINTEXT_FORMATS | CHM_FORMATS | MSG_FORMATS

# Chunk size for plain text and extracted CHM/MSG content
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100


# ============================================================
# CHUNKER — shared utility
# Splits a long string into overlapping chunks as Documents
# ============================================================
def chunk_text(text: str, file_path: str, extra_metadata: dict = None) -> list:
    """
    Split text into overlapping chunks of CHUNK_SIZE chars.
    Returns a list of Document objects.
    """
    source = Path(file_path).name
    base_metadata = {
        "source":    source,
        "file_path": file_path,
        "page":      None,
        "images":    []
    }
    if extra_metadata:
        base_metadata.update(extra_metadata)

    chunks = []
    start  = 0

    while start < len(text):
        end   = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(Document(page_content=chunk, metadata=dict(base_metadata)))
        start += CHUNK_SIZE - CHUNK_OVERLAP

    if not chunks:
        chunks.append(Document(
            page_content="(empty file)",
            metadata=base_metadata
        ))

    return chunks


# ============================================================
# TEXT LOADER
# Routes to correct loader based on file extension
# ============================================================
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

        if suffix in DOCLING_SUPPORTED:
            return self._load_docling(file_path)

        if suffix in PLAINTEXT_FORMATS:
            return self._load_plain_text(file_path)

        if suffix in CHM_FORMATS:
            return self._load_chm(file_path)

        if suffix in MSG_FORMATS:
            return self._load_msg(file_path)

        raise ValueError(
            f"Unsupported format for RAG ingestion: '{suffix}'. "
            f"Supported: {', '.join(sorted(ALL_SUPPORTED))}"
        )

    # ----------------------------------------------------------
    # DOCLING — PDF, DOCX, DOC
    # ----------------------------------------------------------
    def _load_docling(self, file_path: str) -> list:
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

    # ----------------------------------------------------------
    # PLAIN TEXT — TXT
    # ----------------------------------------------------------
    def _load_plain_text(self, file_path: str) -> list:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            full_text = f.read()

        return chunk_text(full_text, file_path)

    # ----------------------------------------------------------
    # CHM — Windows Help files
    # Uses pychm to extract HTML pages, strips tags for clean text
    # Install: pip install pychm
    # ----------------------------------------------------------
    def _load_chm(self, file_path: str) -> list:
        try:
            import chm.chm as chmlib
            from html.parser import HTMLParser
        except ImportError:
            raise ImportError(
                "pychm is required for CHM files. Install it with: pip install pychm"
            )

        class HTMLTextExtractor(HTMLParser):
            """Strips HTML tags and returns plain text."""
            def __init__(self):
                super().__init__()
                self._parts = []
                self._skip  = False

            def handle_starttag(self, tag, attrs):
                if tag in ("script", "style"):
                    self._skip = True

            def handle_endtag(self, tag):
                if tag in ("script", "style"):
                    self._skip = False

            def handle_data(self, data):
                if not self._skip:
                    stripped = data.strip()
                    if stripped:
                        self._parts.append(stripped)

            def get_text(self):
                return "\n".join(self._parts)

        chm_file = chmlib.CHMFile()
        if not chm_file.LoadCHM(file_path):
            raise ValueError(f"Could not open CHM file: {file_path}")

        all_text_parts = []

        def extract_page(ui, file_path_ref):
            """Callback to extract each page from CHM."""
            try:
                result, content = chm_file.RetrieveObject(ui)
                if result == chmlib.CHM_RESOLVE_SUCCESS and content:
                    html = content.decode("utf-8", errors="ignore")
                    parser = HTMLTextExtractor()
                    parser.feed(html)
                    text = parser.get_text()
                    if text.strip():
                        all_text_parts.append(text)
            except Exception:
                pass  # skip unreadable pages silently
            return True

        chm_file.EnumerateDir("/", chmlib.CHM_ENUMERATE_ALL, extract_page, None)
        chm_file.CloseCHM()

        full_text = "\n\n".join(all_text_parts)

        if not full_text.strip():
            raise ValueError(f"No readable text found in CHM file: {Path(file_path).name}")

        return chunk_text(full_text, file_path, extra_metadata={"format": "chm"})

    # ----------------------------------------------------------
    # MSG — Outlook Email files
    # Uses extract-msg to read email fields and body
    # Install: pip install extract-msg
    # ----------------------------------------------------------
    def _load_msg(self, file_path: str) -> list:
        try:
            import extract_msg
        except ImportError:
            raise ImportError(
                "extract-msg is required for MSG files. Install it with: pip install extract-msg"
            )

        msg = extract_msg.Message(file_path)

        # Build structured text from all email fields
        parts = []

        if msg.subject:
            parts.append(f"Subject: {msg.subject}")

        if msg.sender:
            parts.append(f"From: {msg.sender}")

        if msg.to:
            parts.append(f"To: {msg.to}")

        if msg.cc:
            parts.append(f"CC: {msg.cc}")

        if msg.date:
            parts.append(f"Date: {msg.date}")

        parts.append("")  # blank line before body

        # Prefer plain text body; fall back to HTML body stripped of tags
        body = msg.body
        if not body and msg.htmlBody:
            from html.parser import HTMLParser

            class _Stripper(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self._parts = []
                def handle_data(self, data):
                    if data.strip():
                        self._parts.append(data.strip())
                def get_text(self):
                    return "\n".join(self._parts)

            stripper = _Stripper()
            stripper.feed(msg.htmlBody.decode("utf-8", errors="ignore")
                          if isinstance(msg.htmlBody, bytes) else msg.htmlBody)
            body = stripper.get_text()

        if body:
            parts.append(body)

        # List attachment names so they're searchable
        if msg.attachments:
            attachment_names = [
                a.longFilename or a.shortFilename or "unnamed"
                for a in msg.attachments
            ]
            parts.append(f"\nAttachments: {', '.join(attachment_names)}")

        msg.close()

        full_text = "\n".join(parts)

        if not full_text.strip():
            raise ValueError(f"No readable content found in MSG file: {Path(file_path).name}")

        return chunk_text(
            full_text,
            file_path,
            extra_metadata={
                "format":  "msg",
                "subject": msg.subject or "",
                "sender":  msg.sender or ""
            }
        )


# ============================================================
# IMAGE LOADER — PDF only
# ============================================================
class ImageLoader:
    def extract(self, pdf_path: str, output_dir: str) -> list:
        """Extract images from PDF and save uniquely per file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        doc       = fitz.open(pdf_path)
        images    = []
        file_name = Path(pdf_path).stem

        for page_index in range(len(doc)):
            page       = doc[page_index]
            image_list = page.get_images(full=True)

            for img_index, img in enumerate(image_list):
                xref        = img[0]
                base_image  = doc.extract_image(xref)
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


# ============================================================
# MULTIMODAL LOADER (MAIN ENTRY POINT)
# ============================================================
class MultimodalLoader:
    def __init__(self, image_output_dir="data/extracted_images"):
        self.text_loader      = TextLoader()
        self.image_loader     = ImageLoader()
        self.image_output_dir = image_output_dir

    def load(self, file_path: str) -> list:
        """Load file → extract text + images (PDF only) → link them."""
        suffix = Path(file_path).suffix.lower()

        # Text extraction — routed by format inside TextLoader
        text_docs = self.text_loader.load(file_path)

        # Image extraction — PDF only, non-fatal
        images = []
        if suffix == ".pdf":
            try:
                images = self.image_loader.extract(file_path, self.image_output_dir)
            except Exception:
                images = []

        # Map page number → image paths
        page_to_images = {}
        for img in images:
            page_to_images.setdefault(img["page"], []).append(img["image_path"])

        # Link each text doc with images from the same page
        multimodal_docs = []
        for doc in text_docs:
            page          = doc.metadata.get("page")
            linked_images = page_to_images.get(page, [])

            multimodal_docs.append(
                Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "images": linked_images}
                )
            )

        return multimodal_docs


# ============================================================
# PREPARE DOCUMENTS (FOR EMBEDDING)
# ============================================================
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