import os
import re
import base64
import requests
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
DOCLING_SUPPORTED = {".pdf", ".docx", ".doc"}
PLAINTEXT_FORMATS = {".txt"}
CHM_FORMATS       = {".chm"}
MSG_FORMATS       = {".msg"}

ALL_SUPPORTED = DOCLING_SUPPORTED | PLAINTEXT_FORMATS | CHM_FORMATS | MSG_FORMATS

# Vision model used at ingest time to caption extracted images
VISION_MODEL = "qwen/qwen3-vl-235b-thinking:free"

# ============================================================
# CHUNKING CONSTANTS
# ============================================================
MAX_CHUNK_CHARS   = 1000   # Hard ceiling per chunk — oversized paragraphs get sentence-split
MIN_CHUNK_CHARS   = 80     # Chunks below this are merged into the next one
OVERLAP_SENTENCES = 2      # Sentences carried from end of previous chunk into next


# ============================================================
# SENTENCE SPLITTER — shared utility
# Splits text into sentences using punctuation boundaries.
# ============================================================
def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences on . ! ? boundaries.
    Handles abbreviations poorly at scale but good enough for overlap stitching.
    """
    # Split on sentence-ending punctuation followed by whitespace or end
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw if s.strip()]


# ============================================================
# PARAGRAPH CHUNKER
# Used for unstructured text: TXT, CHM, MSG.
#
# Strategy:
#   1. Split by blank lines (paragraph boundaries)
#   2. Merge tiny paragraphs (< MIN_CHUNK_CHARS) upward into the next
#   3. Split oversized paragraphs (> MAX_CHUNK_CHARS) by sentence
#   4. Add sentence-level overlap between consecutive chunks
# ============================================================
def chunk_by_paragraphs(
    text: str,
    file_path: str,
    extra_metadata: dict = None
) -> list[Document]:
    source = Path(file_path).name
    base_meta = {
        "source":         source,
        "file_path":      file_path,
        "page":           None,
        "images":         [],
        "is_image_chunk": False
    }
    if extra_metadata:
        base_meta.update(extra_metadata)

    # ── Step 1: split on blank lines ──────────────────────────────
    raw_paragraphs = re.split(r'\n{2,}', text)
    raw_paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

    # ── Step 2: merge tiny paragraphs forward ─────────────────────
    merged = []
    buffer = ""
    for para in raw_paragraphs:
        if buffer:
            candidate = buffer + " " + para
            if len(buffer) < MIN_CHUNK_CHARS:
                # Buffer is tiny — absorb this paragraph into it
                buffer = candidate
            else:
                merged.append(buffer)
                buffer = para
        else:
            buffer = para
    if buffer:
        merged.append(buffer)

    # ── Step 3: sentence-split any paragraph > MAX_CHUNK_CHARS ────
    raw_chunks = []
    for para in merged:
        if len(para) <= MAX_CHUNK_CHARS:
            raw_chunks.append(para)
        else:
            sentences  = split_into_sentences(para)
            current    = ""
            for sent in sentences:
                if len(current) + len(sent) + 1 <= MAX_CHUNK_CHARS:
                    current = (current + " " + sent).strip() if current else sent
                else:
                    if current:
                        raw_chunks.append(current)
                    current = sent
            if current:
                raw_chunks.append(current)

    if not raw_chunks:
        return [Document(page_content="(empty file)", metadata=base_meta)]

    # ── Step 4: add sentence-level overlap ────────────────────────
    docs           = []
    prev_sentences = []   # tail sentences from the previous chunk

    for i, chunk_text in enumerate(raw_chunks):
        # Prepend overlap from previous chunk
        if prev_sentences:
            overlap_text = " ".join(prev_sentences)
            content      = overlap_text + " " + chunk_text
        else:
            content = chunk_text

        docs.append(Document(page_content=content.strip(), metadata=dict(base_meta)))

        # Save last OVERLAP_SENTENCES of THIS chunk as overlap for next
        these_sentences = split_into_sentences(chunk_text)
        prev_sentences  = these_sentences[-OVERLAP_SENTENCES:] if these_sentences else []

    return docs


# ============================================================
# DOCLING SEGMENT MERGER + OVERLAP
# Used for structured docs: PDF, DOCX, DOC.
# Docling already segments well — we just:
#   1. Merge consecutive tiny segments (< MIN_CHUNK_CHARS) to avoid micro-chunks
#   2. Add OVERLAP_SENTENCES of overlap between consecutive chunks
# ============================================================
def apply_overlap_to_docling_docs(docs: list[Document]) -> list[Document]:
    """
    Takes Docling-output Document list (already semantically segmented).
    Merges tiny consecutive segments, then adds sentence-level overlap.
    Preserves all existing metadata (source, file_path, page, etc.).
    """
    if not docs:
        return docs

    # ── Step 1: merge tiny segments forward ───────────────────────
    merged     = []
    buffer_doc = None

    for doc in docs:
        text = doc.page_content.strip()
        if not text:
            continue

        if buffer_doc is None:
            buffer_doc = Document(
                page_content=text,
                metadata=dict(doc.metadata)
            )
        else:
            if len(buffer_doc.page_content) < MIN_CHUNK_CHARS:
                # Merge into buffer — keep metadata of the first segment
                buffer_doc = Document(
                    page_content=buffer_doc.page_content + " " + text,
                    metadata=buffer_doc.metadata
                )
            else:
                merged.append(buffer_doc)
                buffer_doc = Document(
                    page_content=text,
                    metadata=dict(doc.metadata)
                )

    if buffer_doc:
        merged.append(buffer_doc)

    if not merged:
        return docs

    # ── Step 2: add sentence-level overlap ────────────────────────
    result         = []
    prev_sentences = []

    for doc in merged:
        if prev_sentences:
            overlap_text = " ".join(prev_sentences)
            content      = overlap_text + " " + doc.page_content
        else:
            content = doc.page_content

        result.append(Document(
            page_content=content.strip(),
            metadata=doc.metadata
        ))

        these_sentences = split_into_sentences(doc.page_content)
        prev_sentences  = these_sentences[-OVERLAP_SENTENCES:] if these_sentences else []

    return result


# ============================================================
# VISION CAPTIONER
# ============================================================
class VisionCaptioner:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            print("[Warning] OPENROUTER_API_KEY is not set. Image captioning will fail.")
        self.model = VISION_MODEL

    def caption(self, image_path: str) -> str:
        """
        Send image to Qwen3-VL and return a detailed description.
        Returns empty string on any failure — ingest continues regardless.
        """
        if not self.api_key:
            return ""

        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()

            ext = Path(image_path).suffix.lower().lstrip(".")
            media_type_map = {
                "jpg":  "image/jpeg",
                "jpeg": "image/jpeg",
                "png":  "image/png",
                "gif":  "image/gif",
                "webp": "image/webp",
                "bmp":  "image/bmp",
            }
            media_type = media_type_map.get(ext, "image/png")
            b64_image  = base64.b64encode(image_bytes).decode("utf-8")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{b64_image}"}
                        },
                        {
                            "type": "text",
                            "text": (
                                "Describe this image in detail for use in a document Q&A system. "
                                "If it is a chart or graph: state the chart type, axis labels, "
                                "all data values or ranges, trends, and key takeaways. "
                                "If it is a table: describe its structure and transcribe all cell values. "
                                "If it is a diagram or illustration: explain what it depicts and all labeled components. "
                                "If it contains text: transcribe it fully. "
                                "Be thorough — your description will be used to answer user questions "
                                "about this image without it being shown again."
                            )
                        }
                    ]
                }
            ]

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type":  "application/json"
                },
                json={"model": self.model, "messages": messages},
                timeout=60
            )

            data = response.json()
            if "error" in data:
                print(f"[VisionCaptioner] API error for {Path(image_path).name}: {data['error']}")
                return ""

            return data["choices"][0]["message"]["content"].strip()

        except Exception as e:
            print(f"[VisionCaptioner] Failed to caption {Path(image_path).name}: {e}")
            return ""


# ============================================================
# TEXT LOADER
# ============================================================
class TextLoader:
    def __init__(self):
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.do_ocr             = False
        self.pipeline_options.do_table_structure  = True

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

    def _load_docling(self, file_path: str) -> list:
        suffix = Path(file_path).suffix.lower()
        loader = DoclingLoader(file_path=file_path, converter=self.converter)
        docs   = list(loader.load())

        for doc in docs:
            doc.metadata["source"]         = Path(file_path).name
            doc.metadata["file_path"]      = file_path
            doc.metadata["page"]           = doc.metadata.get("page_number", None)
            doc.metadata["is_image_chunk"] = False
            doc.metadata["images"]         = []
            doc.metadata["format"]         = suffix.replace(".", "")

        # Apply segment-merge + sentence overlap to Docling output
        return apply_overlap_to_docling_docs(docs)

    def _load_plain_text(self, file_path: str) -> list:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            full_text = f.read()
        return chunk_by_paragraphs(full_text, file_path, extra_metadata={"format": "txt"})

    def _load_chm(self, file_path: str) -> list:
        try:
            import chm.chm as chmlib
            from html.parser import HTMLParser
        except ImportError:
            raise ImportError(
                "CHM support requires 'pychm'. Install it using:\n\npip install pychm"
            )

        class HTMLTextExtractor(HTMLParser):
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

        def extract_page(ui, _):
            try:
                result, content = chm_file.RetrieveObject(ui)
                if result == chmlib.CHM_RESOLVE_SUCCESS and content:
                    html   = content.decode("utf-8", errors="ignore")
                    parser = HTMLTextExtractor()
                    parser.feed(html)
                    text = parser.get_text()
                    if text.strip():
                        all_text_parts.append(text)
            except Exception:
                pass
            return True

        chm_file.EnumerateDir("/", chmlib.CHM_ENUMERATE_ALL, extract_page, None)
        chm_file.CloseCHM()

        full_text = "\n\n".join(all_text_parts)
        if not full_text.strip():
            raise ValueError(f"No readable text found in CHM file: {Path(file_path).name}")

        return chunk_by_paragraphs(full_text, file_path, extra_metadata={"format": "chm"})

    def _load_msg(self, file_path: str) -> list:
        try:
            import extract_msg
        except ImportError:
            raise ImportError(
                "MSG support requires 'extract-msg'. Install it using:\n\npip install extract-msg"
            )

        msg   = extract_msg.openMsg(file_path)
        parts = []

        if msg.subject: parts.append(f"Subject: {msg.subject}")
        if msg.sender:  parts.append(f"From: {msg.sender}")
        if msg.to:      parts.append(f"To: {msg.to}")
        if msg.cc:      parts.append(f"CC: {msg.cc}")
        if msg.date:    parts.append(f"Date: {msg.date}")

        parts.append("")

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
            stripper.feed(
                msg.htmlBody.decode("utf-8", errors="ignore")
                if isinstance(msg.htmlBody, bytes) else msg.htmlBody
            )
            body = stripper.get_text()

        if body:
            parts.append(body)

        attachments_text = []
        if msg.attachments:
            for att in msg.attachments:
                name = att.longFilename or att.shortFilename or "unnamed"
                try:
                    data = att.data
                    if isinstance(data, bytes):
                        text = data.decode("utf-8", errors="ignore")
                        if text.strip():
                            attachments_text.append(f"\n[Attachment: {name}]\n{text}")
                except Exception:
                    pass

        if attachments_text:
            parts.append("\n".join(attachments_text))

        msg.close()

        full_text = "\n".join(parts)
        if not full_text.strip():
            raise ValueError(f"No readable content found in MSG file: {Path(file_path).name}")

        return chunk_by_paragraphs(
            full_text, file_path,
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
    MIN_IMAGE_BYTES = 5000

    def extract(self, pdf_path: str, output_dir: str) -> list:
        """Extract content images from PDF pages and save to disk."""
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

                if len(image_bytes) < self.MIN_IMAGE_BYTES:
                    continue

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
        suffix = Path(file_path).suffix.lower()

        text_docs = self.text_loader.load(file_path)

        images = []
        if suffix == ".pdf":
            try:
                images = self.image_loader.extract(file_path, self.image_output_dir)
            except Exception as e:
                print(f"[MultimodalLoader] Image extraction failed: {e}")

        page_to_images = {}
        for img in images:
            page_to_images.setdefault(img["page"], []).append(img["image_path"])

        multimodal_docs = []
        for doc in text_docs:
            page          = doc.metadata.get("page")
            linked_images = page_to_images.get(page, [])
            multimodal_docs.append(
                Document(
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        "images":         linked_images,
                        "is_image_chunk": False
                    }
                )
            )

        return multimodal_docs


# ============================================================
# PREPARE DOCUMENTS (FOR EMBEDDING)
# ============================================================
def prepare_documents(multimodal_docs: list) -> list:
    captioner   = VisionCaptioner()
    processed   = []
    seen_images = set()

    for doc in multimodal_docs:
        images = doc.metadata.get("images", [])

        # ── 1. TEXT CHUNK ──────────────────────────────────────────
        processed.append(
            Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "is_image_chunk": False}
            )
        )

        # ── 2. IMAGE CHUNKS ────────────────────────────────────────
        for image_path in images:
            if image_path in seen_images:
                continue
            seen_images.add(image_path)

            print(f"[prepare_documents] Captioning: {Path(image_path).name}")
            caption = captioner.caption(image_path)

            if not caption:
                caption = (
                    f"Image on page {doc.metadata.get('page', '?')} "
                    f"of {doc.metadata.get('source', 'document')}. "
                    "Visual content could not be automatically described."
                )

            processed.append(
                Document(
                    page_content=caption,
                    metadata={
                        "source":         doc.metadata.get("source"),
                        "file_path":      doc.metadata.get("file_path"),
                        "page":           doc.metadata.get("page"),
                        "image_path":     image_path,
                        "is_image_chunk": True,
                        "images":         []
                    }
                )
            )

    return processed