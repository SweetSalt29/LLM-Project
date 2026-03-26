import os
from pathlib import Path
import fitz  # PyMuPDF

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from langchain_docling import DoclingLoader
from langchain_core.documents import Document


# -----------------------------
# TEXT LOADER
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

    def load(self, file_path: str):
        """
        Load text using Docling (primarily for PDF)
        """
        loader = DoclingLoader(
            file_path=file_path,
            converter=self.converter
        )

        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = Path(file_path).name
            doc.metadata["file_path"] = file_path
            doc.metadata["page"] = doc.metadata.get("page_number", None)

        return docs


# -----------------------------
# IMAGE LOADER
# -----------------------------
class ImageLoader:
    def extract(self, pdf_path: str, output_dir: str):
        """
        Extract images from PDF and save uniquely per file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(pdf_path)
        images = []

        file_name = Path(pdf_path).stem

        for page_index in range(len(doc)):
            page = doc[page_index]
            image_list = page.get_images(full=True)

            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)

                image_bytes = base_image["image"]
                ext = base_image["ext"]

                filename = f"{file_name}_p{page_index+1}_img{img_index+1}.{ext}"
                filepath = output_path / filename

                with open(filepath, "wb") as f:
                    f.write(image_bytes)

                images.append({
                    "image_path": str(filepath),
                    "page": page_index + 1
                })

        return images


# -----------------------------
# MULTIMODAL LOADER (MAIN)
# -----------------------------
class MultimodalLoader:
    def __init__(self, image_output_dir="data/extracted_images"):
        self.text_loader = TextLoader()
        self.image_loader = ImageLoader()
        self.image_output_dir = image_output_dir

    def load(self, file_path: str):
        """
        Load file → extract text + images → link them
        """
        suffix = file_path.split(".")[-1].lower()

        # -----------------------------
        # TEXT EXTRACTION
        # -----------------------------
        text_docs = self.text_loader.load(file_path)

        # -----------------------------
        # IMAGE EXTRACTION (PDF ONLY)
        # -----------------------------
        images = []
        if suffix == "pdf":
            images = self.image_loader.extract(file_path, self.image_output_dir)

        # -----------------------------
        # PAGE → IMAGE MAPPING
        # -----------------------------
        page_to_images = {}
        for img in images:
            page_to_images.setdefault(img["page"], []).append(img["image_path"])

        # -----------------------------
        # LINK TEXT + IMAGES
        # -----------------------------
        multimodal_docs = []

        for doc in text_docs:
            page = doc.metadata.get("page")
            linked_images = page_to_images.get(page, [])

            new_doc = Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "images": linked_images
                }
            )

            multimodal_docs.append(new_doc)

        return multimodal_docs


# -----------------------------
# PREPARE DOCUMENTS (FOR EMBEDDING)
# -----------------------------
def prepare_documents(multimodal_docs):
    """
    Merge text + image metadata into embedding-ready text
    """
    processed_docs = []

    for doc in multimodal_docs:
        images = doc.metadata.get("images", [])

        # ⚠️ For now: use image filenames
        # Later: replace with image captions
        image_text = ""
        if images:
            image_text = "\n".join([f"Image reference: {Path(img).name}" for img in images])

        final_text = f"{doc.page_content}\n\n{image_text}"

        processed_docs.append(
            Document(
                page_content=final_text,
                metadata=doc.metadata
            )
        )

    return processed_docs