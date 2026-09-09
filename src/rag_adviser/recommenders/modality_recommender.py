"""Per-modality ingestion recommendations for non-text corpus content.

A text-only RAG recipe silently drops images, video, audio, spreadsheets,
slide decks, CAD drawings and scanned PDFs. This module turns the modality
inventory produced by the document analyzer into concrete ingestion advice:
how to get each modality into text or vectors, which tools to use locally or
via hosted APIs, how to chunk it, and what to watch out for.
"""

from __future__ import annotations

from rag_adviser.analyzers.constraint_analyzer import ConstraintAnalyzer
from rag_adviser.models import (
    BudgetTier,
    DocumentStats,
    HardwareProfile,
    ModalityRecommendation,
    UserAnswers,
)

# Modalities that need treatment beyond the text pipeline, in display order.
_ORDER = ["scanned_pdf", "image", "spreadsheet", "presentation", "video", "audio", "cad", "other"]


class ModalityRecommender:
    """Recommend ingestion strategies for every non-text modality present."""

    def recommend(
        self, stats: DocumentStats, answers: UserAnswers
    ) -> list[ModalityRecommendation]:
        constraints = answers.constraints
        api_ok = ConstraintAnalyzer().is_api_allowed(constraints)
        paid = constraints.budget == BudgetTier.PAID_API
        gpu = constraints.hardware == HardwareProfile.GPU_AVAILABLE
        total = max(stats.total_files_all, 1)

        counts = dict(stats.modalities)
        if stats.scanned_pdfs > 0:
            # Extrapolate from the sampled PDFs to the PDF population.
            pdf_total = stats.file_types.get(".pdf", 0)
            ratio = stats.scanned_pdfs / max(stats.sampled_pdfs, 1)
            counts["scanned_pdf"] = max(1, round(pdf_total * ratio)) if pdf_total else stats.scanned_pdfs

        recs: list[ModalityRecommendation] = []
        for modality in _ORDER:
            n = counts.get(modality, 0)
            if n <= 0:
                continue
            builder = getattr(self, f"_{modality}")
            rec: ModalityRecommendation = builder(api_ok and paid, gpu, answers)
            rec.modality = modality
            rec.file_count = n
            rec.share = n / total
            rec.extensions = sorted(
                ext for ext, c in stats.file_types.items()
                if c > 0 and _modality_of_extension(ext) == modality
            )
            if not (api_ok and paid) and rec.tools_hosted:
                rec.notes.append(
                    "Hosted services listed for reference only: privacy/budget settings "
                    "keep this pipeline local"
                )
            recs.append(rec)
        return recs

    # ── Builders ───────────────────────────────────────────────────────────

    @staticmethod
    def _scanned_pdf(hosted: bool, gpu: bool, answers: UserAnswers) -> ModalityRecommendation:
        return ModalityRecommendation(
            strategy="OCR + layout-aware parsing before chunking",
            ingestion=[
                "Detect scanned pages (no extractable text layer) and route them to OCR",
                "Prefer a layout-aware parser that outputs Markdown with headings and tables",
                "Keep page numbers and bounding boxes as chunk metadata for citations",
            ],
            tools_local=[
                "docling (IBM) - layout + tables + OCR, outputs Markdown",
                "marker - fast PDF to Markdown with OCR fallback",
                "PaddleOCR or Tesseract (pytesseract) for raw OCR",
            ],
            tools_hosted=[
                "Azure Document Intelligence", "AWS Textract", "Google Document AI",
                "Mistral OCR / LlamaParse",
            ],
            embedding="Text embedding of OCR output (same model as the rest of the corpus)",
            chunking="Chunk the parsed Markdown by heading/section, not by page",
            pip_packages=["docling"],
            notes=[
                "For visually rich pages (forms, engineering tables) consider ColPali-style "
                "visual document retrieval (vidore/colpali) instead of OCR",
            ],
            warnings=["OCR errors propagate into retrieval; sample and spot-check the output"],
        )

    @staticmethod
    def _image(hosted: bool, gpu: bool, answers: UserAnswers) -> ModalityRecommendation:
        rec = ModalityRecommendation(
            strategy="Caption or OCR images into text, or embed them in a separate image index",
            ingestion=[
                "Split images into two groups: document-like (screenshots, scans, diagrams) "
                "and photographic",
                "Document-like: OCR/parse them like scanned PDFs (docling, PaddleOCR)",
                "Photographic/diagrams: generate a dense caption with a vision-language "
                "model and index the caption as text; store the image path as metadata",
                "Optionally add a second collection of image embeddings (CLIP/SigLIP) for "
                "image-to-image or text-to-image search",
            ],
            tools_local=[
                "Qwen2.5-VL or Florence-2 for captioning (GPU recommended)",
                "sentence-transformers clip-ViT-B-32 / clip-ViT-B-32-multilingual-v1 for image embeddings",
                "docling / PaddleOCR for text-heavy images",
            ],
            tools_hosted=[
                "Claude or GPT-4o vision for captioning",
                "Cohere embed-v4 or Voyage voyage-multimodal-3 for joint text+image embeddings",
            ],
            embedding=(
                "Multimodal embedding model (joint text/image space)" if hosted
                else "Text embedding of captions; CLIP-family model for a separate image index"
            ),
            chunking="One chunk per image (caption + OCR text + surrounding document context)",
            pip_packages=["sentence-transformers", "pillow"],
            notes=["Captions should describe what matters for your queries (part numbers, labels, layout), so prompt the captioner with the domain"],
            warnings=[],
        )
        if not gpu:
            rec.warnings.append("Local vision-language captioning is slow on CPU; batch it offline")
        rec.code_snippet = "\n".join([
            "# pip install sentence-transformers pillow",
            "from PIL import Image",
            "from sentence_transformers import SentenceTransformer",
            "",
            'clip = SentenceTransformer("clip-ViT-B-32")  # multilingual: clip-ViT-B-32-multilingual-v1',
            "image_vectors = clip.encode([Image.open(p) for p in image_paths], normalize_embeddings=True)",
            "# Store in a separate collection; query with clip.encode([\"text query\"]) for text-to-image search",
        ])
        return rec

    @staticmethod
    def _spreadsheet(hosted: bool, gpu: bool, answers: UserAnswers) -> ModalityRecommendation:
        return ModalityRecommendation(
            strategy="Treat data tables as structured data (SQL/DataFrame), serialize small lookup tables as text",
            ingestion=[
                "Load each sheet with pandas/openpyxl; detect header rows and merged cells",
                "Large fact tables: load into DuckDB/SQLite and answer with Text-to-SQL, "
                "not embeddings",
                "Small reference tables (< a few hundred rows): serialize each row as "
                "'column: value' pairs with the sheet name and header as context",
                "Keep formulas' computed values, not the formulas; capture sheet/row/column "
                "as metadata",
            ],
            tools_local=["pandas + openpyxl", "DuckDB for Text-to-SQL over sheets", "docling (tables to Markdown)"],
            tools_hosted=["Vanna.ai / LangChain SQL agents for Text-to-SQL"],
            embedding="Text embedding of serialized rows; none for tables handled via SQL",
            chunking="Row-based (header + N rows per chunk) for serialized tables",
            pip_packages=["pandas", "openpyxl"],
            notes=["Embedding raw spreadsheet rows loses numeric semantics; aggregate questions need SQL"],
            warnings=[],
            code_snippet="\n".join([
                "# pip install pandas openpyxl",
                "import pandas as pd",
                "",
                "def sheet_to_chunks(path: str, rows_per_chunk: int = 20) -> list[dict]:",
                "    chunks = []",
                "    for sheet, df in pd.read_excel(path, sheet_name=None).items():",
                "        df = df.dropna(how=\"all\").astype(str)",
                "        for start in range(0, len(df), rows_per_chunk):",
                "            block = df.iloc[start : start + rows_per_chunk]",
                "            lines = [\"; \".join(f\"{c}: {v}\" for c, v in row.items()) for _, row in block.iterrows()]",
                "            chunks.append({\"text\": f\"Sheet {sheet}\\n\" + \"\\n\".join(lines),",
                "                           \"metadata\": {\"file\": path, \"sheet\": sheet, \"row_start\": int(start)}})",
                "    return chunks",
            ]),
        )

    @staticmethod
    def _presentation(hosted: bool, gpu: bool, answers: UserAnswers) -> ModalityRecommendation:
        return ModalityRecommendation(
            strategy="One chunk per slide: title + body text + speaker notes",
            ingestion=[
                "Extract slide text and notes with python-pptx; keep slide number as metadata",
                "Diagram-heavy slides: render to PNG and caption with a vision model, "
                "append the caption to the slide chunk",
                "Prefix each chunk with the deck title so isolated slides stay interpretable",
            ],
            tools_local=["python-pptx", "LibreOffice headless (pptx -> pdf -> images)", "docling"],
            tools_hosted=["Claude / GPT-4o vision for slide image captions"],
            embedding="Text embedding of slide chunks",
            chunking="Per slide; merge very short consecutive slides",
            pip_packages=["python-pptx"],
            notes=[],
            warnings=["Slides are terse; retrieval quality depends heavily on speaker notes and captions"],
        )

    @staticmethod
    def _video(hosted: bool, gpu: bool, answers: UserAnswers) -> ModalityRecommendation:
        rec = ModalityRecommendation(
            strategy="Transcribe speech, caption key frames, chunk by time window",
            ingestion=[
                "Extract audio (ffmpeg) and transcribe with word/segment timestamps",
                "Chunk transcripts into 30-60 second windows with a small overlap; store "
                "start/end timestamps and a deep link as metadata",
                "Detect scene changes (ffmpeg / PySceneDetect), caption key frames with a "
                "vision model and attach captions to the overlapping transcript chunk",
            ],
            tools_local=["faster-whisper (CTranslate2) for transcription", "ffmpeg + PySceneDetect", "Qwen2.5-VL for frame captions"],
            tools_hosted=["OpenAI Whisper API", "AssemblyAI / Deepgram", "Twelve Labs (native video search)"],
            embedding="Text embedding of transcript+caption chunks",
            chunking="Time-based windows (30-60 s) aligned to sentence boundaries",
            pip_packages=["faster-whisper"],
            notes=["Return timestamps in answers so users can jump to the moment"],
            warnings=[],
        )
        if not gpu:
            rec.warnings.append("Transcription on CPU runs at roughly real time; plan batch processing")
        return rec

    @staticmethod
    def _audio(hosted: bool, gpu: bool, answers: UserAnswers) -> ModalityRecommendation:
        return ModalityRecommendation(
            strategy="Transcribe with timestamps and speaker labels, chunk by segment",
            ingestion=[
                "Transcribe with faster-whisper; keep segment timestamps",
                "Add speaker diarization (pyannote.audio) for meetings and calls",
                "Chunk by speaker turn or 30-60 s window; store timestamps as metadata",
            ],
            tools_local=["faster-whisper", "pyannote.audio (diarization)"],
            tools_hosted=["OpenAI Whisper API", "AssemblyAI / Deepgram (transcription + diarization)"],
            embedding="Text embedding of transcript chunks",
            chunking="Speaker-turn or time-window based (see speaker_split strategy)",
            pip_packages=["faster-whisper"],
            notes=[],
            warnings=[],
        )

    @staticmethod
    def _cad(hosted: bool, gpu: bool, answers: UserAnswers) -> ModalityRecommendation:
        return ModalityRecommendation(
            strategy="Extract structured metadata first; render drawings to images for visual description",
            ingestion=[
                "DXF: parse with ezdxf - text entities, layer names, block names, dimensions "
                "and the title block become a structured record per drawing",
                "DWG: convert to DXF with the ODA File Converter (free) before parsing",
                "IFC/BIM: parse with ifcopenshell into elements + properties; query these "
                "with filters or SQL rather than embeddings",
                "Render each sheet to PNG and caption it with a vision model to capture "
                "what the geometry shows; index caption + extracted text together",
            ],
            tools_local=["ezdxf (DXF)", "ifcopenshell (IFC)", "ODA File Converter (DWG -> DXF)", "Qwen2.5-VL for drawing captions"],
            tools_hosted=["Autodesk Platform Services (Model Derivative API)", "Claude / GPT-4o vision"],
            embedding="Text embedding of the structured record + caption; metadata filters on project/discipline/revision",
            chunking="One chunk per drawing sheet (or per IFC storey/system)",
            pip_packages=["ezdxf"],
            notes=["Blueprint questions are usually lookups (which drawing shows X, what revision) - metadata filtering matters more than semantic similarity"],
            warnings=["Geometry itself is not searchable via text embeddings; rely on titles, annotations and captions"],
        )

    @staticmethod
    def _other(hosted: bool, gpu: bool, answers: UserAnswers) -> ModalityRecommendation:
        return ModalityRecommendation(
            strategy="Unrecognised file types: decide per type whether to convert, skip or index metadata only",
            ingestion=["Review the extensions listed; convert convertible formats to PDF/text, otherwise index file names and paths as metadata"],
            tools_local=["LibreOffice headless for office formats", "pandoc for markup formats"],
            tools_hosted=[],
            embedding="n/a",
            chunking="n/a",
            pip_packages=[],
            notes=[],
            warnings=[],
        )


# ── Extension map (kept in sync with the document analyzer) ─────────────────

_IMAGE = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".svg", ".heic"}
_VIDEO = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv"}
_AUDIO = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma"}
_SPREADSHEET = {".xlsx", ".xlsm", ".xls", ".ods", ".numbers"}
_PRESENTATION = {".pptx", ".ppt", ".odp", ".key"}
_CAD = {".dwg", ".dxf", ".dgn", ".ifc", ".rvt", ".step", ".stp", ".iges", ".igs", ".skp", ".3dm"}


def _modality_of_extension(ext: str) -> str:
    ext = ext.lower()
    if ext in _IMAGE:
        return "image"
    if ext in _VIDEO:
        return "video"
    if ext in _AUDIO:
        return "audio"
    if ext in _SPREADSHEET:
        return "spreadsheet"
    if ext in _PRESENTATION:
        return "presentation"
    if ext in _CAD:
        return "cad"
    if ext == ".pdf":
        return "scanned_pdf"  # only used for extension listing of that pseudo-modality
    return "other"
