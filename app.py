# app.py — Streamlit + Pinecone + OpenAI + Cloudflare R2 (private, lazy downloads)

import os
import json
import re
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Tuple, Optional, Set

import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
from PIL import Image

# --- R2 / S3 client ---
import boto3
from botocore.config import Config

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent
JSON_DIR = BASE_DIR / "yardi_rag_jsons"
IMAGE_DIR = BASE_DIR / "yardi_images"
DOC_INDEX_PATH = BASE_DIR / "doc_index.json"

# Ensure local cache dirs exist (for lazy downloads in the cloud)
JSON_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# Must match uploader split settings
MAX_EMBED_CHARS = 3200
SPLIT_OVERLAP = 180

# ---------- ENV / Clients ----------
load_dotenv()

# R2 environment (private storage)
R2_ACCOUNT_ID        = os.getenv("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID     = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET            = os.getenv("R2_BUCKET")
R2_ENDPOINT = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com" if R2_ACCOUNT_ID else None

s3 = None
if all([R2_ENDPOINT, R2_BUCKET, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY]):
    s3 = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 4})
    )

# OpenAI / Pinecone
PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_HOST      = os.getenv("PINECONE_HOST")  # optional
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")

assert PINECONE_API_KEY and PINECONE_INDEX_NAME and OPENAI_API_KEY, "Missing .env / secrets for Pinecone or OpenAI."

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_HOST) if PINECONE_HOST else pc.Index(PINECONE_INDEX_NAME)

oai = OpenAI(api_key=OPENAI_API_KEY)
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"


# ---------- R2 Helpers ----------
def r2_download(key: str, dest: Path) -> bool:
    """Download one object from R2 to local path. Returns True on success."""
    if not s3:
        return False
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(R2_BUCKET, key, str(dest))
        return True
    except Exception:
        return False


# ---------- Utilities ----------
def split_text(text: str, max_chars: int, overlap: int):
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + max_chars)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def ensure_doc_index() -> dict:
    # 1) Local first
    if DOC_INDEX_PATH.exists():
        try:
            with open(DOC_INDEX_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # 2) Try R2 root: doc_index.json
    if s3 and r2_download("doc_index.json", DOC_INDEX_PATH) and DOC_INDEX_PATH.exists():
        try:
            with open(DOC_INDEX_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # 3) Fallback: build from any local JSONs (cloud might be empty on first run)
    mapping = {}
    for fp in JSON_DIR.glob("*.json"):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                doc = json.load(f)
            did = doc.get("doc_id")
            if did:
                mapping[did] = fp.name
        except Exception:
            continue

    if not mapping and s3:
        st.warning("doc_index.json not found locally or in R2; only locally present JSONs (if any) will be used.")

    if mapping:
        try:
            with open(DOC_INDEX_PATH, "w", encoding="utf-8") as f:
                json.dump(mapping, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    return mapping


DOC_INDEX = ensure_doc_index()


def _get_json_path(rel_name: str) -> Path:
    """Return local path for a JSON, fetching from R2 if missing."""
    local = JSON_DIR / rel_name
    if local.exists():
        return local
    # Try to fetch from R2 under yardi_rag_jsons/
    if s3 and r2_download(f"yardi_rag_jsons/{rel_name}", local):
        return local
    return local  # may not exist; caller will handle


@lru_cache(maxsize=256)
def load_doc_by_id(doc_id: str) -> Optional[dict]:
    rel = DOC_INDEX.get(doc_id)
    if not rel:
        return None
    path = _get_json_path(rel)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_vector_id(vec_id: str):
    parts = vec_id.split("::")
    if len(parts) < 2:
        return None, None, None
    doc_id = parts[0]
    chunk_id = parts[1]
    split_idx = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else 0
    return doc_id, chunk_id, split_idx


def get_complete_document_data(doc: dict, chunk_id: str) -> Dict:
    """Extract complete information including all steps and their associated images."""
    result = {
        "sections": [],
        "all_images": set(),
        "steps_with_images": []
    }

    sections = doc.get("sections", [])
    chunks = doc.get("chunks", [])

    relevant_chunk = None
    for chunk in chunks:
        if chunk.get("id") == chunk_id:
            relevant_chunk = chunk
            break

    if not relevant_chunk:
        return result

    # Gather images from the chunk
    chunk_images = relevant_chunk.get("image_rel_paths", []) or []
    for img in chunk_images:
        result["all_images"].add(img)

    # Match sections via breadcrumb
    chunk_breadcrumb = relevant_chunk.get("breadcrumb", "") or ""
    for section in sections:
        section_breadcrumb = section.get("breadcrumb", "") or ""
        if chunk_breadcrumb and section_breadcrumb and (
            chunk_breadcrumb in section_breadcrumb or section_breadcrumb in chunk_breadcrumb
        ):
            result["sections"].append(section)

            # Steps with images
            for step in section.get("steps", []):
                step_data = {
                    "text": step.get("text", ""),
                    "label": step.get("label", ""),
                    "images": step.get("image_rel_paths", []) or [],
                    "substeps": step.get("substeps", []) or [],
                }
                result["steps_with_images"].append(step_data)

                for img in step_data["images"]:
                    result["all_images"].add(img)

                # Substep images
                for sub in step_data["substeps"]:
                    for simg in sub.get("image_rel_paths", []) or []:
                        result["all_images"].add(simg)

    return result


def reconstruct_chunk_text(doc: dict, chunk_id: str, split_idx: int) -> Tuple[str, Dict]:
    """Returns text and complete document data including steps and images."""
    chunks = doc.get("chunks", []) or []
    for ch in chunks:
        if ch.get("id") == chunk_id:
            base = ch.get("vector_text") or ch.get("text") or ""
            parts = split_text(base, MAX_EMBED_CHARS, SPLIT_OVERLAP)
            doc_data = get_complete_document_data(doc, chunk_id)
            if 0 <= split_idx < len(parts):
                return parts[split_idx], doc_data
            return base, doc_data
    return "", {}


def embed_query(text: str) -> List[float]:
    resp = oai.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding


def retrieve(query: str, top_k: int = 6):
    qvec = embed_query(query)
    res = index.query(vector=qvec, top_k=top_k, include_metadata=True)
    return res["matches"] if isinstance(res, dict) else res.matches


def build_enhanced_context(matches) -> Tuple[str, Dict[str, Dict], Set[str]]:
    """Build context with complete information including all steps and images."""
    blocks = []
    all_doc_data: Dict[str, Dict] = {}
    all_images: Set[str] = set()

    for m in matches or []:
        mid = m["id"] if isinstance(m, dict) else m.id
        meta = m.get("metadata") if isinstance(m, dict) else m.metadata
        score = m.get("score") if isinstance(m, dict) else m.score

        doc_id, chunk_id, split_idx = parse_vector_id(mid)
        if not doc_id or not chunk_id:
            continue

        doc = load_doc_by_id(doc_id)
        if not doc:
            continue

        text, doc_data = reconstruct_chunk_text(doc, chunk_id, split_idx)
        title = doc.get("title", "") or ""
        breadcrumb = (meta or {}).get("breadcrumb", "") or ""
        source = (meta or {}).get("source", "") or doc.get("source_rel_path", "") or ""

        # Store complete doc data
        doc_key = f"{doc_id}_{chunk_id}"
        all_doc_data[doc_key] = {
            "title": title,
            "breadcrumb": breadcrumb,
            "doc_data": doc_data,
            "source": source
        }

        # Collect all images
        for img in doc_data.get("all_images", set()):
            all_images.add(img)

        block = f"""[DOC: {title}]
Breadcrumb: {breadcrumb}
Source: {source}
Score: {score:.4f}

{text}"""

        if doc_data.get("steps_with_images"):
            block += f"\n\n[DETAILED_STEPS_AVAILABLE: {doc_key}]"

        blocks.append(block)

    context = "\n\n-----\n\n".join(blocks)
    return context, all_doc_data, all_images


def generate_enhanced_answer(context: str, doc_data: Dict, all_images: Set[str], user_query: str) -> str:
    """Generate answer with proper image placement markers."""
    steps_detail = ""

    for doc_key, data in doc_data.items():
        if data["doc_data"].get("steps_with_images"):
            steps_detail += f"\n\n=== Steps from {data['title']} ===\n"
            for step in data["doc_data"]["steps_with_images"]:
                step_text = step.get("text", "")
                step_label = step.get("label", "")
                step_images = step.get("images", [])
                steps_detail += f"\nStep {step_label}: {step_text}\n"
                for img in step_images:
                    img_name = os.path.basename(img)
                    steps_detail += f"  - Has image: {img_name}\n"
                for substep in step.get("substeps", []):
                    substep_text = substep.get("text", "")
                    substep_label = substep.get("label", "")
                    substep_images = substep.get("image_rel_paths", []) or []
                    steps_detail += f"  {substep_label}) {substep_text}\n"
                    for img in substep_images:
                        img_name = os.path.basename(img)
                        steps_detail += f"    - Has image: {img_name}\n"

    available_images = ", ".join([os.path.basename(img) for img in all_images]) if all_images else "No images available"

    system = """You are an expert Yardi assistant providing comprehensive step-by-step guidance.

CRITICAL FORMATTING INSTRUCTIONS:

1. PRESERVE ALL INFORMATION: Include every detail, note, warning, and step from the source material
2. IMAGE PLACEMENT: Insert {{IMAGE:filename}} exactly after the step that references it
3. HIGHLIGHTING: Use these exact formats:
   - For code/menu paths: `code formatting`
   - For important terms: **bold**
   - Keep light green highlighting by using inline code for: menu selections, button names, field names, tab names
4. STRUCTURE: Maintain exact hierarchical structure from source
5. NOTES: Include all NOTE and IMPORTANT sections with proper formatting

FORMATTING EXAMPLES:
- Menu navigation: Select `Functions > Forecasting Functions > Comm Straight Line Forecast`
- Field names: In the `Book For Straight-Line Rent Book Configuration` field
- Buttons: Click `OK` or Click the `Budget Selection` tab
- Important notes: **NOTE:** Before you calculate...

Remember: The light green highlighting effect is achieved through inline code formatting with backticks."""

    user = f"""User question:
{user_query}

Context from documentation:
{context}

Detailed steps with images:
{steps_detail}

All available images: {available_images}

IMPORTANT:
- Place each image IMMEDIATELY after its corresponding step using {{IMAGE:filename}}
- Use `backticks` for all UI elements to maintain light green highlighting
- Include ALL steps, substeps (a, b, c), notes, and warnings
- Preserve the exact structure from the documentation"""

    resp = oai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=0.1,
        max_tokens=2000,
    )

    answer = resp.choices[0].message.content.strip()
    return answer


def render_answer_with_inline_images(answer: str, image_dir: Path):
    """Render the answer with inline images and proper formatting."""

    # Normalize headings
    answer = re.sub(r'^# (.+)$', r'### \1', answer, flags=re.MULTILINE)
    answer = re.sub(r'^## (.+)$', r'### \1', answer, flags=re.MULTILINE)
    answer = re.sub(r'^### (.+)$', r'### \1', answer, flags=re.MULTILINE)

    # NOTE sections -> styled div
    def replace_note(match):
        note_text = match.group(1).strip()
        return f'<div class="note-section"><strong>NOTE:</strong> {note_text}</div>'

    answer = re.sub(
        r'\*\*NOTE:\*\*\s*((?:[^\n]|\n(?!\n))+?)(?=\n\n|\n\d+\.|\Z)',
        replace_note,
        answer,
        flags=re.DOTALL
    )

    # IMPORTANT sections -> styled div
    def replace_important(match):
        important_text = match.group(1).strip()
        return f'<div class="important-section"><strong>IMPORTANT:</strong> {important_text}</div>'

    answer = re.sub(
        r'\*\*IMPORTANT:\*\*\s*((?:[^\n]|\n(?!\n))+?)(?=\n\n|\n\d+\.|\Z)',
        replace_important,
        answer,
        flags=re.DOTALL
    )

    # Split by image placeholders
    pattern = r'\{\{IMAGE:([^}]+)\}\}'
    parts = re.split(pattern, answer)

    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Text part
            if part.strip():
                st.markdown(part, unsafe_allow_html=True)
        else:
            # Image filename
            img_name = part.strip()
            img_path = image_dir / img_name

            # If not cached locally, try R2
            if not img_path.exists() and s3:
                r2_download(f"yardi_images/{img_name}", img_path)

            if img_path.exists():
                img = Image.open(img_path)
                width, _ = img.size
                max_width = min(800, width)

                # Responsive layout
                if width < 400:
                    col1, col2, col3 = st.columns([1.5, 2, 1.5])
                    with col2:
                        st.image(str(img_path), caption=f"📸 {img_name}", width=width)
                elif width < 600:
                    col1, col2, col3 = st.columns([1, 3, 1])
                    with col2:
                        st.image(str(img_path), caption=f"📸 {img_name}", width=max_width)
                else:
                    col1, col2, col3 = st.columns([0.5, 4, 0.5])
                    with col2:
                        st.image(str(img_path), caption=f"📸 {img_name}", width=max_width)
                st.markdown("")
            else:
                st.error(f"⚠️ Image not found: {img_name}")


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Yardi Chatbot", page_icon="🦖", layout="wide")

# Custom CSS
st.markdown("""
<style>
.stChatMessage .stMarkdown,
.stChatMessage .stMarkdown p,
.stChatMessage .stMarkdown div,
.stChatMessage .stMarkdown span { font-size: 1rem !important; line-height: 1.6 !important; }

.stChatMessage h1, .stChatMessage h2, .stChatMessage h3,
.stChatMessage h4, .stChatMessage h5, .stChatMessage h6 {
  font-size: 1.2rem !important; font-weight: 600 !important;
  margin-top: 1rem !important; margin-bottom: 0.5rem !important; line-height: 1.4 !important;
}

.stChatMessage strong, .stChatMessage b { font-size: inherit !important; font-weight: 600 !important; }

code { background-color: rgba(110,168,126,0.2) !important; padding: 2px 6px !important; border-radius: 3px !important; font-size: 0.9em !important; }

.stMarkdown ol li, .stMarkdown ul li { margin-bottom: 8px; font-size: 1rem !important; line-height: 1.6 !important; }

.stImage { border: 1px solid #e0e0e0; border-radius: 8px; padding: 8px; background-color: #fafafa;
           box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin: 10px 0; }
.stImage img { max-width: 100%; height: auto; }

.stChatMessage { padding: 15px; }

.note-section { background-color: #f0f0f0; padding: 12px; border-radius: 5px; margin: 10px 0;
                border-left: 4px solid #6c757d; display: block; font-size: 1rem !important; }
.important-section { background-color: #fff3cd; padding: 12px; border-radius: 5px; margin: 10px 0;
                     border-left: 4px solid #ffc107; display: block; font-size: 1rem !important; }

.stChatMessage * { text-transform: none !important; letter-spacing: normal !important; }
</style>
""", unsafe_allow_html=True)

st.title("🦖 Yardi Chatbot")
st.markdown("*V1.0 for testing. Created by Vaidelis*")
st.markdown("---")

# Initialize session state
if "history" not in st.session_state:
    st.session_state["history"] = []

# Chat input
user_input = st.chat_input("Ask a Yardi question…")

if user_input:
    with st.spinner("🔍 Searching documentation and preparing comprehensive response..."):
        try:
            matches = retrieve(user_input, top_k=6)
        except Exception as e:
            st.error(f"Pinecone retrieval error: {e}")
            matches = []

        context, doc_data, all_images = build_enhanced_context(matches)
        answer = generate_enhanced_answer(context, doc_data, all_images, user_input)

        st.session_state.history.append({
            "user": user_input,
            "bot": answer
        })

# Render conversation history
for turn in st.session_state.history:
    with st.chat_message("user", avatar="🙋"):
        st.markdown(turn["user"])

    with st.chat_message("assistant", avatar="🦖"):
        render_answer_with_inline_images(turn["bot"], IMAGE_DIR)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.caption("🔧 **TealSpark Yardi Assistant** - Enhanced Edition")
    st.caption("📚 Complete documentation with images")
