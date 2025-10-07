# app.py
# Streamlit + OpenAI Responses API with the built-in file_search tool.
# - Create / reuse a Vector Store on OpenAI
# - Upload files directly to that Vector Store (SDK)
# - Chat with those files via the Responses API

import os
import json
from typing import Dict, List, Any

import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="File Search (Responses API) — Chat with your files", page_icon="🗂️", layout="wide")
st.title("🗂️ Chat with your files (OpenAI Vector Stores + Responses API)")
st.caption("Uploads → OpenAI Vector Store → Responses API with the built-in **file_search** tool.")

# ---------------- Sidebar: Config ----------------
with st.sidebar:
    st.subheader("Configuration")
    api_key = st.text_input("OpenAI API Key", type="password", help="Or set OPENAI_API_KEY env var.")
    model = st.selectbox("Model", ["gpt-4.1", "gpt-4o", "gpt-4o-mini"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.2, 0.1)
    max_results = st.slider("Max retrieved chunks", 1, 30, 20, 1)
    st.markdown("---")

    st.subheader("Vector Store")
    use_existing = st.checkbox("Use existing Vector Store ID", value=False)
    vs_name = st.text_input("New Vector Store name", value="My Docs") if not use_existing else None
    vs_id_input = st.text_input("Existing Vector Store ID") if use_existing else None

    st.markdown("---")
    if st.button("Clear session"):
        for k in ["vector_store_id", "uploads", "messages", "last_response_id", "file_name_by_id"]:
            st.session_state.pop(k, None)
        st.success("Session cleared.")

# ---------------- Session init ----------------
st.session_state.setdefault("vector_store_id", None)
st.session_state.setdefault("uploads", [])  # list of (filename, size, file_id)
st.session_state.setdefault("messages", [])  # [{"role": "user"/"assistant", "content": str}]
st.session_state.setdefault("last_response_id", None)
st.session_state.setdefault("file_name_by_id", {})  # file_id -> filename

client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

# ---------------- Vector Store helpers ----------------
def ensure_vector_store() -> str:
    """Create a new vector store (if needed) or use the provided existing id."""
    if use_existing:
        if not vs_id_input:
            raise RuntimeError("Please provide an existing Vector Store ID.")
        st.session_state["vector_store_id"] = vs_id_input.strip()
        return st.session_state["vector_store_id"]

    if st.session_state.get("vector_store_id"):
        return st.session_state["vector_store_id"]

    vs = client.vector_stores.create(name=vs_name or "My Docs")
    st.session_state["vector_store_id"] = vs.id
    return vs.id

def upload_files_to_vector_store(vs_id: str, files: List[Any]) -> List[Dict[str, str]]:
    """
    Upload files via Files API, then attach to the vector store.
    Uses create_and_poll to wait for indexing.
    """
    uploaded_info = []
    for f in files:
        # Streamlit UploadedFile -> bytes
        file_bytes = f.getvalue()
        # 1) Upload to Files API
        up = client.files.create(
            file=(f.name, file_bytes),
            purpose="assistants",  # required for use with file_search/vector stores
        )
        # 2) Attach to vector store and poll until processed
        client.vector_stores.files.create_and_poll(
            vector_store_id=vs_id,
            file_id=up.id,
        )
        st.session_state["file_name_by_id"][up.id] = f.name
        uploaded_info.append({"file_id": up.id, "filename": f.name, "bytes": len(file_bytes)})
    return uploaded_info

# ---------------- Upload UI ----------------
st.subheader("1) Upload files to your Vector Store")
uploads = st.file_uploader(
    "Drop PDFs, DOCX, TXT, MD, CSV, etc.",
    accept_multiple_files=True,
)

colA, colB = st.columns([1,1])
with colA:
    if st.button("Create (or select) Vector Store"):
        try:
            vs_id = ensure_vector_store()
            st.success(f"Vector Store ready: `{vs_id}`")
        except Exception as e:
            st.error(f"Couldn't get Vector Store: {e}")

with colB:
    if st.button("Upload → Index to Vector Store", type="primary", disabled=not uploads):
        try:
            vs_id = ensure_vector_store()
            info = upload_files_to_vector_store(vs_id, uploads)
            st.session_state["uploads"].extend([(i["filename"], i["bytes"], i["file_id"]) for i in info])
            st.success(f"Uploaded & indexed {len(info)} file(s) to Vector Store `{vs_id}`.")
        except Exception as e:
            st.exception(e)

# Show current state
if st.session_state["vector_store_id"]:
    st.info(f"Active Vector Store ID: `{st.session_state['vector_store_id']}`")
if st.session_state["uploads"]:
    st.caption("Uploaded files in this session:")
    for (name, size, fid) in st.session_state["uploads"]:
        st.write(f"• {name} ({size} bytes) — file_id: `{fid}`")

# ---------------- Chat UI ----------------
st.subheader("2) Chat over your files")

# render history
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_q = st.chat_input("Ask something about your uploaded files…")
if user_q:
    # Show user message
    st.session_state["messages"].append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    # guardrails
    if not st.session_state.get("vector_store_id"):
        with st.chat_message("assistant"):
            st.warning("Please create/select a Vector Store and upload at least one file first.")
    else:
        # Call Responses API with the built-in file_search tool referencing our Vector Store
        try:
            with st.chat_message("assistant"):
                placeholder = st.empty()

                kwargs = {
                    "model": model,
                    "input": user_q,   # simplest form; state kept via previous_response_id below
                    "temperature": temperature,
                    "tools": [{
                        "type": "file_search",
                        "vector_store_ids": [st.session_state["vector_store_id"]],
                        "max_num_results": max_results
                    }],
                }

                resp = client.responses.create(**kwargs)

                # Save last response id for stateful convo
                st.session_state["last_response_id"] = resp.id

                # Helpers to extract text + citations
                def response_to_text(r) -> str:
                    # Try the convenience property first
                    text = getattr(r, "output_text", None)
                    if text:
                        return text
                    # Fallback: stitch text from output array
                    chunks = []
                    for item in getattr(r, "output", []) or []:
                        if getattr(item, "type", None) == "message":
                            for c in getattr(item, "content", []) or []:
                                if getattr(c, "type", None) == "output_text":
                                    chunks.append(getattr(c, "text", ""))
                    return "\n".join(chunks).strip() or "(no text)"

                def extract_file_citations(r) -> List[str]:
                    names = []
                    for item in getattr(r, "output", []) or []:
                        if getattr(item, "type", None) == "message":
                            for c in getattr(item, "content", []) or []:
                                ann = getattr(c, "annotations", None) or []
                                for a in ann:
                                    t = getattr(a, "type", None)
                                    # annotations may be dicts on some SDK versions
                                    if isinstance(a, dict):
                                        t = a.get("type")
                                        if t in ("file_citation", "vector_store_citation"):
                                            fid = a.get("file_id")
                                            if fid:
                                                names.append(st.session_state["file_name_by_id"].get(fid, fid))
                                    else:
                                        if t in ("file_citation", "vector_store_citation"):
                                            fid = getattr(a, "file_id", None)
                                            if fid:
                                                names.append(st.session_state["file_name_by_id"].get(fid, fid))
                    # de-dupe & keep order
                    seen, out = set(), []
                    for n in names:
                        if n not in seen:
                            out.append(n); seen.add(n)
                    return out

                text = response_to_text(resp)
                cites = extract_file_citations(resp)
                if cites:
                    text = text.rstrip() + "\n\n— **Sources:** " + "; ".join(cites)

                placeholder.markdown(text)
                st.session_state["messages"].append({"role": "assistant", "content": text})

        except Exception as e:
            with st.chat_message("assistant"):
                st.exception(e)
