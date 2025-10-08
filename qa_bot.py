# app.py
# Streamlined UX for OpenAI Vector Stores + Responses API (file_search)
# Flow:
#   0) Ask for API key (block UI until provided)
#   A) Use existing store: list -> select -> (optional) upload files -> chat
#   B) Create new store -> upload files -> chat
#
# pip install streamlit openai>=1.50.0

import os
from typing import List, Dict, Any, Optional

import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Chat with Files (Vector Stores + Responses API)", page_icon="🗂️", layout="wide")
st.title("🗂️ Chat with your files")
st.caption("OpenAI Vector Stores + Responses API with the built-in **file_search** tool. Clean, intuitive flow.")

# ------------------ 0) API Key (blocker) ------------------
with st.sidebar:
    st.subheader("Authentication")
    api_key = st.text_input("OpenAI API Key", type="password", help="Required to proceed")
    st.markdown("---")
    st.subheader("Model & Retrieval")
    model = st.selectbox("Model", ["gpt-5", "gpt-4o", "gpt-4o-mini"], index=0)
    max_results = st.slider("Max retrieved chunks", 1, 30, 20, 1)
    stream_output = st.checkbox("Stream output", value=True)

if not (api_key or os.getenv("OPENAI_API_KEY")):
    st.warning("Enter your OpenAI API key in the left sidebar to begin.")
    st.stop()

client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

# ------------------ Helpers ------------------
def list_vector_stores(limit: int = 100) -> List[Any]:
    """Return vector stores (best effort). Some SDK versions may have list bugs; we handle gracefully."""
    try:
        page = client.vector_stores.list(limit=limit)
        items = list(getattr(page, "data", []) or [])
        return items
    except Exception as e:
        st.info(f"Couldn't list vector stores automatically: {e}")
        return []

def list_store_files(vs_id: str, limit: int = 100) -> List[Any]:
    try:
        page = client.vector_stores.files.list(vector_store_id=vs_id, limit=limit)
        return list(getattr(page, "data", []) or [])
    except Exception as e:
        st.warning(f"Couldn't list files for store {vs_id}: {e}")
        return []

def upload_files_to_vector_store(vs_id: str, files: List[Any]) -> List[Dict[str, str]]:
    """Upload files (purpose=assistants) and attach to vector store; poll until indexed."""
    results = []
    for f in files:
        try:
            file_bytes = f.getvalue()
            up = client.files.create(file=(f.name, file_bytes), purpose="assistants")
            client.vector_stores.files.create_and_poll(vector_store_id=vs_id, file_id=up.id)
            results.append({"file_id": up.id, "filename": f.name, "bytes": len(file_bytes)})
        except Exception as e:
            st.error(f"Upload failed for {getattr(f, 'name', 'file')}: {e}")
    return results

def response_text_and_citations(resp) -> str:
    # Prefer convenience property if present
    text = getattr(resp, "output_text", None)
    if not text:
        chunks = []
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", None) == "message":
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", None) == "output_text":
                        chunks.append(getattr(c, "text", ""))
        text = "\n".join(chunks).strip() or "(no text)"

    # Pull file citations if present
    file_names_by_id = st.session_state.get("file_name_by_id", {})
    cited_names = []
    for item in getattr(resp, "output", []) or []:
        if getattr(item, "type", None) == "message":
            for c in getattr(item, "content", []) or []:
                for ann in (getattr(c, "annotations", None) or []):
                    # Handle both dicts and typed objects across SDK versions
                    t = ann.get("type") if isinstance(ann, dict) else getattr(ann, "type", None)
                    if t in ("file_citation", "vector_store_citation"):
                        fid = ann.get("file_id") if isinstance(ann, dict) else getattr(ann, "file_id", None)
                        if fid:
                            cited_names.append(file_names_by_id.get(fid, fid))
    # dedupe in order
    seen, dedup = set(), []
    for n in cited_names:
        if n not in seen:
            dedup.append(n); seen.add(n)
    if dedup:
        text = text.rstrip() + "\n\n— **Sources:** " + "; ".join(dedup)
    return text

# ------------------ Session ------------------
st.session_state.setdefault("selected_vs", None)
st.session_state.setdefault("file_name_by_id", {})
st.session_state.setdefault("uploads_log", [])   # (filename, bytes, file_id)
st.session_state.setdefault("last_response_id", None)
st.session_state.setdefault("messages", [])

# ------------------ 1) Choose Flow ------------------
st.header("1) Pick a path")
tab_existing, tab_new = st.tabs(["Use existing Vector Store", "Create new Vector Store"])

# ---------- Flow A: Use existing ----------
with tab_existing:
    st.subheader("A. Select an existing Vector Store")

    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("🔄 Refresh list"):
            st.session_state["vs_list_cache"] = list_vector_stores()
    with col2:
        manual_vs = st.text_input("...or paste a Vector Store ID (fallback)")

    vs_list = st.session_state.get("vs_list_cache")
    if vs_list is None:
        vs_list = list_vector_stores()
        st.session_state["vs_list_cache"] = vs_list

    vs_names = ["— select —"] + [f"{vs.name or '(unnamed)'} • {vs.id}" for vs in vs_list]
    idx = st.selectbox("Vector Stores", options=range(len(vs_names)), format_func=lambda i: vs_names[i], index=0)

    picked_vs: Optional[str] = None
    if idx > 0:
        picked_vs = vs_list[idx-1].id
    elif manual_vs.strip():
        picked_vs = manual_vs.strip()

    if picked_vs:
        st.success(f"Selected Vector Store: `{picked_vs}`")
        st.session_state["selected_vs"] = picked_vs

        # Files in this store
        st.markdown("**Files in store**")
        files = list_store_files(picked_vs)
        if files:
            for f in files:
                # capture names for citation mapping
                st.session_state["file_name_by_id"][f.id] = getattr(f, "filename", getattr(f, "display_name", f.id))
                status = getattr(f, "status", "unknown")
                size = getattr(f, "usage_bytes", None)
                st.write(f"• {st.session_state['file_name_by_id'][f.id]} — status: `{status}`"
                         + (f" — {size} bytes" if size is not None else ""))
        else:
            st.info("No files yet in this store.")

        # Optional uploads
        st.markdown("**Add files (optional)**")
        new_files = st.file_uploader("Drop more files (PDF/DOCX/TXT/MD/CSV, etc.)", accept_multiple_files=True, key="uploader_existing")
        if st.button("Upload to selected store", type="primary", disabled=not new_files):
            info = upload_files_to_vector_store(picked_vs, new_files)
            for i in info:
                st.session_state["uploads_log"].append((i["filename"], i["bytes"], i["file_id"]))
                st.session_state["file_name_by_id"][i["file_id"]] = i["filename"]
            st.success(f"Uploaded & indexed {len(info)} file(s).")
            st.session_state["vs_list_cache"] = list_vector_stores()  # refresh usage bytes
            st.rerun()

# ---------- Flow B: Create new ----------
with tab_new:
    st.subheader("B. Create a new Vector Store")
    new_name = st.text_input("Vector Store name", value="My Docs")
    create_btn = st.button("Create Vector Store", type="primary")

    if create_btn:
        try:
            vs = client.vector_stores.create(name=new_name or "My Docs")
            st.session_state["selected_vs"] = vs.id
            st.success(f"Created Vector Store: `{vs.id}`")
        except Exception as e:
            st.error(f"Create failed: {e}")

    if st.session_state.get("selected_vs"):
        st.info(f"Active Vector Store: `{st.session_state['selected_vs']}`")
        new_files2 = st.file_uploader("Upload files to this new store", accept_multiple_files=True, key="uploader_new")
        if st.button("Upload to new store", type="primary", disabled=not new_files2):
            info = upload_files_to_vector_store(st.session_state["selected_vs"], new_files2)
            for i in info:
                st.session_state["uploads_log"].append((i["filename"], i["bytes"], i["file_id"]))
                st.session_state["file_name_by_id"][i["file_id"]] = i["filename"]
            st.success(f"Uploaded & indexed {len(info)} file(s).")

# ------------------ 2) Chat ------------------
st.header("2) Chat")
active_vs = st.session_state.get("selected_vs")
if not active_vs:
    st.info("Select or create a Vector Store above to enable chat.")
else:
    # Show prior messages
    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_q = st.chat_input("Ask something about your files…")
    if user_q:
        st.session_state["messages"].append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        try:
            kwargs = {
                "model": model,
                "input": user_q,
                "tools": [{
                    "type": "file_search",
                    "vector_store_ids": [active_vs],
                    "max_num_results": max_results
                }],
            }
            # if st.session_state["last_response_id"]:
            #     kwargs["previous_response_id"] = st.session_state["last_response_id"]

            with st.chat_message("assistant"):
                placeholder = st.empty()

                if stream_output:
                    # Streaming
                    streamed = ""
                    with client.responses.stream(**kwargs) as stream:
                        for event in stream:
                            if event.type == "response.output_text.delta":
                                streamed += event.delta
                                placeholder.markdown(streamed)
                            elif event.type == "response.completed":
                                resp = event.response
                    # If we never got a final object, do a non-stream fallback:
                    if 'resp' not in locals():
                        resp = client.responses.create(**kwargs)
                    st.session_state["last_response_id"] = resp.id
                    text = response_text_and_citations(resp)
                    placeholder.markdown(text)
                    st.session_state["messages"].append({"role": "assistant", "content": text})
                else:
                    # Non-streaming
                    resp = client.responses.create(**kwargs)
                    st.session_state["last_response_id"] = resp.id
                    text = response_text_and_citations(resp)
                    st.markdown(text)
                    st.session_state["messages"].append({"role": "assistant", "content": text})

        except Exception as e:
            with st.chat_message("assistant"):
                st.exception(e)
