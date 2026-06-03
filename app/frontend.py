import streamlit as st
import requests

# Config
API_BASE = "http://localhost:8000"
st.set_page_config(
    page_title="PDF Q&A Assistant",
    layout="wide"
)

# Session state
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False

if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None

# Sidebar - PDF Loader
st.sidebar.title("📄 PDF Loader")

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if st.sidebar.button("Load PDF"):
    if uploaded_file is None:
        st.sidebar.error("Please upload a PDF")
    else:
        with st.spinner("Loading PDF..."):
            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),       # raw PDF bytes
                        "application/pdf"
                    )
                }

                res = requests.post(
                    f"{API_BASE}/load-pdf",
                    files=files
                )

                if res.status_code == 200:
                    data = res.json()
                    st.session_state.pdf_loaded = True
                    st.session_state.pdf_path = uploaded_file.name

                    st.sidebar.success(
                        f"Loaded! Chunks: {data['chunks']}"
                    )
                else:
                    st.sidebar.error(res.text)

            except Exception as e:
                st.sidebar.error(str(e))

# Main UI
st.title("📚 PDF Q&A Assistant")

if not st.session_state.pdf_loaded:
    st.warning("Please load a PDF first from the sidebar.")
    st.stop()

st.success(f"PDF Loaded: {st.session_state.pdf_path}")

# Question input
question = st.text_input("Ask a question from your PDF:")

if st.button("Get Answer"):
    if not question:
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Searching relevant chunks..."):
        try:
            res = requests.post(
                f"{API_BASE}/query",
                json={"question": question}
            )

            if res.status_code != 200:
                st.error(res.text)
                st.stop()

            data = res.json()

        except Exception as e:
            st.error(str(e))
            st.stop()

    # Results
    st.subheader("🔎 Top Matching Chunks")

    for i, chunk in enumerate(data["top_chunks"], 1):
        with st.expander(f"Chunk {i}"):
            st.write(chunk)

    st.subheader("💡 Top Matching Sentences")

    for i, item in enumerate(data["answer_sentences"], 1):
        st.markdown(f"### Result {i}")

        # sentence (already concatenated prev + curr + next)
        st.info(item["sentence"])

        # score
        if "score" in item:
            st.write(f"**Score:** {item['score']:.4f}")

        st.divider()