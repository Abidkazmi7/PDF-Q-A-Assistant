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

pdf_path = st.sidebar.text_input(
    "Enter PDF file path (backend accessible path):"
)

if st.sidebar.button("Load PDF"):
    if not pdf_path:
        st.sidebar.error("Please enter a PDF path")
    else:
        with st.spinner("Loading PDF..."):
            try:
                res = requests.post(
                    f"{API_BASE}/load-pdf",
                    params={"pdf_path": pdf_path}
                )

                if res.status_code == 200:
                    data = res.json()
                    st.session_state.pdf_loaded = True
                    st.session_state.pdf_path = pdf_path

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