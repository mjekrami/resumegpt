import streamlit as st
from optimizer_core import *

st.set_page_config(page_title="Resume Opimizer", layout="wide")
st.title("Resume Opimzer with Akash Chat")

uploaded_file = st.file_uploader(
    "Upload your resume (PDF or DOCX)", type=["pdf", "docx"]
)

if uploaded_file is not None:
    with st.spinner("Extracting text..."):
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif (
            uploaded_file.type
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ):
            text = extract_from_docx(uploaded_file)
        else:
            st.error("Unsupported file type.")
            st.stop()
    st.text_area("Extracted text", value=text, height=300)

    if st.button("Optimze Resume"):
        with st.spinner("Optimizing..."):
            graph = build_optimizer_graph()
            result = graph.invoke(text)
            st.subheader("Opimzed Resume")
            st.code(result, language="markdown")
