import streamlit as st
from QAWithPDF.data_ingestion import load_data
from QAWithPDF.embedding import download_gemini_embedding
from QAWithPDF.model_api import load_model

st.set_page_config(page_title="QA with Documents")


@st.cache_resource(show_spinner=False)

def build_query_engine(uploaded_file):
    documents = load_data(uploaded_file)
    model = load_model()
    return download_gemini_embedding(model, documents)


def main():
    st.header("QA with Documents (Information Retrieval)")

    uploaded_file = st.file_uploader("Upload PDF", type="pdf") # accept_multiple_files=True

    if uploaded_file is not None:

        
        with st.spinner("Reading file....."):
            query_engine = build_query_engine(uploaded_file)

        user_question = st.text_input("Ask your question")

        if st.button("Submit & Process") and user_question:
            with st.spinner("Generating answer..."):
                response = query_engine.query(user_question)
                st.write(response.response)


if __name__ == "__main__":
    main()
