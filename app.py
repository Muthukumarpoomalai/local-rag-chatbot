
import os
import streamlit as st

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate

# Streamlit App UI
st.title("🤖 RAG Chatbot (Local Embeddings)")
st.sidebar.title("Upload or paste text")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Upload a TXT or PDF file", type=["txt", "pdf"]
)

# Text input
text_input = st.sidebar.text_area("Or paste text here")

documents = []

# Step 1: Load documents
if uploaded_file is not None:

    temp_path = "temp_uploaded" + os.path.splitext(uploaded_file.name)[1]

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if uploaded_file.name.lower().endswith(".pdf"):
        loader = PyPDFLoader(temp_path)
    else:
        loader = TextLoader(temp_path, encoding="utf-8")

    documents = loader.load()

elif text_input.strip() != "":

    temp_path = "temp_input.txt"

    with open(temp_path, "w", encoding="utf-8") as f:
        f.write(text_input)

    loader = TextLoader(temp_path, encoding="utf-8")
    documents = loader.load()

# Step 2: Build embeddings + vectorstore
if documents:

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(documents, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # Step 3: Use local LLM (Flan-T5)
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    from langchain_community.llms import HuggingFacePipeline

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    # Step 4: Create RetrievalQA chain
    prompt = PromptTemplate(
        template="""Answer the question using only the given context.

Context: {context}

Question: {question}

Answer:""",
        input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    # Step 5: Ask questions
    st.subheader("Ask a question about the text")
    query = st.text_input("Your question:")

    if query:
        response = qa.invoke({"query": query})

        st.write("### Answer:")
        st.write(response["result"])

else:
    st.info("👈 Upload a text file or PDF or paste content to start.")
