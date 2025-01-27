import streamlit as st
import os
from pdf2image import convert_from_path
import pytesseract
import tempfile
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

pytesseract.pytesseract.tesseract_cmd = "Add path of the tesseract.exe file"
poppler_path = "Add path of poppler.exe file"

def get_pdftext(uploaded_file):
    """
    Extract text from a single uploaded PDF file.
    """
    extracted_text = ""

    try:
        # Step 1: Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        # Step 2: Attempt direct text extraction
        reader = PdfReader(temp_path)
        for page in reader.pages:
            extracted_text += page.extract_text() or ""

        # Step 3: If direct text extraction fails, use OCR for image-based text
        if not extracted_text.strip():
            print("No direct text found. Performing OCR...")
            extracted_text = perform_ocr(temp_path)
        else:
            print("Direct text extraction succeeded.")
    except Exception as e:
        print(f"Error during text extraction: {e}")
        return None

    return extracted_text


def perform_ocr(pdf_path):
    ocr_text = ""
    try:
        # Convert PDF to images
        images = convert_from_path(pdf_path, poppler_path=poppler_path)

        # Perform OCR on each image and specify the language as 'eng' for English
        for img in images:
            ocr_text += pytesseract.image_to_string(img, lang='eng')
        
    except Exception as e:
        print(f"Error during OCR: {e}")
    
    return ocr_text


def get_chunks(text):
    if text is None:
        raise ValueError("Input text is None")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


def get_embeddings(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore


def user_query(question):
    response = st.session_state.conversation({"question": question})
    st.session_state.chat_history = response["chat_history"]
    st.write(response["answer"])


def rag_pipeline(vectorstore):
    llm = GoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return chain


def main():
    load_dotenv()
    os.environ["GOOGLE_API_KEY"] = "Add you API Key"
    st.set_page_config(
        page_title="Chat with Doc",
        page_icon="📚",
        layout="wide"
    )

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.title("📚 CHAT WITH DOC")
    st.markdown("Upload your PDF documents and ask questions about their content!")

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        user_query(user_question)

    with st.sidebar:
        st.subheader("Upload your PDF documents")
        uploaded_files = st.file_uploader("Upload your PDFs", type=["pdf"], accept_multiple_files=True)
        
        if st.button("Process") and uploaded_files:
            with st.spinner("Processing..."):
                all_text = ""
                # Process each file separately
                for uploaded_file in uploaded_files:
                    # Save the uploaded file to a temporary file
                    text = get_pdftext(uploaded_file)
                    if text:
                        all_text += text + "\n"
                    else:
                        st.error(f"Failed to extract text from {uploaded_file.name}")

                if all_text:
                    chunks = get_chunks(all_text)
                    vectorstore = get_embeddings(chunks)
                    st.session_state.conversation = rag_pipeline(vectorstore)
                    st.success("Documents processed successfully!")
                else:
                    st.error("No data loaded")

main()