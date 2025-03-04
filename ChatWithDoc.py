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
from langchain.vectorstores import FAISS  # Ensure this is used for vectorstore
from langchain_google_genai import GoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import pandas as pd
import matplotlib.pyplot as plt
from docx import Document

pytesseract.pytesseract.tesseract_cmd = r"ADD PATH"
poppler_path = r"ADD PATH"


def process_documents(uploaded_file):
    extracted_text = ""

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        if uploaded_file.name.endswith('.pdf'):
            reader = PdfReader(temp_path)
            for page in reader.pages:
                extracted_text += page.extract_text() or ""

            if not extracted_text.strip():
                print("No direct text found. Performing OCR...")
                extracted_text = perform_ocr(temp_path)
            else:
                print("Direct text extraction succeeded.")
        elif uploaded_file.name.endswith('.doc') or uploaded_file.name.endswith('.docx'):
            extracted_text = extract_text_from_docx(temp_path)
        else:
            st.error(f"Unsupported file type: {uploaded_file.name}")
            return None
    except Exception as e:
        print(f"Error during text extraction: {e}")
        return None

    return extracted_text


def extract_text_from_docx(docx_path):
    text = ""
    try:
        doc = Document(docx_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
    return text


def perform_ocr(pdf_path):
    ocr_text = ""
    try:
        images = convert_from_path(pdf_path, poppler_path=poppler_path)
        for img in images:
            ocr_text += pytesseract.image_to_string(img, lang='eng')
    except Exception as e:
        print(f"Error during OCR: {e}")
    return ocr_text

def process_csv(uploaded_file):
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)
        # Store the DataFrame in the session state for later use
        st.session_state.csv_data = df
        return df
    except Exception as e:
        st.error(f"Error processing CSV file: {uploaded_file.name}")
        st.error(str(e))
        return None



def get_chunks(text):
    if text is None:
        raise ValueError("Input text is None")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


def get_embeddings(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)  # Ensure this is FAISS
    return vectorstore


def user_query(question):
    response = st.session_state.conversation({"question": question})
    st.session_state.chat_history = response["chat_history"]
    st.write(response["answer"])

def rag_pipeline(vectorstore):
    llm = GoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True  # This is already correct
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )


def document_details_tab(uploaded_files):
    if uploaded_files:
        st.markdown("### Document Details:")
        for file in uploaded_files:
            st.write(f"**File Name:** {file.name}")
            st.write(f"**File Size:** {file.size / 1024:.2f} KB")
        if "chunk_count" in st.session_state:
            st.write(f"**Indexed Chunks:** {st.session_state['chunk_count']}")
    else:
        st.warning("No document uploaded.")


def Query_History():
    st.subheader("Query History")
    if "chat_history" in st.session_state and st.session_state.chat_history:
        for entry in st.session_state.chat_history:
            if hasattr(entry, 'type'):  # LangChain message object
                prefix = "Q" if entry.type == "human" else "A"
                st.write(f"**{prefix}:** {entry.content}")
            else:  # Fallback for other formats
                st.write(f"**Q:** {entry.get('question', '')}")
                st.write(f"**A:** {entry.get('answer', '')}")
    else:
        st.write("No queries yet.")


def chat_tab():
    st.subheader("Chat")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        user_query(user_question)


def process_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error processing CSV file: {uploaded_file.name}")
        st.error(str(e))
        return None


def plot_data(df):
    st.write("Select columns to plot:")
    columns = st.multiselect("Columns", df.columns)
    if st.button('Plot'):
        if len(columns) > 1:
            for column in columns:
                if df[column].dtype == 'object':
                    # Categorical data
                    fig, ax = plt.subplots()
                    ax.bar(df[column].value_counts().index, df[column].value_counts().values)
                    ax.set_xlabel('Category')
                    ax.set_ylabel('Count')
                    ax.set_title('Categorical Data')
                    st.pyplot(fig)
                elif df[column].dtype in ['int64', 'float64']:
                    # Numerical data
                    fig, ax = plt.subplots()
                    ax.hist(df[column], bins=50)
                    ax.set_xlabel('Value')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Numerical Data')
                    st.pyplot(fig)
                else:
                    st.warning("Unsupported data type.")
        else:
            column = columns[0]
            if df[column].dtype == 'object':
                # Categorical data
                fig, ax = plt.subplots()
                ax.bar(df[column].value_counts().index, df[column].value_counts().values)
                ax.set_xlabel('Category')
                ax.set_ylabel('Count')
                ax.set_title('Categorical Data')
                st.pyplot(fig)
            elif df[column].dtype in ['int64', 'float64']:
                # Numerical data
                fig, ax = plt.subplots()
                ax.hist(df[column], bins=50)
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                ax.set_title('Numerical Data')
                st.pyplot(fig)
            else:
                st.warning("Unsupported data type.")


def add_csv_tab():
    with st.expander("📊 CSV Data Visualization"):
        if "csv_data" in st.session_state and st.session_state.csv_data is not None:
            st.write("### CSV Data")
            st.dataframe(st.session_state.csv_data)
            plot_data(st.session_state.csv_data)
        else:
            st.warning("No CSV data uploaded.")

def main():
    load_dotenv()
    os.environ["GOOGLE_API_KEY"] = "ADD YOUR API KEY HERE"
    st.set_page_config(
        page_title="Chat with Doc",
        page_icon="📚",
        layout="wide"
    )

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "document_info_list" not in st.session_state:
        st.session_state.document_info_list = []
    if "csv_data" not in st.session_state:
        st.session_state.csv_data = None

    st.title("📚 Chat with Your Documents")
    st.markdown("Upload your PDF, DOC/DOCX, or CSV documents and ask questions about their content!")

    # Tabs with icons for navigation
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📄 Document Details", "🕰️ Query History", "📊 CSV Visualization"])

    # In the sidebar processing block (inside main()):
    with st.sidebar:
        st.subheader("Upload and Manage Documents")
        uploaded_files = st.file_uploader("Document types supported are PDFs, DOCX, or CSVs", type=["pdf", "doc", "docx", "csv"], accept_multiple_files=True)

        if st.button("Process") and uploaded_files:
            with st.spinner("Processing..."):
                for file in uploaded_files:
                    # Handle CSV files separately
                    if file.name.endswith('.csv'):
                        # Process CSV and skip text extraction steps
                        st.session_state.csv_data = process_csv(file)
                    else:
                        # Process text documents (PDF/DOCX)
                        extracted_text = process_documents(file)
                        if isinstance(extracted_text, str):  # Check for valid text
                            chunks = get_chunks(extracted_text)
                            st.session_state.chunk_count = len(chunks)
                            vectorstore = get_embeddings(chunks)
                            st.session_state.conversation = rag_pipeline(vectorstore)
                            st.success("Documents processed successfully!")
                        else:
                            st.error("No data loaded")

    with tab1:
        chat_tab()
    with tab2:
        document_details_tab(uploaded_files)
    with tab3:
        Query_History()
    with tab4:
        add_csv_tab()


if __name__ == "__main__":
    main()
