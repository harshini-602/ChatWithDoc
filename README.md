# Chat with Doc

Chat with Doc is a smart PDF assistant that extracts and answers user queries based on the content of PDF documents. It uses OCR (Optical Character Recognition) to handle image-based text, making it versatile for various document types.

---

## Features
- **Text Extraction**: Extracts text content from PDF files.
- **OCR Integration**: Reads text from images embedded in PDFs.
- **Query Response**: Provides precise answers to user queries based on the document content.
- **User-Friendly**: Simple and intuitive interface for seamless interaction.
- **Document Types Supported**: Can process .pdf, .doc, .docx, .csv files
---

## How It Works
1. Upload a PDF file.
2. The system processes the document, extracting text and using OCR for image-based content.
3. Ask questions about the document, and the assistant provides accurate answers.

---

## Technologies Used
- **Programming Language**: Python
- **OCR Tool**: Tesseract OCR
- **Frameworks/Libraries**:
  - PyPDF2
  - pytesseract
  

---

## Setup Instructions

### Prerequisites
- Python 3.8+
- Tesseract OCR installed ([Installation Guide](https://github.com/tesseract-ocr/tesseract))
- poppler installed ([Installation Guide](https://github.com/oschwartz10612/poppler-windows/releases/))

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/chat-with-doc.git
   cd chat-with-doc
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run ChatWithDoc.py
   ```

---

## Usage
1. Launch the application.
2. Upload a PDF file via the interface.
3. Enter queries in the input field to retrieve relevant information from the document.

---

## Example Queries
- "What is the summary of the document?"
- "Question related to topic present in the document"

---


Thank you for checking out Chat with Doc! 🚀
