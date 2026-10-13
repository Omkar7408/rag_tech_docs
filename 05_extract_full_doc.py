import pymupdf as fitz
import pdfplumber

file_path = "data/samsung.pdf"  # your real filename

def extract_full_document(file_path):
    doc_content = []

    fitz_doc = fitz.open(file_path)

    with pdfplumber.open(file_path) as pdf:
        for i in range(len(pdf.pages)):
            text = fitz_doc[i].get_text()
            tables = pdf.pages[i].extract_tables()

            page_data = {
                "page_num": i + 1,
                "text": text.strip(),
                "tables": tables
            }
            doc_content.append(page_data)

    fitz_doc.close()
    return doc_content

pages = extract_full_document(file_path)
print(f"Extracted {len(pages)} pages")
print(f"Page 4 has {len(pages[3]['tables'])} table(s)")