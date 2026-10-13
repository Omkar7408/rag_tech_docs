import pymupdf as fitz
import pdfplumber
import re

file_path = "data/samsung.pdf"  # your real filename

def table_to_text(table):
    """Convert a pdfplumber table (list of rows) into a readable text block."""
    lines = []
    for row in table:
        clean_row = [str(cell).replace("\n", " ") if cell else "" for cell in row]
        lines.append(" | ".join(clean_row))
    return "\n".join(lines)

def build_chunks(file_path):
    chunks = []
    fitz_doc = fitz.open(file_path)

    with pdfplumber.open(file_path) as pdf:
        for i in range(len(pdf.pages)):
            text = fitz_doc[i].get_text().strip()
            tables = pdf.pages[i].extract_tables()

            table_texts = [table_to_text(t) for t in tables]

            # combine page text + its tables into ONE chunk (keeps context together)
            combined = text
            if table_texts:
                combined += "\n\n[TABLES ON THIS PAGE]\n" + "\n\n".join(table_texts)

            if combined.strip():
                chunks.append({
                    "page": i + 1,
                    "content": combined
                })

    fitz_doc.close()
    return chunks

chunks = build_chunks(file_path)

print(f"Total chunks: {len(chunks)}")
print("\n---- Sample chunk (page 4, has a table) ----")
for c in chunks:
    if c["page"] == 4:
        print(c["content"])
        break
print(f"Total chunks: {len(chunks)}")

# find and print the first chunk that actually contains a table
for c in chunks:
    if "[TABLES ON THIS PAGE]" in c["content"]:
        print(f"\n---- First chunk with a table (page {c['page']}) ----")
        print(c["content"])
        break    