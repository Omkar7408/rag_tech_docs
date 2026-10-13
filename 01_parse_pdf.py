import pdfplumber

file_path = "data/samsung.pdf"  # your real filename

with pdfplumber.open(file_path) as pdf:
    for i, page in enumerate(pdf.pages):
        tables = page.extract_tables()
        if tables:
            print(f"\n=== Page {i+1} ===")
            for t_idx, table in enumerate(tables):
                print(f"\n--- Table {t_idx+1} ---")
                for row in table:
                    print(row)