import pymupdf as fitz
import pdfplumber
import pytesseract
from PIL import Image
import io

# update this path if your Tesseract install location is different
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

file_path = "data/samsung.pdf"  # your real filename

def extract_images_text(fitz_page, fitz_doc):
    image_texts = []
    for img in fitz_page.get_images(full=True):
        xref = img[0]
        base_image = fitz_doc.extract_image(xref)
        image_bytes = base_image["image"]
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            ocr_text = pytesseract.image_to_string(pil_image).strip()
            if ocr_text:
                image_texts.append(ocr_text)
        except Exception as e:
            print(f"  (skipped one image: {e})")
    return image_texts

def extract_links(fitz_page):
    links = []
    for link in fitz_page.get_links():
        if "uri" in link:
            links.append(link["uri"])
    return links

fitz_doc = fitz.open(file_path)

total_image_texts = 0
total_links = 0

for i in range(fitz_doc.page_count):
    fpage = fitz_doc[i]
    image_texts = extract_images_text(fpage, fitz_doc)
    links = extract_links(fpage)

    if image_texts:
        total_image_texts += len(image_texts)
        print(f"\n=== Page {i+1}: text found in {len(image_texts)} image(s) ===")
        for t in image_texts:
            print(t[:200])

    if links:
        total_links += len(links)
        print(f"\n=== Page {i+1}: {len(links)} link(s) ===")
        for l in links:
            print(l)

fitz_doc.close()

print(f"\n\nTotal: {total_image_texts} image(s) with text, {total_links} link(s) across document")