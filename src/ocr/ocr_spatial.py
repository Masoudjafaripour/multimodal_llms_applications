import cv2
import pytesseract
from pytesseract import Output

# === Load an example image (document, form, etc.) ===
image_path = "sample_doc.png"   # replace with your image
img = cv2.imread(image_path)

# === 1️⃣ Basic OCR: simple text only ===
plain_text = pytesseract.image_to_string(img)
print("=== BASIC OCR (text only) ===")
print(plain_text)

# === 2️⃣ Layout-aware OCR: text + coordinates ===
data = pytesseract.image_to_data(img, output_type=Output.DICT)
print("\n=== LAYOUT-AWARE OCR (text + bounding boxes) ===")

for i in range(len(data['text'])):
    text = data['text'][i]
    if text.strip():  # skip blanks
        (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
        print(f"Text: {text:15s}  |  BBox: ({x}, {y}, {w}, {h})")

# === Optional: visualize bounding boxes ===
for i in range(len(data['text'])):
    if data['text'][i].strip():
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

cv2.imwrite("results/layout_aware_output.png", img)
print("\nSaved visualization to results/layout_aware_output.png")
