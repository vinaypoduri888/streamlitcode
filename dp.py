import streamlit as st
from PIL import Image
import pytesseract
from fpdf import FPDF
import tempfile
import os

# Explicitly set the Tesseract command path
tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = tesseract_path

# Verify Tesseract path
if not os.path.exists(tesseract_path):
    st.error(f"Tesseract not found at {tesseract_path}")
else:
    st.success(f"Tesseract found at {tesseract_path}")

def extract_text_from_image(image):
    """Extract text from an image using Tesseract OCR."""
    return pytesseract.image_to_string(image)

def main():
    st.title("Image Text Extractor and PDF Generator")

    # Allow users to upload multiple images
    uploaded_files = st.file_uploader("Choose images...", accept_multiple_files=True, type=['jpg', 'png', 'jpeg'])
    
    if uploaded_files:
        extracted_texts = []
        st.write("### Uploaded Images and Extracted Texts")
        for uploaded_file in uploaded_files:
            # Open the image file
            image = Image.open(uploaded_file)
            
            # Display the image
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Extract text from the image
            extracted_text = extract_text_from_image(image)
            
            # Show the extracted text and allow user to modify
            modified_text = st.text_area(f"Extracted Text from {uploaded_file.name}", extracted_text, height=150)
            
            # Store the modified text
            extracted_texts.append(modified_text)

        if st.button("Generate PDF"):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                pdf = FPDF()
                pdf.set_auto_page_break(auto=True, margin=15)
                pdf.add_page()
                pdf.set_font("Arial", size=12)

                # Add each modified text to the PDF
                for text in extracted_texts:
                    pdf.multi_cell(0, 10, text)
                    pdf.ln(10)  # Add a line break after each text block

                # Output the PDF to the temporary file
                pdf.output(tmp_file.name)
                tmp_file.close()
                
                st.success("PDF generated successfully! You can download it below.")
                with open(tmp_file.name, "rb") as file:
                    st.download_button("Download PDF", file, file_name="extracted_texts.pdf")
            
            # Optionally delete the temporary file after use
            # os.remove(tmp_file.name)

if __name__ == "__main__":
    main()
