import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load the BLIP model and processor
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    return processor, model

processor, model = load_model()

# Video transformer to capture a single frame from the webcam
class SingleImageCapture(VideoTransformerBase):
    def __init__(self):
        self.frame = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame = Image.fromarray(img)
        return frame

# Define the Streamlit application
def main():
    st.title("Visual Question Answering with BLIP")

    # Add a sidebar with information
    st.sidebar.title("About This App")
    st.sidebar.write("## Features")
    st.sidebar.write("""
    - Upload an image or capture an image using the webcam
    - Ask a question about the image
    - Get an AI-generated answer
    """)
    st.sidebar.write("## How to Use")
    st.sidebar.write("""
    1. Select whether to upload an image or capture a photo using the webcam.
    2. Upload an image or capture a photo.
    3. Enter a question about the image in the text input.
    4. View the generated answer below the input.
    """)
    st.sidebar.write("### Developed by Vinay")

    # Option to select image source
    option = st.selectbox("Select image source:", ("Upload an image", "Capture an image with webcam"))

    raw_image = None
    if option == "Upload an image":
        image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if image_file is not None:
            raw_image = Image.open(image_file).convert('RGB')
    else:
        # Webcam capture
        webrtc_ctx = webrtc_streamer(key="example", video_transformer_factory=SingleImageCapture)
        if webrtc_ctx.video_transformer:
            video_transformer = webrtc_ctx.video_transformer
            if st.button("Capture Image"):
                if video_transformer.frame is not None:
                    raw_image = video_transformer.frame.convert('RGB')
                    st.image(raw_image, caption='Captured Image', use_column_width=True)
                else:
                    st.write("No image captured yet.")

    if raw_image:
        st.image(raw_image, caption='Selected Image', use_column_width=True)

        # Ask a question
        question = st.text_input("Ask a question about the image:")

        if question:
            # Process the inputs
            inputs = processor(raw_image, question, return_tensors="pt")

            # Generate the output
            out = model.generate(**inputs)

            # Decode and print the result
            answer = processor.decode(out[0], skip_special_tokens=True)
            st.write("Answer: ", answer)

# Run the Streamlit application
if __name__ == "__main__":
    main()
