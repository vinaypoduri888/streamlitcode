import streamlit as st
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

# Load the model and processor
processor = AutoProcessor.from_pretrained("microsoft/git-base")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")

def generate_description(image):
    # Prepare the image for the model
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.generate(**inputs)
    
    # Decode the generated description
    description = processor.decode(outputs[0], skip_special_tokens=True)
    return description

st.title("Image Description Generator")

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    if st.button('Generate Description'):
        description = generate_description(image)
        st.write("Description:")
        st.write(description)
