import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Pencil Sketch Generator using OpenCV")

st.write(
    "Upload an image and let OpenCV's built-in pencilSketch function "
    "generate both a grayscale and a color sketch."
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Convert the uploaded file to a NumPy array and decode it using OpenCV.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        st.error("Error: Image not found or cannot be processed.")
    else:
        # Apply OpenCV's pencilSketch function.
        sketch_gray, sketch_color = cv2.pencilSketch(
            img, sigma_s=60, sigma_r=0.07, shade_factor=0.05
        )
        
        # Convert images from BGR to RGB for proper color display.
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sketch_color_rgb = cv2.cvtColor(sketch_color, cv2.COLOR_BGR2RGB)
        
        st.header("Results")
        
        # Display the original, grayscale sketch, and color sketch side by side.
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(img_rgb, caption="Original Image", use_column_width=True)
        with col2:
            st.image(sketch_gray, caption="Grayscale Sketch", use_column_width=True, channels="GRAY")
        with col3:
            st.image(sketch_color_rgb, caption="Color Sketch", use_column_width=True)
