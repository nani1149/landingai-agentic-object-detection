import streamlit as st
import requests
import cv2
import numpy as np
import json  # for pretty printing JSON
import tempfile

# Streamlit UI
st.title("Image Upload and Object Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_image_path = temp_file.name
    
    # API details
    url = "https://api.landing.ai/v1/tools/agentic-object-detection"
    headers = {
        "Authorization": "Basic {API_KEY}"
    }
    
    # User input for prompts
    prompt = st.text_input("Enter prompt:", "find ev car")
    
    if st.button("Detect Objects"):
        with open(temp_image_path, "rb") as image_file:
            files = {"image": image_file}
            data = {"prompts": [prompt], "model": "agentic"}
            
            response = requests.post(url, files=files, data=data, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                st.json(result)
                
                image = cv2.imread(temp_image_path)
                
                for detection in result.get('data', [])[0]:
                    x1, y1, x2, y2 = map(int, detection['bounding_box'])
                    score = detection['score']
                    label = detection['label']
                    
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text = f"{label} (Score: {score:.2f})"
                    cv2.putText(image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                st.image(image, channels="BGR", caption="Detected Objects")
            else:
                st.error(f"Error: {response.status_code}, {response.text}")
