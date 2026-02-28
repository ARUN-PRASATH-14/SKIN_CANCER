import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Skin Cancer Detection", layout="centered")

st.title("🩺 Skin Cancer Detection System")
st.write("Upload a skin image to detect skin type and cancer status.")

# ===============================
# LOAD TFLITE MODEL
# ===============================
interpreter = tf.lite.Interpreter(model_path="skin_cancer_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ===============================
# CLASS NAMES
# ===============================
class_names = [
    "Actinic Keratosis",
    "Basal Cell Carcinoma (BCC)",
    "Benign Keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic Nevus (Normal Skin)",
    "Vascular Lesion"
]

cancer_classes = [0, 1, 4]  # AKIEC, BCC, MEL

# ===============================
# IMAGE UPLOAD
# ===============================
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Resize to 224x224 (IMPORTANT)
    image = image.resize((224, 224))
    img_array = np.array(image).astype(np.float32)

    # Normalize if your model used EfficientNet preprocessing
    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    # ===============================
    # PREDICTION
    # ===============================
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100

    skin_type = class_names[class_index]

    # Cancer or Not
    if class_index in cancer_classes:
        status = "⚠ Cancer Detected"
        cancer_type = skin_type
    else:
        status = "✅ Non-Cancerous"
        cancer_type = "None"

    # ===============================
    # OUTPUT
    # ===============================
    st.subheader("🔍 Detection Result")

    st.write("### 🧬 Skin Type:")
    st.success(skin_type)

    st.write("### 🏥 Cancer Status:")
    if status == "⚠ Cancer Detected":
        st.error(status)
        st.write("### Cancer Type:")
        st.warning(cancer_type)
    else:
        st.success(status)

    st.write("### 📊 Confidence:")
    st.info(f"{confidence:.2f}%")

    st.progress(min(int(confidence), 100))

    # Extra verification message
    if class_index == 1:
        st.write("✔ This image is detected as Basal Cell Carcinoma (BCC)")
    elif class_index == 6:
        st.write("✔ This image is detected as Vascular Lesion")
    elif class_index == 5:
        st.write("✔ This image is detected as Normal Skin (Melanocytic Nevus)")
