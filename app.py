import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image

# ===============================
# LOAD MODEL
# ===============================
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model

# Build base model
base_model = EfficientNetB3(
    weights=None,
    include_top=False,
    input_shape=(224, 224, 3)
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
output = Dense(7, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# Load trained weights
model.load_weights("skin_model.weights.h5")
#model = tf.keras.models.load_model("skin_cancer_model.keras", compile=False)

# ===============================
# CLASS NAMES
# ===============================
class_names = [
    "Actinic Keratosis",
    "Basal Cell Carcinoma",
    "Benign Keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic Nevus (Normal Skin)",
    "Vascular Lesion"
]

cancer_classes = [0, 1, 4]  # AKIEC, BCC, MEL

# ===============================
# STREAMLIT UI
# ===============================
st.title("🧬 AI Skin Cancer Detection System")
st.write("Upload a dermoscopy skin image for analysis")

uploaded_file = st.file_uploader("Choose a skin image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = img.resize((224, 224))
    arr = np.array(img_resized)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    # Prediction
    pred = model.predict(arr)
    class_index = np.argmax(pred)
    confidence = float(np.max(pred)) * 100

    skin_type = class_names[class_index]

    st.subheader("🔍 Prediction Result")

    # Low confidence warning
    if confidence < 50:
        st.warning("⚠ Low Confidence Prediction — Image may not match training data")

    # Cancer logic
    if class_index in cancer_classes:
        st.error("⚠ Cancer Detected")
        st.write("### Cancer Type:", skin_type)
    else:
        st.success("✅ Non-Cancerous")
        st.write("### Skin Type:", skin_type)

    st.write(f"### Confidence: {confidence:.2f}%")
