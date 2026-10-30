
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="FloraCoders | AI Flower Studio",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================================================
# FLOWER CLASSES
# ==================================================
class_names = [
    "dandelion",
    "daisy",
    "tulips",
    "sunflowers",
    "roses"
]

flower_emojis = {
    "dandelion": "🌼",
    "daisy": "🌼",
    "tulips": "🌷",
    "sunflowers": "🌻",
    "roses": "🌹"
}

# ==================================================
# CUSTOM DESIGN
# ==================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Playfair+Display:wght@500;600;700&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at 5% 5%, #fce4ef 0%, transparent 35%),
        radial-gradient(ellipse at 95% 15%, #e8ddff 0%, transparent 35%),
        radial-gradient(ellipse at 15% 95%, #e0f1e4 0%, transparent 35%),
        linear-gradient(135deg, #fff9fc, #f8f4ff 55%, #f4fbf5);
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    max-width: 1250px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

[data-testid="stHeader"] {
    background: transparent;
}

h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: #49255f !important;
}

p, label, span {
    font-family: 'DM Sans', sans-serif;
}

.hero {
    text-align: center;
    padding: 42px 20px;
    background: linear-gradient(135deg, #ffffffee, #f9eafaee);
    border: 1px solid #ffffff;
    border-radius: 30px;
    box-shadow: 0 12px 40px #7d429018;
    margin-bottom: 30px;
}

.hero .brand {
    font-family: 'Playfair Display', serif;
    font-size: clamp(42px, 6vw, 65px);
    font-weight: 700;
    letter-spacing: -1.5px;
    background: linear-gradient(90deg, #853ba3, #d94f92, #537b58);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero .tagline {
    font-size: 19px;
    color: #715c81;
    margin-top: 8px;
}

.hero .description {
    color: #71677b;
    font-size: 14px;
    margin-top: 14px;
}

.pill {
    display: inline-block;
    background: #f0e5f8;
    color: #74418b;
    border-radius: 30px;
    padding: 8px 16px;
    margin: 5px;
    font-size: 12px;
    font-weight: 700;
}

.section-title {
    color: #542e70;
    font-family: 'Playfair Display', serif;
    font-size: 27px;
    font-weight: 700;
    margin: 5px 0 8px;
}

.section-subtitle {
    color: #81738d;
    font-size: 13px;
    margin-bottom: 20px;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffffd9;
    border: 1px solid #ffffff !important;
    border-radius: 24px;
    box-shadow: 0 8px 30px #603d7b12;
    padding: 5px;
}

div[data-testid="stFileUploader"] {
    background: #ffffffb8;
    border: 2px dashed #d8b6e8;
    border-radius: 20px;
    padding: 15px;
}

div[data-testid="stCameraInput"] {
    background: #ffffffb8;
    border: 2px dashed #d8b6e8;
    border-radius: 20px;
    padding: 15px;
}

div[data-testid="stButton"] > button {
    background: linear-gradient(100deg, #8b3ba3, #c24b99);
    color: white;
    border: none;
    border-radius: 15px;
    min-height: 49px;
    font-weight: 700;
    box-shadow: 0 6px 18px #a346a72b;
    transition: 0.2s ease;
}

div[data-testid="stButton"] > button:hover {
    color: white;
    border: 1px solid white;
    transform: translateY(-2px);
    box-shadow: 0 9px 24px #a346a740;
}

.result-banner {
    background: linear-gradient(120deg, #fce4ef, #eee5ff);
    border: 1px solid white;
    border-radius: 22px;
    padding: 25px;
    text-align: center;
    margin: 15px 0 20px;
}

.result-label {
    color: #85658d;
    font-size: 13px;
    font-weight: 700;
}

.result-flower {
    font-family: 'Playfair Display', serif;
    font-size: 36px;
    font-weight: 700;
    color: #773a91;
    margin-top: 5px;
}

.confidence {
    display: inline-block;
    margin-top: 10px;
    background: white;
    color: #a23d85;
    padding: 9px 17px;
    border-radius: 30px;
    font-weight: 700;
}

.small-note {
    text-align: center;
    color: #85768f;
    font-size: 12px;
    padding: 10px;
}

.gallery-card {
    text-align: center;
    padding: 15px 5px;
    color: #603b75;
}

.gallery-emoji {
    font-size: 42px;
}

#MainMenu, footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# HEADER
# ==================================================
st.markdown("""
<div class="hero">
<div style="font-size:38px;">
🌷 🌸 🌼 🌺 🌻
</div>
<div class="brand">
FloraCoders
</div>
<div class="tagline">
A Little Magic in Every Flower ✨
</div>
<div class="description">
Discover the beauty of nature through Artificial Intelligence.
Upload a photo or capture a flower with your camera!
</div>
<div style="margin-top:20px;">
<span class="pill">🌿 5 Flower Species</span>
<span class="pill">🤖 Deep Learning</span>
<span class="pill">📷 Live Camera</span>
<span class="pill">💗 AI Powered</span>
</div>
</div>
""", unsafe_allow_html=True)


# ==================================================
# MODEL LOADING
# ==================================================
@st.cache_resource
def load_flower_model():
    return tf.keras.models.load_model(
        "floracoders_best_model.keras"
    )

try:
    model = load_flower_model()
except Exception as e:
    st.error("Flower AI model could not be loaded.")
    st.exception(e)
    st.stop()


# ==================================================
# FLOWER PREDICTION FUNCTION
# ==================================================
def predict_flower(image):
    image = image.convert("RGB")
    resized_image = image.resize((180, 180))

    image_array = np.array(
        resized_image,
        dtype=np.float32
    ) / 255.0

    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    if len(predictions) != 5:
        raise ValueError("Unexpected model output.")

    predicted_index = int(np.argmax(predictions))
    flower = class_names[predicted_index]

    confidence = float(predictions[predicted_index]) * 100

    return {
        "flower": flower,
        "confidence": confidence,
        "scores": predictions.tolist(),
        "image": image.copy()
    }


# ==================================================
# MAIN INTERFACE
# ==================================================
left_col, right_col = st.columns(
    [1.05, 0.95],
    gap="large"
)


# ==================================================
# LEFT: IMAGE UPLOAD + CAMERA
# ==================================================
with left_col:
    with st.container(border=True):

        st.markdown(
            '<div class="section-title">🌸 Discover Your Flower</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-subtitle">Choose your favourite way to identify a flower.</div>',
            unsafe_allow_html=True
        )

        upload_tab, camera_tab = st.tabs(
            ["📁 Upload Photo", "📷 Live Camera"]
        )

        # IMAGE UPLOAD
        with upload_tab:
            st.markdown("#### 🌷 Upload Your Flower")

            uploaded_file = st.file_uploader(
                "Choose a flower image",
                type=["jpg", "jpeg", "png"],
                key="flower_upload"
            )

            if uploaded_file is not None:
                try:
                    image = Image.open(uploaded_file).convert("RGB")

                    st.image(
                        image,
                        caption="Your uploaded flower 🌸",
                        use_container_width=True
                    )

                    if st.button(
                        "✨ Identify My Flower",
                        key="upload_predict",
                        use_container_width=True
                    ):
                        with st.spinner(
                            "Our AI is exploring your flower..."
                        ):
                            st.session_state["flower_result"] = (
                                predict_flower(image)
                            )
                            st.session_state["result_source"] = (
                                "Uploaded Image"
                            )

                        st.success("Flower prediction completed!")

                except Exception:
                    st.error(
                        "Unable to process this image. "
                        "Please upload another photo."
                    )
            else:
                st.markdown(
                    '<div class="small-note">💐 Choose a clear JPG or PNG flower picture.</div>',
                    unsafe_allow_html=True
                )

        # CAMERA
        with camera_tab:
            st.markdown("#### 📷 Capture a Flower Live")

            st.markdown(
                '<div class="small-note">Allow camera access and point your camera towards any flower.</div>',
                unsafe_allow_html=True
            )

            camera_image = st.camera_input(
                "Open Camera and Capture Flower",
                key="flower_camera"
            )

            if camera_image is not None:
                try:
                    image = Image.open(camera_image).convert("RGB")

                    st.image(
                        image,
                        caption="Your captured flower 🌷",
                        use_container_width=True
                    )

                    if st.button(
                        "🌸 Identify Camera Flower",
                        key="camera_predict",
                        use_container_width=True
                    ):
                        with st.spinner(
                            "AI is recognising your flower..."
                        ):
                            st.session_state["flower_result"] = (
                                predict_flower(image)
                            )
                            st.session_state["result_source"] = (
                                "Live Camera"
                            )

                        st.success("Camera flower identified!")

                except Exception:
                    st.error(
                        "Unable to process the camera picture."
                    )
            else:
                st.info(
                    "📸 Your camera preview will appear here. "
                    "Take a picture to continue."
                )


# ==================================================
# RIGHT: RESULTS PANEL
# ==================================================
with right_col:
    with st.container(border=True):

        st.markdown(
            '<div class="section-title">🔮 AI Prediction Results</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-subtitle">Discover what our trained CNN model recognises.</div>',
            unsafe_allow_html=True
        )

        result = st.session_state.get("flower_result")

        if result is not None:
            flower = result["flower"]
            confidence = result["confidence"]

            st.markdown(f"""
<div class="result-banner">
<div style="font-size:55px;">
{flower_emojis[flower]}
</div>
<div class="result-label">
OUR AI PREDICTS
</div>
<div class="result-flower">
{flower.title()}
</div>
<div class="confidence">
{confidence:.2f}% Confidence
</div>
</div>
""", unsafe_allow_html=True)

            st.caption(
                "Input method: "
                + st.session_state.get(
                    "result_source", "Flower Image"
                )
            )

            st.markdown("#### 🌈 All Flower Prediction Scores")

            for i, name in enumerate(class_names):
                score = float(result["scores"][i])

                score_col, bar_col = st.columns([1.3, 3])

                with score_col:
                    st.write(
                        f"{flower_emojis[name]} {name.title()}"
                    )

                with bar_col:
                    st.progress(
                        min(max(score, 0.0), 1.0),
                        text=f"{score * 100:.2f}%"
                    )

            st.markdown(
                '<div class="small-note">Prediction generated using our trained CNN model.</div>',
                unsafe_allow_html=True
            )

        else:
            st.markdown("""
<div style="text-align:center; padding:40px 10px;">
<div style="font-size:85px;">
💐
</div>
<h3 style="color:#815391;">
Your Results Will Bloom Here!
</h3>
<p style="color:#897991; font-size:14px;">
Upload a flower image or capture one using Live Camera to discover its species.
</p>
<div style="font-size:30px; margin-top:20px;">
🌼 🌷 🌻 🌹 🌸
</div>
</div>
""", unsafe_allow_html=True)


# ==================================================
# MODEL INFORMATION
# ==================================================
st.markdown("---")

st.markdown(
    '<div class="section-title" style="text-align:center;">🤖 Meet Our AI Model</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle" style="text-align:center;">Our trained Convolutional Neural Network recognises five flower categories.</div>',
    unsafe_allow_html=True
)

m1, m2, m3 = st.columns(3)

with m1:
    st.metric("Test Accuracy", "89.09%")

with m2:
    st.metric("Flower Classes", "5")

with m3:
    st.metric("Test Images", "550")


# ==================================================
# FLOWER GALLERY
# ==================================================
st.markdown("---")

st.markdown(
    '<div class="section-title" style="text-align:center;">🌺 Explore Our Little Garden</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle" style="text-align:center;">The five beautiful flower species learned by FloraCoders AI.</div>',
    unsafe_allow_html=True
)

gallery = st.columns(5)

for col, name in zip(gallery, class_names):
    with col:
        with st.container(border=True):
            st.markdown(f"""
<div class="gallery-card">
<div class="gallery-emoji">
{flower_emojis[name]}
</div>
<div style="font-weight:700;">
{name.title()}
</div>
</div>
""", unsafe_allow_html=True)


# ==================================================
# FOOTER
# ==================================================
st.markdown("""
<div style="text-align:center; padding:35px 10px 15px; color:#806b8a;">
<div style="font-family:'Playfair Display',serif; font-size:25px; font-weight:700;">
🌸 FloraCoders 🌸
</div>
<div style="font-size:13px; margin-top:10px;">
Small Flowers · Big Dreams · Powered by AI
</div>
<div style="font-size:11px; margin-top:12px;">
Flower Species Classification Project
</div>
</div>
""", unsafe_allow_html=True)
