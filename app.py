
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="FloraCoders | Flower AI",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- BEAUTIFUL DESIGN ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at 5% 5%, #fce4ef 0%, transparent 35%),
        radial-gradient(ellipse at 95% 20%, #e8ddff 0%, transparent 35%),
        radial-gradient(ellipse at 20% 90%, #e0f1e4 0%, transparent 40%),
        linear-gradient(135deg, #fff9fc, #f8f4ff 55%, #f4fbf5);
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    max-width: 1200px;
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
    padding: 30px 15px 25px;
    background: linear-gradient(135deg, #ffffffdd, #f9eafaee);
    border: 1px solid #ffffff;
    border-radius: 28px;
    box-shadow: 0 12px 40px #7d429018;
    margin-bottom: 25px;
}

.hero .brand {
    font-family: 'Playfair Display', serif;
    font-size: clamp(38px, 5vw, 58px);
    font-weight: 700;
    letter-spacing: -1.5px;
    background: linear-gradient(90deg, #853ba3, #d94f92, #537b58);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero .tagline {
    font-size: 17px;
    color: #715c81;
    margin-top: 5px;
}

.hero .description {
    color: #71677b;
    font-size: 14px;
    margin-top: 12px;
}

.pill {
    display: inline-block;
    background: #f0e5f8;
    color: #74418b;
    border-radius: 30px;
    padding: 7px 15px;
    margin: 5px 3px;
    font-size: 12px;
    font-weight: 600;
}

.section-title {
    color: #542e70;
    font-family: 'Playfair Display', serif;
    font-size: 25px;
    font-weight: 700;
    margin: 4px 0 8px;
}

.section-subtitle {
    color: #81738d;
    font-size: 13px;
    margin-bottom: 18px;
}

div[data-testid="stFileUploader"] {
    background: #ffffffb8;
    border: 2px dashed #d8b6e8;
    border-radius: 20px;
    padding: 15px;
}

div[data-testid="stFileUploader"] section {
    background: transparent;
}

div[data-testid="stButton"] > button {
    background: linear-gradient(100deg, #8b3ba3, #c24b99);
    color: white;
    border: none;
    border-radius: 15px;
    min-height: 48px;
    font-weight: 700;
    box-shadow: 0 6px 18px #a346a72b;
    transition: 0.2s ease;
}

div[data-testid="stButton"] > button:hover {
    color: white;
    border: 1px solid #ffffff;
    transform: translateY(-2px);
    box-shadow: 0 9px 24px #a346a740;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffffc9;
    border: 1px solid #ffffff !important;
    border-radius: 23px;
    box-shadow: 0 8px 30px #603d7b12;
}

.result-banner {
    background: linear-gradient(120deg, #fce4ef, #eee5ff);
    border: 1px solid #ffffff;
    border-radius: 20px;
    padding: 20px;
    text-align: center;
    margin: 12px 0 18px;
}

.result-label {
    color: #85658d;
    font-size: 13px;
    font-weight: 600;
}

.result-flower {
    font-family: 'Playfair Display', serif;
    font-size: 33px;
    font-weight: 700;
    color: #773a91;
    margin-top: 3px;
}

.confidence {
    display: inline-block;
    margin-top: 8px;
    background: #ffffffc9;
    color: #a23d85;
    padding: 7px 15px;
    border-radius: 30px;
    font-weight: 700;
}

.small-note {
    text-align: center;
    color: #85768f;
    font-size: 12px;
    padding: 8px;
}

hr {
    border-color: #e9dcec;
}

#MainMenu, footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ---------------- FLOWER CLASSES ----------------
# Keep the exact order used during model training.
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

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <div style="font-size:30px;">🌷 🌸 🌼 🌺 🌻</div>
    <div class="brand">FloraCoders</div>
    <div class="tagline">A little magic in every flower ✨</div>
    <div class="description">
        Discover the beauty of nature with our AI-powered flower classifier.
    </div>
    <div style="margin-top:12px;">
        <span class="pill">🌿 5 Flower Species</span>
        <span class="pill">🤖 Deep Learning</span>
        <span class="pill">💗 Made with Care</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_flower_model():
    return tf.keras.models.load_model(
        "floracoders_best_model.keras"
    )

try:
    model = load_flower_model()
except Exception as e:
    st.error("The flower model could not be loaded.")
    st.info(
        "Please check that floracoders_best_model.keras "
        "exists in the GitHub repository root and that "
        "requirements.txt contains the required libraries."
    )
    st.stop()

# ---------------- MAIN INTERFACE ----------------
left_col, right_col = st.columns(
    [1, 1], gap="large"
)

with left_col:
    with st.container(border=True):
        st.markdown(
            '<div class="section-title">🌸 Meet Your Flower</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="section-subtitle">'
            'Upload a flower photo and let your AI explore it.'
            '</div>',
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "Choose your flower image",
            type=["jpg", "jpeg", "png"],
            help="Upload a JPG, JPEG or PNG flower photo."
        )

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(
                    image,
                    caption="Your beautiful flower 🌷",
                    use_container_width=True
                )

                st.markdown(
                    '<div class="small-note">'
                    'Your image is ready for identification.'
                    '</div>',
                    unsafe_allow_html=True
                )

                predict_clicked = st.button(
                    "✨  Identify My Flower",
                    use_container_width=True
                )

                if predict_clicked:
                    with st.spinner(
                        "🌿 Our AI is exploring your flower..."
                    ):
                        resized_image = image.resize((180, 180))
                        image_array = np.array(
                            resized_image, dtype=np.float32
                        ) / 255.0
                        image_array = np.expand_dims(
                            image_array, axis=0
                        )

                        predictions = model.predict(
                            image_array, verbose=0
                        )[0]

                    if len(predictions) != len(class_names):
                        st.error(
                            "The model output does not match "
                            "the five flower classes."
                        )
                    else:
                        predicted_index = int(
                            np.argmax(predictions)
                        )
                        predicted_flower = class_names[
                            predicted_index
                        ]
                        confidence = float(
                            predictions[predicted_index]
                        ) * 100

                        st.session_state["flower_result"] = {
                            "image": image.copy(),
                            "flower": predicted_flower,
                            "confidence": confidence,
                            "scores": predictions.tolist()
                        }

            except Exception:
                st.error(
                    "We couldn't read this image. "
                    "Please try another JPG or PNG file."
                )
        else:
            st.markdown(
                '<div class="small-note">'
                '💐 Tip: A clear flower photo usually helps '
                'the model recognise its features.'
                '</div>',
                unsafe_allow_html=True
            )

with right_col:
    with st.container(border=True):
        st.markdown(
            '<div class="section-title">🔮 Your Flower Story</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="section-subtitle">'
            'Your AI prediction will appear here.'
            '</div>',
            unsafe_allow_html=True
        )

        result = st.session_state.get("flower_result")

        if result is not None:
            flower = result["flower"]
            confidence = result["confidence"]

            st.markdown(
                f"""
                <div class="result-banner">
                    <div style="font-size:42px;">
                        {flower_emojis[flower]}
                    </div>
                    <div class="result-label">OUR AI PREDICTS</div>
                    <div class="result-flower">
                        {flower.title()}
                    </div>
                    <div class="confidence">
                        {confidence:.2f}% confidence
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("#### 🌈 Prediction Scores")

            scores = result["scores"]

            for i, name in enumerate(class_names):
                score = float(scores[i])
                c1, c2 = st.columns([1, 4])

                with c1:
                    st.write(
                        f"{flower_emojis[name]} "
                        f"{name.title()}"
                    )

                with c2:
                    st.progress(
                        min(max(score, 0.0), 1.0),
                        text=f"{score * 100:.2f}%"
                    )

            st.caption(
                "These scores represent the model's output "
                "for each of the five flower classes."
            )
        else:
            st.markdown(
                """
                <div style="text-align:center; padding:25px 8px;">
                    <div style="font-size:75px;">
                        💐
                    </div>
                    <h3 style="color:#815391;">
                        Your results will bloom here!
                    </h3>
                    <p style="color:#897991; font-size:14px;">
                        Upload a flower image and click
                        <b>Identify My Flower</b> to discover
                        what our AI predicts.
                    </p>
                    <div style="font-size:27px; margin-top:20px;">
                        🌼 🌷 🌻 🌹 🌸
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

# ---------------- FLOWER GALLERY ----------------
st.markdown("---")
st.markdown(
    '<div class="section-title" style="text-align:center;">'
    '🌺 Our Little Garden</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="section-subtitle" style="text-align:center;">'
    'Five flower species our AI has learned to recognise.'
    '</div>',
    unsafe_allow_html=True
)

gallery = st.columns(5)

for col, name in zip(gallery, class_names):
    with col:
        with st.container(border=True):
            st.markdown(
                f"""
                <div style="text-align:center; padding:8px 0;">
                    <div style="font-size:38px;">
                        {flower_emojis[name]}
                    </div>
                    <div style="font-weight:700; color:#603b75;">
                        {name.title()}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

# ---------------- FOOTER ----------------
st.markdown("""
<div style="
    text-align:center;
    padding:25px 10px 8px;
    color:#806b8a;
">
    <div style="font-size:22px;">🌸 FloraCoders 🌸</div>
    <div style="font-size:13px; margin-top:8px;">
        Small flowers · Big dreams · Powered by AI
    </div>
    <div style="font-size:11px; margin-top:10px;">
        Flower Species Classification Project
    </div>
</div>
""", unsafe_allow_html=True)


