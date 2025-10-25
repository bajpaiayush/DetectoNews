import streamlit as st
import joblib
import base64

# Load model and vectorizer
model = joblib.load("naive_bayes_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Page Config
st.set_page_config(page_title="DetectoNews", page_icon="🕵️", layout="centered")

# --- Add background image ---
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}
        </style>
        """,
        unsafe_allow_html=True
    )

# Add your background image (make sure it's in the same folder)
add_bg_from_local("background.jpg")

# --- Title ---
st.markdown(
    "<h1 style='text-align: center; color: white;'>🕵️ DetectoNews</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; color: #f0f0f0; font-size:18px;'>Paste an article below — our AI will detect if it's real or fake 🔍</p>",
    unsafe_allow_html=True
)

# --- User Input ---
user_input = st.text_area("📰 Enter news content:", height=200)

# --- Prediction Button ---
if st.button("Check News"):
    if user_input.strip() != "":
        try:
            ip_vectorized = vectorizer.transform([user_input])
            prediction = model.predict(ip_vectorized)[0]

            if prediction == 1:
                st.success("✅ This news looks REAL!")
            else:
                st.error("🚨 This news seems FAKE!")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("⚠️ Please enter some text to analyze.")
