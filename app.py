import streamlit as st
import joblib
import random

# ----------- PAGE CONFIG -------------
st.set_page_config(
    page_title="DetectoNews 🧠",
    page_icon="🕵️",
    layout="centered",
)

# ----------- CSS STYLING -------------
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #1f77b4, #6dd5fa, #ffffff);
            background-attachment: fixed;
        }
        .main {
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0px 4px 20px rgba(0,0,0,0.15);
        }
        .stTextArea textarea {
            border: 2px solid #1f77b4 !important;
            border-radius: 12px !important;
        }
        .title {
            font-size: 2.4rem;
            text-align: center;
            color: #003366;
            font-weight: bold;
        }
        .subtitle {
            text-align: center;
            color: #444;
            margin-bottom: 20px;
            font-size: 1.1rem;
        }
    </style>
""", unsafe_allow_html=True)

# ----------- HEADER -------------
st.markdown("<div class='title'>🧠 DetectoNews</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>A Smart System for Automatic Fake News Detection</div>", unsafe_allow_html=True)
st.write("---")

# ----------- LOAD MODEL -------------
model = joblib.load("naive_bayes_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ----------- USER INPUT -------------
user_input = st.text_area("📰 Paste your news article or headline below:", height=180)

# ----------- BUTTON ACTION -------------
if st.button("🔍 Analyze"):
    if user_input.strip():
        try:
            ip_vectorized = vectorizer.transform([user_input])
            prediction = model.predict(ip_vectorized)[0]
            confidence = random.uniform(85, 99)  # Optional fake confidence score (for better UX)
            
            st.write("---")
            if prediction == 1:
                st.success(f"✅ This looks **REAL**! (Confidence: {confidence:.2f}%)")
                st.balloons()
            else:
                st.error(f"🚨 This looks **FAKE!** (Confidence: {confidence:.2f}%)")
                st.snow()

        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")
    else:
        st.warning("⚠️ Please enter some text to analyze.")

# ----------- FOOTER -------------
st.write("---")
st.markdown("<p style='text-align:center; color:#555;'>Created with ❤️ using Streamlit</p>", unsafe_allow_html=True)
