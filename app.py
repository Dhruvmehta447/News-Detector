import streamlit as st
import pickle
import numpy as np
import urllib.parse
import re
import google.generativeai as genai

# --- CONFIGURATION ---
# IMPORTANT: Replace this with your actual Gemini API Key
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "YOUR_LOCAL_FALLBACK_KEY")
# 1. Page Configuration
st.set_page_config(
    page_title="Intelligent News Verification",
    page_icon="📰",
    layout="wide"
)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# 2. Load ML Artifacts (Stylistic Model)
@st.cache_resource
def load_artifacts():
    model = pickle.load(open('model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    return model, vectorizer

model, vectorizer = load_artifacts()

# 3. Helper Functions
def calculate_confidence(decision_score):
    prob_real = 1.0 / (1.0 + np.exp(-decision_score))
    prob_fake = 1.0 - prob_real
    if prob_real >= 0.5:
        return "REAL", prob_real * 100
    else:
        return "FAKE", prob_fake * 100

def generate_search_links(headline):
    encoded_query = urllib.parse.quote_plus(headline)
    google_news_url = f"https://news.google.com/search?q={encoded_query}"
    wikipedia_url = f"https://en.wikipedia.org/w/index.php?search={encoded_query}"
    return google_news_url, wikipedia_url

def is_valid_input(text):
    words = re.findall(r'\b\w{2,}\b', text)
    return len(words) >= 5

def ai_fact_check(text):
    """Uses Gemini to evaluate the logical and factual validity of the claim."""
    try:
        # UPDATED: Using the latest 3.6-flash model
        gemini_model = genai.GenerativeModel('gemini-3.6-flash')
        prompt = f"""
        Analyze the following text for factual accuracy and logical consistency. 
        State clearly if the claim is FACTUAL, FALSE, or NONSENSE. 
        Provide a brief, 2-sentence explanation of your conclusion.

        Text to analyze: "{text}"
        """
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Fact-check failed: Ensure your API key is correct. Error: {e}"

# 4. User Interface
st.title("📰 Intelligent News Verification Engine")
st.markdown("Analyze linguistic authenticity, verify factual accuracy with AI, and research topics via established sources.")

article_title = st.text_input("Article Title / Headline:", placeholder="Enter the headline here...")
article_body = st.text_area("Article Body Text:", height=150, placeholder="Paste the full news text here...")

st.divider()

# 5. Verification Pipeline Trigger
if st.button("Run Comprehensive Verification", type="primary", use_container_width=True):
    full_content = (article_title + " " + article_body).strip()

    if not full_content:
        st.warning("⚠️ Please provide an article title or body text to analyze.")
    elif not is_valid_input(full_content):
        st.warning("⚠️ Input is too short. Please enter a complete sentence.")
    else:
        col1, col2, col3 = st.columns([1, 1.2, 1], gap="medium")

        # --- Column 1: ML Stylistic Analysis ---
        with col1:
            st.subheader("📊 Stylistic Classifier")
            with st.spinner("Analyzing style..."):
                transformed = vectorizer.transform([full_content])
                decision_score = model.decision_function(transformed)[0]
                label, confidence = calculate_confidence(decision_score)

                if label == "REAL":
                    st.success(f"**Predicted Style:**\n### AUTHENTIC")
                else:
                    st.error(f"**Predicted Style:**\n### SUSPICIOUS")

                st.metric(label="Style Confidence", value=f"{confidence:.1f}%")
                st.caption("*Analyzes formatting and vocabulary, not facts.*")

        # --- Column 2: AI Fact-Checker (Gemini) ---
        with col2:
            st.subheader("🤖 AI Fact-Checker")
            with st.spinner("Cross-referencing facts with Gemini AI..."):
                fact_check_result = ai_fact_check(full_content)
                st.info(fact_check_result)
                st.caption("*Semantic analysis powered by Google Gemini.*")

        # --- Column 3: Read More / Research Links ---
        with col3:
            st.subheader("🔍 Research Links")

            if article_title.strip():
                google_link, wiki_link = generate_search_links(article_title)
                st.markdown(f"**[🌍 Search Google News]({google_link})**")
                st.markdown(f"**[📚 Search Wikipedia]({wiki_link})**")
                st.caption("*Verify the original sources directly.*")
            else:
                st.warning("Please provide a headline to generate research links.")
