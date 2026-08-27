# 📰 Intelligent News Verification Engine

A comprehensive, full-stack web application designed to combat disinformation. This system goes beyond traditional fake news detection by combining statistical machine learning (stylometric analysis) with advanced generative AI (semantic fact-checking) to evaluate both the *style* and the *meaning* of a news article.

**Live Demo:** https://dhruv-news-detector.streamlit.app/

---

## ✨ Key Features

*   **📊 Stylistic Classifier (Machine Learning):** Uses a `TfidfVectorizer` and a `PassiveAggressiveClassifier` trained on the WELFake dataset. It calculates the mathematical probability that an article uses sensationalist, manipulative, or clickbait language.
*   **🤖 AI Fact-Checker (LLM):** Integrates Google's **Gemini 3.6-flash** API to semantically analyze the text. It understands real-world logic to detect factual inaccuracies, contradictions, and absurd claims that evade standard ML word-counters.
*   **🔍 Smart Research Links:** Automatically URL-encodes user headlines to generate direct, one-click search queries for Google News and Wikipedia, encouraging primary-source verification.
*   **🛡️ Input Validation:** Implements Regex-based validation to prevent "Zero Vector" false positives on gibberish or extremely short inputs.

---

## 🛠️ Tech Stack

*   **Language:** Python
*   **Frontend / Deployment:** Streamlit, Streamlit Community Cloud
*   **Machine Learning:** Scikit-Learn, Pandas, NumPy
*   **AI Integration:** Google Generative AI SDK (Gemini API)
