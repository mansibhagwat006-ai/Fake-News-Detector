import re
import joblib
import numpy as np
import streamlit as st

# Must be the first Streamlit command
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")


# IMPORTANT: this must be the SAME cleaning function you used in training.
def clean_text(text):
    text = str(text)
    text = re.sub(r"^.{0,100}?\(Reuters\)\s*-?\s*", "", text)  # remove agency tag
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)       # remove links
    text = re.sub(r"[^a-z\s]", " ", text)      # keep letters only
    text = re.sub(r"\s+", " ", text).strip()   # tidy spaces
    return text


# Load the saved model once and keep it in memory
@st.cache_resource
def load_model():
    return joblib.load("models/fake_news_model.pkl")


model = load_model()
tfidf = model.named_steps["tfidf"]
clf = model.named_steps["clf"]


def top_words(cleaned, n=8):
    """Words that pushed the prediction most toward Real and toward Fake."""
    vec = tfidf.transform([cleaned]).toarray()[0]
    contrib = vec * clf.coef_[0]               # word score x model weight
    names = tfidf.get_feature_names_out()
    nonzero = np.where(vec > 0)[0]             # only words present in this article
    order = nonzero[np.argsort(contrib[nonzero])]
    fake_idx = order[:n]                       # most negative -> Fake
    real_idx = order[::-1][:n]                 # most positive -> Real
    fake_words = [names[i] for i in fake_idx if contrib[i] < 0]
    real_words = [names[i] for i in real_idx if contrib[i] > 0]
    return real_words, fake_words


# ---------- Sidebar ----------
with st.sidebar:
    st.header("About")
    st.write(
        "This app uses a TF-IDF + Logistic Regression model trained on the "
        "ISOT Fake and Real News dataset (mostly US political news, 2016-2017)."
    )
    st.warning(
        "The model detects **writing patterns**, not facts. "
        "It can be wrong on other topics, years, or countries. "
        "Always verify news with trusted sources."
    )

# ---------- Main page ----------
st.title("📰 Fake News Detector")
st.write("Paste a news article below and the model will predict whether it looks **Real** or **Fake**.")

article = st.text_area("Article text (headline + body works best):", height=250)

if st.button("Analyze", type="primary"):
    if len(article.strip()) < 50:
        st.warning("Please paste a longer article (at least a few sentences).")
    else:
        cleaned = clean_text(article)

        if len(cleaned.split()) < 10:
            st.warning("After cleaning, too little text is left. Please paste a longer article.")
        else:
            proba = model.predict_proba([cleaned])[0]   # [P(Fake), P(Real)]
            p_fake, p_real = proba[0], proba[1]

            if p_real >= 0.5:
                st.success(f"Prediction: REAL ({p_real:.1%} confidence)")
            else:
                st.error(f"Prediction: FAKE ({p_fake:.1%} confidence)")

            col1, col2 = st.columns(2)
            col1.metric("Probability Real", f"{p_real:.1%}")
            col2.metric("Probability Fake", f"{p_fake:.1%}")
            st.progress(float(p_real), text="Real probability")

            if max(p_fake, p_real) < 0.65:
                st.info("The model is not very sure about this one. Treat the result with extra caution.")

            with st.expander("Why did the model say this?"):
                real_words, fake_words = top_words(cleaned)
                st.writestrea("**Words pushing toward REAL:**", ", ".join(real_words) or "none")
                st.write("**Words pushing toward FAKE:**", ", ".join(fake_words) or "none")
                st.caption("These come from the model's learned word weights, so they show patterns, not proof.")