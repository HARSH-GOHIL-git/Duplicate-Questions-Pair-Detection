import streamlit as st
from sentence_transformers import CrossEncoder, SentenceTransformer, util

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Duplicate Question Detector",
    page_icon="🔍",
    layout="centered"
)

# --- MODEL LOADING (CACHED) ---
# This ensures models are downloaded/loaded into memory only once
@st.cache_resource
def load_models():
    bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")
    return bi_encoder, nli_model

with st.spinner("Loading models... This might take a minute on the first run."):
    bi_encoder, nli_model = load_models()

# --- LOGIC FUNCTIONS ---
def check_similarity(q1, q2):
    emb1 = bi_encoder.encode(q1, convert_to_tensor=True)
    emb2 = bi_encoder.encode(q2, convert_to_tensor=True)
    similarity = util.cos_sim(emb1, emb2)
    return similarity.item()

def check_contradiction(q1, q2):
    # Predict returns logits for contradiction, neutral, entailment
    scores = nli_model.predict([(q1, q2)])[0]
    return {
        "contradiction": float(scores[0]),
        "neutral": float(scores[1]),
        "entailment": float(scores[2])
    }

def is_duplicated(q1, q2, sim_threshold, contradiction_threshold):
    sim_score = check_similarity(q1, q2)
    nli_scores = check_contradiction(q1, q2)
    
    # C1: Low Similarity
    if sim_score < sim_threshold:
        return {"From": "C1", "Similarity": "NO", "similarity_score": sim_score, "nli_score": nli_scores}
    
    # C2: High Similarity, but Contradictory
    if nli_scores['contradiction'] > contradiction_threshold:
        return {"From": "C2", "Similarity": "NO", "similarity_score": sim_score, "nli_score": nli_scores}
        
    # C3: High Similarity, Not Contradictory -> Duplicate
    return {"From": "C3", "Similarity": "Yes", "similarity_score": sim_score, "nli_score": nli_scores}

# --- UI LAYOUT ---
st.title("🔍 Duplicate Question Pair Detector")
st.markdown("Determine if two questions carry the same semantic meaning using Bi-Encoders and Cross-Encoders.")

# Sidebar for threshold tweaking
st.sidebar.header("⚙️ Model Parameters")
st.sidebar.markdown("Adjust the thresholds used to determine duplication.")
sim_threshold = st.sidebar.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.75, step=0.05)
contradiction_threshold = st.sidebar.slider("Contradiction Threshold (Logit)", min_value=-5.0, max_value=10.0, value=0.6, step=0.1)

st.markdown("### Enter Questions")
q1 = st.text_input("Question 1", placeholder="e.g., What is good for eyes?")
q2 = st.text_input("Question 2", placeholder="e.g., What is not good for eyes?")

if st.button("Check Duplication", type="primary"):
    if not q1 or not q2:
        st.warning("Please enter both questions to begin the analysis.")
    else:
        with st.spinner("Analyzing semantics and entailment..."):
            result = is_duplicated(q1, q2, sim_threshold, contradiction_threshold)
            
            st.markdown("---")
            
            # Display Final Verdict with visual cues
            if result["Similarity"] == "Yes":
                st.success("### ✅ These questions are Duplicates.")
            else:
                st.error("### ❌ These questions are NOT Duplicates.")
                if result["From"] == "C1":
                    st.info("Reason: They do not meet the minimum similarity threshold.")
                elif result["From"] == "C2":
                    st.info("Reason: Despite having high similarity, they contain contradictory semantic logic.")

            # Display Metrics cleanly using Streamlit columns
            st.markdown("### 📊 Detailed Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("Cosine Similarity", f"{result['similarity_score']:.3f}")
            col2.metric("Contradiction", f"{result['nli_score']['contradiction']:.3f}")
            col3.metric("Neutral", f"{result['nli_score']['neutral']:.3f}")
            col4.metric("Entailment", f"{result['nli_score']['entailment']:.3f}")