import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from src.generation import get_llm, get_rag_chain

# --- Page Optimization & Aesthetics ---
st.set_page_config(
    page_title="Multi-LLM ClimateWater RAG | India",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Theme Styling ---
st.markdown("""
<style>
    :root {
        --primary: #0077B6;
        --secondary: #00B4D8;
        --background: #03045E;
        --text: #CAF0F8;
    }

    .stApp {
        background: linear-gradient(135deg, #03045E 0%, #0077B6 100%);
        color: #CAF0F8;
    }

    .main-header {
        font-family: 'Outfit', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #90E0EF, #00B4D8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-align: center;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1rem;
    }

    .retrieved-chunk {
        background: rgba(0, 180, 216, 0.1);
        border-left: 4px solid #00B4D8;
        padding: 0.8rem;
        margin-top: 0.5rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

load_dotenv()

# --- Sidebar Configuration ---
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?auto=format&fit=crop&q=80&w=300", 
             use_container_width=True, caption="India's Water Resilience")
    st.title("⚙️ RAG Configuration")
    st.markdown("---")
    
    # 1. Select Provider
    llm_provider = st.selectbox("Select LLM Provider", ["Groq", "OpenAI", "Gemini"])
    
    # 2. Dynamic Defaults based on selection
    if llm_provider == "Groq":
        default_model = "llama-3.3-70b-versatile"
        env_key = os.getenv("GROQ_API_KEY", "")
    elif llm_provider == "OpenAI":
        default_model = "gpt-4-turbo"
        env_key = os.getenv("OPENAI_API_KEY", "")
    elif llm_provider == "Gemini":
        default_model = "gemini-pro"
        env_key = os.getenv("GOOGLE_API_KEY", "")

    # 3. Inputs for API Key and Model Name
    api_key = st.text_input(f"Enter {llm_provider} API Key", value="", type="password")
    model_name = st.text_input(f"Enter Model Name", value=default_model)
    
    st.markdown("---")
    st.markdown("### 🔄 Data Management")
    if st.button("Reload Knowledge Base / Clear Cache"):
        st.cache_resource.clear()
        st.rerun()
        
    st.markdown("---")
    st.markdown("### 🔍 Static Components")
    st.info("**Embeddings**: HuggingFace (all-MiniLM-L6-v2)\n\n**Vector DB**: FAISS")

# --- Resource Loading ---
@st.cache_resource
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("data/faiss_index", embeddings, allow_dangerous_deserialization=True)

try:
    vector_store = load_vector_store()
except Exception as e:
    st.error(f"Error loading FAISS index: {e} | Run src/ingestion.py first.")
    st.stop()

# --- Main Page UI ---
st.markdown("<h1 class='main-header'>Multi-LLM RAG Dashboard</h1>", unsafe_allow_html=True)
st.write("<div style='text-align: center; margin-bottom: 2rem;'>Select any LLM to analyze the impact of climate change on Indian water resources.</div>", unsafe_allow_html=True)

# Validation of Keys
if not api_key:
    st.warning(f"⚠️ Please enter your {llm_provider} API key in the sidebar to start!")
else:
    # Setup LLM and Chain
    try:
        llm = get_llm(llm_provider, api_key, model_name)
        chain = get_rag_chain(vector_store, llm)
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        st.stop()

    query = st.text_input("🌍 Ask about Indian Water Sustainability", 
                          placeholder="How does urban flooding affect Bengaluru's drainage system?")

    if query:
        with st.status(f"Querying {llm_provider}...", expanded=True) as status:
            st.write("🔍 Retrieving niche context...")
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(query)
            
            st.write("🧠 Generating answer...")
            start_time = time.time()
            try:
                response = chain.invoke(query)
                end_time = time.time()
                status.update(label=f"Analysis Complete! ({round(end_time - start_time, 2)}s)", state="complete")
            except Exception as e:
                st.error(f"LLM Error: {e}")
                st.stop()

        # Results Display
        st.markdown(f"### 📝 AI Expert Answer ({model_name})")
        st.markdown(f"<div class='glass-card'>{response}</div>", unsafe_allow_html=True)

        with st.expander("📚 Source Evidence (Context)"):
            for i, doc in enumerate(docs):
                st.markdown(f"**Source: {os.path.basename(doc.metadata['source'])}**")
                st.markdown(f"<div class='retrieved-chunk'>{doc.page_content}</div>", unsafe_allow_html=True)
    else:
        st.markdown("---")
        st.markdown("#### 📚 Knowledge Base — 7 Specialized Reports")
        # Feature grid — Row 1
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<div class='glass-card'><h4>🏔️ Himalayan Glaciers</h4>Siachen thinning rates, black carbon albedo, and Third Pole runoff projections.</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='glass-card'><h4>🌧️ Monsoon Variability</h4>Extreme rainfall frequency, break-day spells, and micro-drought creep across India.</div>", unsafe_allow_html=True)
        with col3:
            st.markdown("<div class='glass-card'><h4>💧 Groundwater Crisis</h4>Punjab aquifer depletion, salinity intrusion in KG Delta, and Atal Bhujal Yojana.</div>", unsafe_allow_html=True)
        # Feature grid — Row 2
        col4, col5, col6 = st.columns(3)
        with col4:
            st.markdown("<div class='glass-card'><h4>🌊 Coastal Sea-Level Rise</h4>Mumbai & Chennai inundation risk, NIO 2026 projections, and salinity contamination.</div>", unsafe_allow_html=True)
        with col5:
            st.markdown("<div class='glass-card'><h4>🏙️ Urban Flooding</h4>Bengaluru drain encroachment, Delhi Yamuna breaches, and Sponge City policy.</div>", unsafe_allow_html=True)
        with col6:
            st.markdown("<div class='glass-card'><h4>🌾 Climate-Smart Agriculture</h4>Millet missions, IoT drip irrigation, and treated wastewater reuse for farms.</div>", unsafe_allow_html=True)
        # Feature grid — Row 3 (centered)
        _, col7, _ = st.columns([1, 1, 1])
        with col7:
            st.markdown("<div class='glass-card'><h4>🌉 River Linking Project</h4>Ken-Betwa link, NRLP feasibility under climate shift, and ecological tradeoffs.</div>", unsafe_allow_html=True)

st.caption("AI Engineer Intern Portfolio Project")
