import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import io
import google.generativeai as genai

# Konfigurasi Halaman
st.set_page_config(page_title="KAM Analyzer", layout="wide", page_icon="📄")
st.title("📄 KAM (Key Audit Matters) Analyzer")

# --- FUNGSI PENDUKUNG ---

@st.cache_data
def extract_text_from_pdf(file_bytes):
    text = ""
    with pdfplumber.open(file_bytes) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

def calculate_readability(text):
    return {
        "Word Count": textstat.lexicon_count(text),
        "Sentence Count": textstat.sentence_count(text),
        "Flesch Reading Ease": textstat.flesch_reading_ease(text),
        "Gunning Fog": textstat.gunning_fog(text),
        "FK Grade": textstat.flesch_kincaid_grade(text)
    }

def calculate_similarity(texts):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    sim_matrix = cosine_similarity(tfidf_matrix)
    return sim_matrix

# --- INISIALISASI SESSION STATE ---
if 'documents' not in st.session_state:
    st.session_state['documents'] = {}  # Format: {"filename": "text content"}
if 'readability_df' not in st.session_state:
    st.session_state['readability_df'] = pd.DataFrame()

# --- STRUKTUR UI (TABS) ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 Upload PDF", 
    "📖 Readability Analysis", 
    "🔍 Boilerplate Analysis", 
    "🤖 AI Summary", 
    "💾 Export Excel"
])

# 1. UPLOAD PDF
with tab1:
    st.header("Upload Dokumen KAM")
    uploaded_files = st.file_uploader("Pilih file PDF", type=['pdf'], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Proses Dokumen"):
            with st.spinner("Mengekstrak teks dari PDF..."):
                for file in uploaded_files:
                    st.session_state['documents'][file.name] = extract_text_from_pdf(file)
            st.success(f"{len(uploaded_files)} dokumen berhasil diproses!")

    if st.session_state['documents']:
        st.write("### Dokumen yang telah diproses:")
        for name in st.session_state['documents'].keys():
            st.markdown(f"- {name}")

# 2. READABILITY ANALYSIS
with tab2:
    st.header("Readability Analysis")
    if not st.session_state['documents']:
        st.info("Silakan upload dan proses dokumen PDF terlebih dahulu di tab 'Upload PDF'.")
    else:
        if st.button("Jalankan Analisis Keterbacaan"):
            results = []
            for name, text in st.session_state['documents'].items():
                metrics = calculate_readability(text)
                metrics['Filename'] = name
                results.append(metrics)
            
            df_readability = pd.DataFrame(results)
            # Menyusun ulang urutan kolom
            cols = ['Filename', 'Word Count', 'Sentence Count', 'Flesch Reading Ease', 'Gunning Fog', 'FK Grade']
            df_readability = df_readability[cols]
            st.session_state['readability_df'] = df_readability
            
        if not st.session_state['readability_df'].empty:
            st.dataframe(st.session_state['readability_df'], use_container_width=True)

# 3. BOILERPLATE ANALYSIS
with tab3:
    st.header("Boilerplate Analysis (Kemiripan Teks)")
    if len(st.session_state['documents']) < 2:
        st.warning("Dibutuhkan minimal 2 dokumen untuk melakukan analisis boilerplate/kemiripan.")
    else:
        doc_names = list(st.session_state['documents'].keys())
        doc_texts = list(st.session_state['documents'].values())
        
        sim_matrix = calculate_similarity(doc_texts)
        df_sim = pd.DataFrame(sim_matrix, index=doc_names, columns=doc_names)
        
        st.subheader("Cosine Similarity Matrix")
        st.dataframe(df_sim.style.background_gradient(cmap='YlGnBu', axis=None))
        
        st.subheader("Heatmap")
        fig = px.imshow(sim_matrix,
                        labels=dict(x="Dokumen", y="Dokumen", color="Similarity"),
                        x=doc_names,
                        y=doc_names,
                        color_continuous_scale="YlGnBu",
                        text_auto=".2f")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Insight")
        col1, col2 = st.columns(2)
        
        # Mencari nilai kemiripan tertinggi dan terendah (mengabaikan diagonal/dirinya sendiri)
        np.fill_diagonal(sim_matrix, -1) # Set diagonal ke -1 agar tidak terdeteksi sebagai max
        max_idx = np.unravel_index(np.argmax(sim_matrix, axis=None), sim_matrix.shape)
        
        np.fill_diagonal(sim_matrix, 2) # Set diagonal ke 2 agar tidak terdeteksi sebagai min
        min_idx = np.unravel_index(np.argmin(sim_matrix, axis=None), sim_matrix.shape)
        
        with col1:
            st.info(f"**Most Similar Documents:**\n\n📄 {doc_names[max_idx[0]]} \n\n📄 {doc_names[max_idx[1]]}")
        with col2:
            st.info(f"**Most Unique/Least Similar Documents:**\n\n📄 {doc_names[min_idx[0]]} \n\n📄 {doc_names[min_idx[1]]}")

# 4. AI SUMMARY
with tab4:
    st.header("AI Summary")
    st.markdown("Fitur ini menggunakan **Google Gemini AI** untuk mengekstrak dan meringkas risiko audit utama dari dokumen KAM.")
    
    # Input API Key agar aman dan tidak di-hardcode
    api_key = st.text_input("🔑 Masukkan Google Gemini API Key Anda:", type="password", help="Dapatkan API key di aistudio.google.com")
    
    if not st.session_state['documents']:
        st.info("Silakan upload dokumen PDF terlebih dahulu di tab 'Upload PDF'.")
    else:
        selected_doc = st.selectbox("Pilih dokumen KAM yang ingin diringkas:", list(st.session_state['documents'].keys()))
        
        if st.button("✨ Generate Summary"):
            if not api_key:
                st.warning("⚠️ Silakan masukkan API Key Anda terlebih dahulu sebelum meng-generate summary.")
            else:
                with st.spinner("AI sedang membaca dan menyusun ringkasan..."):
                    try:
                        # 1. Konfigurasi API
                        genai.configure(api_key=api_key)
                        
                        # 2. Inisialisasi Model
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        
                        # 3. Menyiapkan Teks dan Prompt
                        doc_text = st.session_state['documents'][selected_doc]
                        
                        prompt = f"""
                        Anda adalah seorang asisten auditor senior yang ahli. 
                        Bacalah teks Key Audit Matters (KAM) berikut dan berikan ringkasan eksekutif.
                        
                        Tolong strukturkan jawaban Anda dengan format berikut:
                        1. **Fokus Audit Utama**: (Sebutkan apa yang menjadi isu utamanya secara singkat)
                        2. **Alasan Menjadi KAM**: (Mengapa hal ini dianggap berisiko tinggi)
                        3. **Respons Auditor**: (Langkah-langkah yang dilakukan auditor untuk memitigasi risiko tersebut, gunakan bullet points)
                        
                        Berikut adalah teks KAM-nya:
                        ---
                        {doc_text}
                        """
                        
                        # 4. Generate Response
                        response = model.generate_content(prompt)
                        
                        # 5. Tampilkan Hasil
                        st.success("Ringkasan berhasil dibuat!")
                        st.markdown("### 📋 Hasil Ringkasan AI:")
                        st.info(response.text)
                        
                    except Exception as e:
                        st.error(f"❌ Terjadi kesalahan saat menghubungi AI: {e}")

# 5. EXPORT EXCEL
with tab5:
    st.header("Export Hasil Analisis")
    if st.session_state['readability_df'].empty:
        st.warning("Belum ada data analisis yang bisa diekspor. Jalankan analisis di tab 'Readability Analysis' terlebih dahulu.")
    else:
        st.write("Unduh laporan lengkap dalam format Excel.")
        
        # Membuat buffer untuk Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Sheet Readability
            st.session_state['readability_df'].to_excel(writer, sheet_name='Readability', index=False)
            
            # Sheet Boilerplate jika ada dokumen lebih dari 1
            if len(st.session_state['documents']) >= 2:
                doc_names = list(st.session_state['documents'].keys())
                doc_texts = list(st.session_state['documents'].values())
                sim_matrix = cosine_similarity(TfidfVectorizer(stop_words='english').fit_transform(doc_texts))
                df_sim = pd.DataFrame(sim_matrix, index=doc_names, columns=doc_names)
                df_sim.to_excel(writer, sheet_name='Similarity Matrix')

        processed_data = output.getvalue()
        
        st.download_button(
            label="📥 Download Laporan Excel",
            data=processed_data,
            file_name="KAM_Analysis_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
