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
from pdf2image import convert_from_bytes
import pytesseract

# Konfigurasi Halaman
st.set_page_config(page_title="KAM Analyzer", layout="wide", page_icon="📄")
st.title("📄 KAM Analyzer (Teks & Scan PDF)")

# --- FUNGSI PENDUKUNG ---

@st.cache_data
def extract_text_from_pdf(file_bytes):
    text = ""
    # 1. Coba ekstraksi teks digital normal dulu
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    
    # 2. Jika teks kosong/sangat pendek (berarti PDF gambar/scan), gunakan OCR
    if len(text.strip()) < 50:
        try:
            # Mengubah PDF menjadi gambar di memori
            images = convert_from_bytes(file_bytes)
            
            # Membaca teks menggunakan OCR dengan dukungan Bahasa Indonesia + Inggris
            for img in images:
                text += pytesseract.image_to_string(img, lang='ind+eng') + "\n" 
                
        except Exception as e:
            st.error(f"⚠️ Mode OCR gagal dijalankan. Detail: {e}")
            
    return text

def calculate_readability(text):
    if not text.strip():
        return {
            "Word Count": 0, "Sentence Count": 0, 
            "Flesch Reading Ease": 0, "Gunning Fog": 0, "FK Grade": 0
        }
    return {
        "Word Count": textstat.lexicon_count(text),
        "Sentence Count": textstat.sentence_count(text),
        "Flesch Reading Ease": textstat.flesch_reading_ease(text),
        "Gunning Fog": textstat.gunning_fog(text),
        "FK Grade": textstat.flesch_kincaid_grade(text)
    }

def calculate_similarity(texts):
    try:
        # Tfidf Tanpa filter stop words bahasa inggris agar teks indonesia aman
        vectorizer = TfidfVectorizer() 
        tfidf_matrix = vectorizer.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf_matrix)
        return sim_matrix
    except ValueError:
        # Mengembalikan None jika seluruh dokumen teksnya kosong/gagal OCR
        return None

# --- INISIALISASI SESSION STATE ---
if 'documents' not in st.session_state:
    st.session_state['documents'] = {}  
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
    st.info("💡 Aplikasi ini dilengkapi sistem **Auto-OCR**. PDF hasil gambar/scan akan diproses otomatis (proses memakan waktu sedikit lebih lama).")
    
    uploaded_files = st.file_uploader("Pilih file PDF KAM (Disarankan halaman KAM saja)", type=['pdf'], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Proses Dokumen"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, file in enumerate(uploaded_files):
                status_text.text(f"Memproses file ({idx+1}/{len(uploaded_files)}): {file.name}...")
                st.session_state['documents'][file.name] = extract_text_from_pdf(file.getvalue())
                progress_bar.progress((idx + 1) / len(uploaded_files))
                
            status_text.text("Seluruh dokumen berhasil diproses!")
            st.success("Selesai!")

    if st.session_state['documents']:
        st.write("### Dokumen yang siap dianalisis:")
        for name in st.session_state['documents'].keys():
            st.markdown(f"- {name}")

# 2. READABILITY ANALYSIS
with tab2:
    st.header("Readability Analysis")
    if not st.session_state['documents']:
        st.info("Silakan upload dokumen PDF terlebih dahulu di tab 'Upload PDF'.")
    else:
        if st.button("Jalankan Analisis Keterbacaan"):
            results = []
            for name, text in st.session_state['documents'].items():
                metrics = calculate_readability(text)
                metrics['Filename'] = name
                results.append(metrics)
            
            df_readability = pd.DataFrame(results)
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
        
        if sim_matrix is None:
            st.error("❌ Analisis kemiripan gagal karena teks di dalam dokumen kosong atau sistem OCR belum berhasil mengekstrak karakter.")
        else:
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
            
            sim_matrix_insight = sim_matrix.copy()
            np.fill_diagonal(sim_matrix_insight, -1)
            max_idx = np.unravel_index(np.argmax(sim_matrix_insight, axis=None), sim_matrix_insight.shape)
            
            np.fill_diagonal(sim_matrix_insight, 2)
            min_idx = np.unravel_index(np.argmin(sim_matrix_insight, axis=None), sim_matrix_insight.shape)
            
            with col1:
                st.info(f"**Most Similar Documents:**\n\n📄 {doc_names[max_idx[0]]} \n\n📄 {doc_names[max_idx[1]]}")
            with col2:
                st.info(f"**Most Unique/Least Similar Documents:**\n\n📄 {doc_names[min_idx[0]]} \n\n📄 {doc_names[min_idx[1]]}")

# 4. AI SUMMARY
with tab4:
    st.header("AI Summary")
    st.markdown("Fitur ini menggunakan **Google Gemini AI** untuk mengekstrak risiko audit utama langsung dari teks KAM.")
    
    api_key = st.text_input("🔑 Masukkan Google Gemini API Key Anda:", type="password", help="Dapatkan API key gratis di aistudio.google.com")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        cek_model = st.button("🛠️ Cek Model Tersedia (Debug)")
    
    if cek_model:
        if not api_key:
            st.warning("Masukkan API Key dulu bos!")
        else:
            try:
                genai.configure(api_key=api_key)
                available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                st.success("API Key Valid! Ini daftar model yang bisa bos pakai:")
                st.write(available_models)
            except Exception as e:
                st.error(f"Gagal mengecek API Key: {e}")
    
    if not st.session_state['documents']:
        st.info("Silakan upload dokumen PDF terlebih dahulu di tab 'Upload PDF'.")
    else:
        selected_doc = st.selectbox("Pilih dokumen KAM yang ingin diringkas:", list(st.session_state['documents'].keys()))
        
        if st.button("✨ Generate Summary"):
            if not api_key:
                st.warning("⚠️ Silakan masukkan API Key Anda terlebih dahulu.")
            else:
                doc_text = st.session_state['documents'][selected_doc]
                
                if not doc_text.strip():
                    st.error("❌ Teks dokumen kosong. AI tidak dapat membuat ringkasan.")
                else:
                    with st.spinner("AI sedang menyusun ringkasan eksekutif..."):
                        try:
                            genai.configure(api_key=api_key)
                            # Memaksa menggunakan 1.5 flash versi terbaru
                            model = genai.GenerativeModel('gemini-2.5-flash')
                            
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
                            
                            response = model.generate_content(prompt)
                            
                            st.success("Ringkasan berhasil dibuat!")
                            st.markdown("### 📋 Hasil Ringkasan AI:")
                            st.info(response.text)
                            
                        except Exception as e:
                            st.error(f"❌ Gagal menghubungi AI. Detail error: {e}")

# 5. EXPORT EXCEL
with tab5:
    st.header("Export Hasil Analisis")
    if st.session_state['readability_df'].empty:
        st.warning("Belum ada data analisis. Silakan jalankan analisis di tab 'Readability Analysis' terlebih dahulu.")
    else:
        st.write("Unduh laporan lengkap dalam format Excel.")
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            st.session_state['readability_df'].to_excel(writer, sheet_name='Readability', index=False)
            
            if len(st.session_state['documents']) >= 2:
                doc_names = list(st.session_state['documents'].keys())
                doc_texts = list(st.session_state['documents'].values())
                sim_matrix = calculate_similarity(doc_texts)
                if sim_matrix is not None:
                    df_sim = pd.DataFrame(sim_matrix, index=doc_names, columns=doc_names)
                    df_sim.to_excel(writer, sheet_name='Similarity Matrix')

        processed_data = output.getvalue()
        
        st.download_button(
            label="📥 Download Laporan Excel",
            data=processed_data,
            file_name="KAM_Analysis_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
