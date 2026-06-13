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
st.title("📄 KAM Analyzer (Master Report Format)")

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
            images = convert_from_bytes(file_bytes)
            for img in images:
                text += pytesseract.image_to_string(img, lang='ind+eng') + "\n" 
        except Exception as e:
            st.error(f"⚠️ Mode OCR gagal dijalankan. Detail: {e}")
            
    return text

def calculate_readability(text):
    if not text.strip():
        return {
            "Word Count": 0, "Sentence": 0, 
            "Flesch Reading Ease": 0, "Gunning Fog": 0, "FK Grade": 0
        }
    return {
        "Word Count": textstat.lexicon_count(text),
        "Sentence": textstat.sentence_count(text),
        "Flesch Reading Ease": textstat.flesch_reading_ease(text),
        "Gunning Fog": textstat.gunning_fog(text),
        "FK Grade": textstat.flesch_kincaid_grade(text)
    }

def calculate_similarity(texts):
    try:
        vectorizer = TfidfVectorizer() 
        tfidf_matrix = vectorizer.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf_matrix)
        return sim_matrix
    except ValueError:
        return None

# --- INISIALISASI SESSION STATE (MASTER DATA) ---
if 'documents' not in st.session_state:
    st.session_state['documents'] = {}  

# Membuat struktur DataFrame persis seperti di gambar user
if 'main_df' not in st.session_state:
    st.session_state['main_df'] = pd.DataFrame(columns=[
        'Filename', 'tgl KAM', 'Word Count', 'Sentence', 
        'Flesch Reading Ease', 'Gunning Fog', 'FK Grade', 
        'boilerplate', 'ai summary', 'ai comparison'
    ])

# --- STRUKTUR UI (TABS) ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 Upload PDF", 
    "📖 Readability Analysis", 
    "🔍 Boilerplate Analysis", 
    "🤖 AI Summary & Comparison", 
    "💾 Export Excel"
])

# 1. UPLOAD PDF
with tab1:
    st.header("Upload Dokumen KAM")
    st.info("💡 Aplikasi ini dilengkapi sistem **Auto-OCR**. PDF hasil gambar/scan akan diproses otomatis.")
    
    uploaded_files = st.file_uploader("Pilih file PDF KAM", type=['pdf'], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Proses Dokumen"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            rows = []
            for idx, file in enumerate(uploaded_files):
                status_text.text(f"Memproses file ({idx+1}/{len(uploaded_files)}): {file.name}...")
                
                # Ekstrak Teks
                st.session_state['documents'][file.name] = extract_text_from_pdf(file.getvalue())
                
                # Inisialisasi baris baru dengan struktur persis seperti di gambar
                rows.append({
                    'Filename': file.name,
                    'tgl KAM': '',  # Kosongkan dulu sesuai format gambar
                    'Word Count': 0,
                    'Sentence': 0,
                    'Flesch Reading Ease': 0.0,
                    'Gunning Fog': 0.0,
                    'FK Grade': 0.0,
                    'boilerplate': '',
                    'ai summary': '',
                    'ai comparison': ''
                })
                progress_bar.progress((idx + 1) / len(uploaded_files))
                
            st.session_state['main_df'] = pd.DataFrame(rows)
            status_text.text("Seluruh dokumen berhasil didaftarkan ke Master Report!")
            st.success("Selesai!")

    if not st.session_state['main_df'].empty:
        st.write("### Preview Tabel Master Saat Ini:")
        st.dataframe(st.session_state['main_df'], use_container_width=True)

# 2. READABILITY ANALYSIS
with tab2:
    st.header("Readability Analysis")
    if st.session_state['main_df'].empty:
        st.info("Silakan upload dokumen PDF terlebih dahulu di tab 'Upload PDF'.")
    else:
        if st.button("Jalankan Analisis Keterbacaan"):
            with st.spinner("Menghitung metrik keterbacaan teks..."):
                for idx, row in st.session_state['main_df'].iterrows():
                    name = row['Filename']
                    text = st.session_state['documents'].get(name, "")
                    metrics = calculate_readability(text)
                    
                    # Isi data langsung ke sel Master DataFrame
                    st.session_state['main_df'].at[idx, 'Word Count'] = metrics['Word Count']
                    st.session_state['main_df'].at[idx, 'Sentence'] = metrics['Sentence']
                    st.session_state['main_df'].at[idx, 'Flesch Reading Ease'] = metrics['Flesch Reading Ease']
                    st.session_state['main_df'].at[idx, 'Gunning Fog'] = metrics['Gunning Fog']
                    st.session_state['main_df'].at[idx, 'FK Grade'] = metrics['FK Grade']
            st.success("Metrik keterbacaan berhasil diperbarui di tabel master!")
            
        st.dataframe(st.session_state['main_df'], use_container_width=True)

# 3. BOILERPLATE ANALYSIS
with tab3:
    st.header("Boilerplate Analysis (Kemiripan Teks)")
    if len(st.session_state['main_df']) < 2:
        st.warning("Dibutuhkan minimal 2 dokumen untuk melakukan analisis boilerplate/kemiripan.")
    else:
        doc_names = list(st.session_state['documents'].keys())
        doc_texts = list(st.session_state['documents'].values())
        
        sim_matrix = calculate_similarity(doc_texts)
        
        if sim_matrix is None:
            st.error("❌ Analisis kemiripan gagal.")
        else:
            df_sim = pd.DataFrame(sim_matrix, index=doc_names, columns=doc_names)
            
            # Update status boilerplate di tabel master jika diinginkan
            if st.button("Simpan Status Boilerplate ke Master"):
                st.session_state['main_df']['boilerplate'] = "Teranalisis (Lihat Heatmap)"
                st.success("Status diperbarui!")

            st.subheader("Cosine Similarity Matrix")
            st.dataframe(df_sim.style.background_gradient(cmap='YlGnBu', axis=None))
            
            st.subheader("Heatmap Matriks")
            fig = px.imshow(sim_matrix, x=doc_names, y=doc_names, color_continuous_scale="YlGnBu", text_auto=".2f")
            st.plotly_chart(fig, use_container_width=True)

# 4. AI SUMMARY & COMPARISON
with tab4:
    st.header("🤖 AI Summary & Comparison")
    api_key = st.text_input("🔑 Masukkan Google Gemini API Key Anda:", type="password")
    
    if st.session_state['main_df'].empty:
        st.info("Silakan upload dokumen PDF terlebih dahulu.")
    else:
        mode = st.radio("Pilih Mode Analisis AI:", ["Meringkas 1 Dokumen", "⚖️ Membandingkan 2 Dokumen"], horizontal=True)
        
        # --- MODE 1: RINGKAS SINGLE DOKUMEN ---
        if mode == "Meringkas 1 Dokumen":
            selected_doc = st.selectbox("Pilih dokumen KAM:", st.session_state['main_df']['Filename'].tolist())
            
            if st.button("✨ Generate Summary"):
                if not api_key:
                    st.warning("⚠️ Masukkan API Key Anda.")
                else:
                    doc_text = st.session_state['documents'][selected_doc]
                    with st.spinner("AI sedang menyusun ringkasan..."):
                        try:
                            genai.configure(api_key=api_key)
                            model = genai.GenerativeModel('gemini-2.5-flash')
                            prompt = f"Berikan ringkasan eksekutif singkat untuk teks Key Audit Matters berikut:\n\n{doc_text}"
                            response = model.generate_content(prompt)
                            
                            # MASUKKAN HASIL KE KOLOM 'ai summary' PADA BARIS YANG COCOK
                            st.session_state['main_df'].loc[st.session_state['main_df']['Filename'] == selected_doc, 'ai summary'] = response.text
                            st.success(f"Ringkasan untuk {selected_doc} berhasil disimpan ke kolom master!")
                            st.write(response.text)
                        except Exception as e:
                            st.error(f"Error: {e}")

        # --- MODE 2: BANDINGKAN DUA DOKUMEN ---
        else:
            doc_list = st.session_state['main_df']['Filename'].tolist()
            col_a, col_b = st.columns(2)
            with col_a: doc_a = st.selectbox("Pilih Dokumen A:", doc_list, index=0)
            with col_b: doc_b = st.selectbox("Pilih Dokumen B:", doc_list, index=1 if len(doc_list) > 1 else 0)
            
            if doc_a != doc_b:
                if st.button("🔍 Jalankan Analisis Perbandingan"):
                    if not api_key:
                        st.warning("⚠️ Masukkan API Key.")
                    else:
                        text_a = st.session_state['documents'][doc_a]
                        text_b = st.session_state['documents'][doc_b]
                        with st.spinner("Gemini sedang menganalisis perbedaan..."):
                            try:
                                genai.configure(api_key=api_key)
                                model = genai.GenerativeModel('gemini-2.5-flash')
                                prompt = f"Bandingkan risiko audit utama dari Dokumen A dan Dokumen B berikut secara ringkas:\n\nDokumen A:\n{text_a}\n\nDokumen B:\n{text_b}"
                                response = model.generate_content(prompt)
                                
                                # MASUKKAN HASIL KE KOLOM 'ai comparison' PADA KEDUA DOKUMEN YANG TERLIBAT
                                st.session_state['main_df'].loc[st.session_state['main_df']['Filename'] == doc_a, 'ai comparison'] = f"Dibandingkan dengan {doc_b}: {response.text}"
                                st.session_state['main_df'].loc[st.session_state['main_df']['Filename'] == doc_b, 'ai comparison'] = f"Dibandingkan dengan {doc_a}: {response.text}"
                                
                                st.success("Hasil analisis komparatif berhasil disimpan ke kolom master kedua file!")
                                st.write(response.text)
                            except Exception as e:
                                st.error(f"Error: {e}")

# 5. EXPORT EXCEL
with tab5:
    st.header("Export Hasil Analisis")
    if st.session_state['main_df'].empty:
        st.warning("Belum ada data di dalam Master Report.")
    else:
        st.write("Unduh laporan gabungan satu lembar persis seperti format yang Anda minta.")
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Mengeluarkan DataFrame Utama yang strukturnya persis seperti di gambar foto Anda
            st.session_state['main_df'].to_excel(writer, sheet_name='Master Report', index=False)
            
            # Opsional: Tetap sertakan matriks kesamaan di sheet terpisah jika ada
            if len(st.session_state['documents']) >= 2:
                try:
                    doc_texts = list(st.session_state['documents'].values())
                    sim_matrix = calculate_similarity(doc_texts)
                    if sim_matrix is not None:
                        df_sim = pd.DataFrame(sim_matrix, index=list(st.session_state['documents'].keys()), columns=list(st.session_state['documents'].keys()))
                        df_sim.to_excel(writer, sheet_name='Similarity Matrix')
                except:
                    pass

        processed_data = output.getvalue()
        
        st.download_button(
            label="📥 Download Laporan Excel (Format Sesuai Gambar)",
            data=processed_data,
            file_name="KAM_Master_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
