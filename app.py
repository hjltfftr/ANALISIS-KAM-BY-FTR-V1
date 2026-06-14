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
st.set_page_config(page_title="KAM Analyzer Pro (One-Click)", layout="wide", page_icon="🚀")
st.title("🚀 KAM Analyzer Pro (One-Click Analysis)")
st.markdown("Unggah 2-4 dokumen KAM Anda, masukkan API Key, dan biarkan sistem melakukan **ekstraksi, analisis readability, uji kemiripan (boilerplate), ringkasan AI, dan perbandingan komprehensif** dalam satu kali proses.")

# --- FUNGSI PENDUKUNG ---

def interpret_flesch(score):
    if score >= 60: return "Standar / Mudah"
    elif score >= 30: return "Sulit (Tingkat Akademik/Profesional)"
    else: return "Sangat Sulit (Sangat Kompleks)"

def interpret_similarity(score):
    if score > 0.80: return "Sangat Tinggi (Indikasi kuat Boilerplate / Copy-Paste)"
    elif score > 0.50: return "Sedang (Ada kesamaan format/isu standar industri)"
    else: return "Rendah (Dokumen sangat berbeda / Pendekatan Unik)"

@st.cache_data
def extract_text_from_pdf(file_bytes):
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted: text += extracted + "\n"
    
    if len(text.strip()) < 50:
        try:
            images = convert_from_bytes(file_bytes)
            for img in images: text += pytesseract.image_to_string(img, lang='ind+eng') + "\n" 
        except Exception: pass
    return text

def calculate_readability(text):
    if not text.strip():
        return {"Word Count": 0, "Sentence Count": 0, "Flesch Score": 0, "Interpretasi Flesch": "Teks Kosong", "Gunning Fog": 0, "FK Grade": 0}
    
    flesch_score = textstat.flesch_reading_ease(text)
    return {
        "Word Count": textstat.lexicon_count(text),
        "Sentence Count": textstat.sentence_count(text),
        "Flesch Score": flesch_score,
        "Interpretasi Flesch": interpret_flesch(flesch_score),
        "Gunning Fog": textstat.gunning_fog(text),
        "FK Grade": textstat.flesch_kincaid_grade(text)
    }

def calculate_similarity(texts):
    try:
        vectorizer = TfidfVectorizer() 
        tfidf_matrix = vectorizer.fit_transform(texts)
        return cosine_similarity(tfidf_matrix)
    except ValueError:
        return None

# --- INISIALISASI SESSION STATE ---
if 'is_processed' not in st.session_state:
    st.session_state.update({
        'is_processed': False,
        'df_readability': pd.DataFrame(),
        'df_sim': None,
        'sim_matrix': None,
        'doc_names': [],
        'ai_summaries': {},
        'ai_comparison': "",
        'excel_bytes': None
    })

# --- AREA INPUT ---
st.header("1. Persiapan Data")
col_api, col_upload = st.columns([1, 2])
with col_api:
    api_key = st.text_input("🔑 Masukkan Google Gemini API Key:", type="password")
with col_upload:
    uploaded_files = st.file_uploader("📂 Pilih Dokumen PDF (Disarankan 2-4 File KAM)", type=['pdf'], accept_multiple_files=True)

# --- TOMBOL EKSEKUSI ---
if st.button("⚡ PROSES SELURUH ANALISIS ⚡", use_container_width=True, type="primary"):
    if not api_key:
        st.error("⚠️ Masukkan API Key terlebih dahulu bos!")
    elif not uploaded_files or len(uploaded_files) < 2:
        st.error("⚠️ Mohon unggah minimal 2 dokumen agar fitur perbandingan bisa berjalan.")
    else:
        st.session_state['doc_names'] = [f.name for f in uploaded_files]
        documents = {}
        
        # 1. Ekstraksi Teks & Readability
        with st.spinner("⏳ [1/4] Mengekstrak teks & menghitung Readability..."):
            results_readability = []
            for file in uploaded_files:
                text = extract_text_from_pdf(file.getvalue())
                documents[file.name] = text
                
                metrics = calculate_readability(text)
                metrics['Filename'] = file.name
                results_readability.append(metrics)
                
            df_read = pd.DataFrame(results_readability)[['Filename', 'Word Count', 'Sentence Count', 'Flesch Score', 'Interpretasi Flesch', 'Gunning Fog', 'FK Grade']]
            st.session_state['df_readability'] = df_read
            
        # 2. Boilerplate Analysis (Kemiripan)
        with st.spinner("⏳ [2/4] Menganalisis kemiripan dokumen (Boilerplate)..."):
            doc_texts = list(documents.values())
            sim_matrix = calculate_similarity(doc_texts)
            st.session_state['sim_matrix'] = sim_matrix
            if sim_matrix is not None:
                st.session_state['df_sim'] = pd.DataFrame(sim_matrix, index=st.session_state['doc_names'], columns=st.session_state['doc_names'])

        # 3. AI Summaries & Comparison
        with st.spinner("⏳ [3/4] AI sedang membaca dan menyusun ringkasan..."):
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            summaries = {}
            for name, text in documents.items():
                prompt_sum = f"Ringkas Key Audit Matters berikut secara eksekutif (1. Fokus Audit, 2. Alasan, 3. Respons):\n\n{text}"
                try:
                    res = model.generate_content(prompt_sum)
                    summaries[name] = res.text
                except Exception as e:
                    summaries[name] = f"Error: {e}"
            st.session_state['ai_summaries'] = summaries
            
            combined_texts = ""
            for name, text in documents.items():
                combined_texts += f"\n\n### Dokumen: {name}\n{text}\n"
                
            prompt_comp = f"""Anda adalah auditor senior. Baca dokumen-dokumen Key Audit Matters berikut.
            Buatlah analisis perbandingan antar dokumen tersebut.
            Format jawaban:
            * **Persamaan Risiko Utama**: ...
            * **Perbedaan Signifikan**: ...
            * **Insight Komparatif**: (Dokumen mana yang profil risikonya paling kompleks dan mengapa)
            
            Berikut adalah dokumennya:{combined_texts}"""
            
            try:
                res_comp = model.generate_content(prompt_comp)
                st.session_state['ai_comparison'] = res_comp.text
            except Exception as e:
                st.session_state['ai_comparison'] = f"Error Perbandingan: {e}"

        # 4. Generate Excel
        with st.spinner("⏳ [4/4] Menyusun laporan Excel..."):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_final = st.session_state['df_readability'].copy()
                df_final['AI Summary'] = df_final['Filename'].map(st.session_state['ai_summaries'])
                df_final['AI Comparison Analysis'] = ""
                df_final.loc[0, 'AI Comparison Analysis'] = st.session_state['ai_comparison']
                
                df_final.to_excel(writer, sheet_name='Laporan Utama', index=False)
                
                if st.session_state['df_sim'] is not None:
                    st.session_state['df_sim'].to_excel(writer, sheet_name='Matrix Kemiripan')
            
            st.session_state['excel_bytes'] = output.getvalue()
            st.session_state['is_processed'] = True

# --- AREA HASIL ---
if st.session_state['is_processed']:
    st.success("✅ Seluruh proses analisis berhasil diselesaikan!")
    st.divider()
    
    # Menampilkan Readability beserta Interpretasinya
    st.header("📊 1. Analisis Keterbacaan (Readability)")
    st.markdown("Semakin kecil angka **Flesch Score**, semakin kompleks bahasa yang digunakan auditor dalam menyusun KAM.")
    st.dataframe(st.session_state['df_readability'], use_container_width=True)
    
    # Menampilkan Boilerplate beserta Interpretasinya
    st.header("🔍 2. Peta Kemiripan Teks (Boilerplate)")
    if st.session_state['sim_matrix'] is not None:
        fig = px.imshow(st.session_state['sim_matrix'], 
                        x=st.session_state['doc_names'], y=st.session_state['doc_names'], 
                        color_continuous_scale="YlGnBu", text_auto=".2f")
        st.plotly_chart(fig, use_container_width=True)
        
        # Logika Pencarian Nilai Kemiripan Tertinggi
        sim_matrix_copy = st.session_state['sim_matrix'].copy()
        np.fill_diagonal(sim_matrix_copy, -1) # Hindari 1.0 dari dokumen yang sama
        max_sim = np.max(sim_matrix_copy)
        max_idx = np.unravel_index(np.argmax(sim_matrix_copy, axis=None), sim_matrix_copy.shape)
        
        st.subheader("💡 Kesimpulan Boilerplate:")
        kategori_sim = interpret_similarity(max_sim)
        pesan = f"Kemiripan tertinggi sebesar **{max_sim*100:.1f}%** ditemukan antara dokumen **{st.session_state['doc_names'][max_idx[0]]}** dan **{st.session_state['doc_names'][max_idx[1]]}**.\n\nStatus: **{kategori_sim}**"
        
        if max_sim > 0.80:
            st.warning(pesan)
        elif max_sim > 0.50:
            st.info(pesan)
        else:
            st.success(pesan)
    else:
        st.warning("Teks tidak dapat dibandingkan (mungkin kosong/gagal OCR).")
        
    st.header("🤖 3. Hasil AI (Ringkasan & Perbandingan)")
    col_sum, col_comp = st.columns([1, 1])
    
    with col_sum:
        st.subheader("Ringkasan Per Dokumen")
        for name, summary in st.session_state['ai_summaries'].items():
            with st.expander(f"📄 Ringkasan: {name}"):
                st.write(summary)
                
    with col_comp:
        st.subheader("⚖️ Analisis Perbandingan Keseluruhan")
        st.info(st.session_state['ai_comparison'])
        
    st.divider()
    st.header("📥 Unduh Laporan")
    st.download_button(
        label="💾 Download Laporan Lengkap (Format Excel)",
        data=st.session_state['excel_bytes'],
        file_name="KAM_Full_Analysis_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary"
    )
