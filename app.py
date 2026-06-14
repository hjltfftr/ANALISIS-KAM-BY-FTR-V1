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
st.title("🚀 KAM Analyzer Pro (With Readability Interpretation)")
st.markdown("Unggah 2-4 dokumen KAM Anda, masukkan API Key, dan biarkan sistem melakukan **ekstraksi, analisis readability beserta interpretasinya, uji kemiripan (boilerplate), ringkasan AI, dan perbandingan komprehensif** dalam satu kali proses.")

# --- FUNGSI INTERPRETASI SKOR ---
def interpret_fre(score):
    if score >= 90: return "Sangat Mudah"
    elif score >= 80: return "Mudah"
    elif score >= 70: return "Agak Mudah"
    elif score >= 60: return "Standar / Sedang"
    elif score >= 50: return "Agak Sulit"
    elif score >= 30: return "Sulit (Tingkat Sarjana)"
    else: return "Sangat Sulit (Profesional / Pascasarjana)"

def interpret_fog(score):
    if score < 6: return "Sangat Mudah"
    elif score <= 8: return "Mudah (Tingkat SMP)"
    elif score <= 12: return "Standar (Tingkat SMA)"
    elif score <= 16: return "Sulit (Tingkat Sarjana)"
    else: return "Sangat Sulit (Profesional / Pascasarjana)"

def interpret_fk(score):
    if score <= 6: return "Mudah (Tingkat SD)"
    elif score <= 9: return "Sedang (Tingkat SMP)"
    elif score <= 12: return "Tinggi (Tingkat SMA)"
    elif score <= 16: return "Sangat Tinggi (Tingkat Kuliah)"
    else: return "Akademik Lanjut / Profesional"

# --- FUNGSI PENDUKUNG ---
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
        return {
            "Word Count": 0, "Sentence Count": 0, 
            "Flesch Reading Ease": 0, "FRE Interpretation": "Teks Kosong",
            "Gunning Fog": 0, "Fog Interpretation": "Teks Kosong",
            "FK Grade": 0, "FK Interpretation": "Teks Kosong"
        }
    
    fre = textstat.flesch_reading_ease(text)
    fog = textstat.gunning_fog(text)
    fk = textstat.flesch_kincaid_grade(text)
    
    return {
        "Word Count": textstat.lexicon_count(text),
        "Sentence Count": textstat.sentence_count(text),
        "Flesch Reading Ease": fre,
        "FRE Interpretation": interpret_fre(fre),
        "Gunning Fog": fog,
        "Fog Interpretation": interpret_fog(fog),
        "FK Grade": fk,
        "FK Interpretation": interpret_fk(fk)
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
        with st.spinner("⏳ [1/4] Mengekstrak teks & menghitung Readability + Interpretasi..."):
            results_readability = []
            for file in uploaded_files:
                text = extract_text_from_pdf(file.getvalue())
                documents[file.name] = text
                
                metrics = calculate_readability(text)
                metrics['Filename'] = file.name
                results_readability.append(metrics)
                
            # Menyusun urutan kolom agar angka berdampingan langsung dengan interpretasinya
            cols_order = [
                'Filename', 'Word Count', 'Sentence Count', 
                'Flesch Reading Ease', 'FRE Interpretation', 
                'Gunning Fog', 'Fog Interpretation', 
                'FK Grade', 'FK Interpretation'
            ]
            df_read = pd.DataFrame(results_readability)[cols_order]
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
            
            # Ringkasan per dokumen
            summaries = {}
            for name, text in documents.items():
                prompt_sum = f"Ringkas Key Audit Matters berikut secara eksekutif (1. Fokus Audit, 2. Alasan, 3. Respons):\n\n{text}"
                try:
                    res = model.generate_content(prompt_sum)
                    summaries[name] = res.text
                except Exception as e:
                    summaries[name] = f"Error: {e}"
            st.session_state['ai_summaries'] = summaries
            
            # Perbandingan Keseluruhan
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
                # Siapkan Dataframe Laporan Utama
                df_final = st.session_state['df_readability'].copy()
                df_final['AI Summary'] = df_final['Filename'].map(st.session_state['ai_summaries'])
                df_final['AI Comparison Analysis'] = ""
                df_final.loc[0, 'AI Comparison Analysis'] = st.session_state['ai_comparison']
                
                df_final.to_excel(writer, sheet_name='Laporan Utama', index=False)
                
                if st.session_state['df_sim'] is not None:
                    st.session_state['df_sim'].to_excel(writer, sheet_name='Matrix Kemiripan')
            
            st.session_state['excel_bytes'] = output.getvalue()
            st.session_state['is_processed'] = True

# --- AREA HASIL (Ditampilkan setelah proses selesai) ---
if st.session_state['is_processed']:
    st.success("✅ Seluruh proses analisis berhasil diselesaikan!")
    st.divider()
    
    st.header("📊 1. Analisis Keterbacaan & Interpretasi Skor")
    st.dataframe(st.session_state['df_readability'], use_container_width=True)
    
    st.header("🔍 2. Peta Kemiripan Teks (Boilerplate)")
    if st.session_state['sim_matrix'] is not None:
        fig = px.imshow(st.session_state['sim_matrix'], 
                        x=st.session_state['doc_names'], y=st.session_state['doc_names'], 
                        color_continuous_scale="YlGnBu", text_auto=".2f")
        st.plotly_chart(fig, use_container_width=True)
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
        label="💾 Download Laporan Lengkap + Interpretasi (Format Excel)",
        data=st.session_state['excel_bytes'],
        file_name="KAM_Analysis_With_Interpretation.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary"
    )
