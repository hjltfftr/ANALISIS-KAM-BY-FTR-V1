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
st.set_page_config(page_title="KAM Analyzer Pro (Auto-Key)", layout="wide", page_icon="🚀")
st.title("🚀 KAM Analyzer Pro (Auto-Key)")
st.markdown("Unggah 2-4 dokumen KAM Anda. Sistem akan otomatis menggunakan API Key yang tersimpan di server untuk melakukan **ekstraksi, analisis readability, uji kemiripan (boilerplate), ringkasan AI, dan perbandingan komprehensif**.")

# --- MENGAMBIL API KEY DARI STREAMLIT SECRETS ---
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    API_KEY = None

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
        return {"Word Count": 0, "Sentence Count": 0, "Flesch Reading Ease": 0, "Gunning Fog": 0, "FK Grade": 0}
    return {
        "Word Count": textstat.lexicon_count(text),
        "Sentence Count": textstat.sentence_count(text),
        "Flesch Reading Ease": textstat.flesch_reading_ease(text),
        "Gunning Fog": textstat.gunning_fog(text),
        "FK Grade": textstat.flesch_kincaid_grade(text)
    }

def interpret_flesch(score):
    if score >= 60: return "Mudah (Bahasa Standar)"
    elif score >= 30: return "Sulit (Bahasa Formal/Bisnis)"
    else: return "Sangat Sulit (Bahasa Akademis/Hukum)"

def interpret_grade(score):
    if score < 10: return "Tingkat Menengah (SMA)"
    elif score <= 16: return "Tingkat Lanjut (Sarjana/Kuliah)"
    else: return "Tingkat Pakar (Profesional/Auditor)"

def interpret_similarity(score):
    if score >= 0.75: return "🔴 Sangat Mirip (Indikasi Kuat Boilerplate)"
    elif score >= 0.40: return "🟡 Kemiripan Sedang (Sebagian Format Standar)"
    else: return "🟢 Unik (Berbeda Signifikan / Customized)"

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
        'boilerplate_insights': "",
        'excel_bytes': None
    })

# --- AREA INPUT ---
st.header("1. Persiapan Data")

# Jika API Key belum disetting di Streamlit Cloud, tampilkan peringatan
if not API_KEY:
    st.error("⚠️ **API Key belum diatur!** Silakan buka pengaturan Streamlit Cloud (Manage App -> Settings -> Secrets) dan tambahkan `GEMINI_API_KEY = 'kunci_anda_disini'`.")

uploaded_files = st.file_uploader("📂 Pilih Dokumen PDF (Disarankan 2-4 File KAM)", type=['pdf'], accept_multiple_files=True)

# --- TOMBOL EKSEKUSI ---
if st.button("⚡ PROSES SELURUH ANALISIS ⚡", use_container_width=True, type="primary"):
    if not API_KEY:
        st.error("⚠️ Proses dihentikan. Silakan atur GEMINI_API_KEY di menu Secrets Streamlit Cloud terlebih dahulu bos!")
    elif not uploaded_files or len(uploaded_files) < 2:
        st.error("⚠️ Mohon unggah minimal 2 dokumen agar fitur perbandingan bisa berjalan secara optimal.")
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
                
            df_read = pd.DataFrame(results_readability)
            df_read['Interpretasi Flesch'] = df_read['Flesch Reading Ease'].apply(interpret_flesch)
            df_read['Interpretasi Gunning Fog'] = df_read['Gunning Fog'].apply(interpret_grade)
            df_read['Interpretasi FK Grade'] = df_read['FK Grade'].apply(interpret_grade)
            
            cols = ['Filename', 'Word Count', 'Sentence Count', 'Flesch Reading Ease', 'Interpretasi Flesch', 'Gunning Fog', 'Interpretasi Gunning Fog', 'FK Grade', 'Interpretasi FK Grade']
            st.session_state['df_readability'] = df_read[cols]
            
        # 2. Boilerplate Analysis
        with st.spinner("⏳ [2/4] Menganalisis kemiripan dokumen (Boilerplate)..."):
            doc_texts = list(documents.values())
            sim_matrix = calculate_similarity(doc_texts)
            st.session_state['sim_matrix'] = sim_matrix
            
            boilerplate_insights = []
            if sim_matrix is not None:
                st.session_state['df_sim'] = pd.DataFrame(sim_matrix, index=st.session_state['doc_names'], columns=st.session_state['doc_names'])
                doc_names = st.session_state['doc_names']
                for i in range(len(doc_names)):
                    for j in range(i + 1, len(doc_names)):
                        score = sim_matrix[i][j]
                        interp = interpret_similarity(score)
                        boilerplate_insights.append(f"{doc_names[i]} vs {doc_names[j]} : {score:.2%} -> {interp}")
            
            st.session_state['boilerplate_insights'] = "\n".join(boilerplate_insights)

        # 3. AI Summaries & Comparison
        with st.spinner("⏳ [3/4] AI sedang membaca dan menyusun ringkasan..."):
            genai.configure(api_key=API_KEY)
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            summaries = {}
            for name, text in documents.items():
                prompt_sum = f"Ringkas Key Audit Matters berikut secara eksekutif (1. Fokus Audit, 2. Alasan, 3. Respons):\n\n{text}"
                try:
                    res = model.generate_content(prompt_sum)
                    summaries[name] = res.text
                except Exception as e:
                    summaries[name] = f"Error menghubungi AI: {e}. Pastikan API Key di Secrets valid."
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
                st.session_state['ai_comparison'] = f"Error Perbandingan AI: {e}. Pastikan API Key di Secrets valid."

        # 4. Generate Excel
        with st.spinner("⏳ [4/4] Menyusun laporan Excel..."):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_final = st.session_state['df_readability'].copy()
                df_final['AI Summary'] = df_final['Filename'].map(st.session_state['ai_summaries'])
                
                df_final['Interpretasi Boilerplate (Keseluruhan)'] = ""
                df_final.loc[0, 'Interpretasi Boilerplate (Keseluruhan)'] = st.session_state['boilerplate_insights']
                
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
    
    st.header("📊 1. Analisis Keterbacaan & Interpretasi")
    st.dataframe(st.session_state['df_readability'], use_container_width=True)
    
    st.header("🔍 2. Kemiripan Teks & Indikasi Boilerplate")
    col_heatmap, col_insights = st.columns([2, 1])
    with col_heatmap:
        if st.session_state['sim_matrix'] is not None:
            fig = px.imshow(st.session_state['sim_matrix'], 
                            x=st.session_state['doc_names'], y=st.session_state['doc_names'], 
                            color_continuous_scale="YlGnBu", text_auto=".2f")
            st.plotly_chart(fig, use_container_width=True)
    with col_insights:
        st.subheader("💡 Interpretasi Boilerplate")
        st.info(st.session_state['boilerplate_insights'] if st.session_state['boilerplate_insights'] else "Tidak ada data.")
        
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
