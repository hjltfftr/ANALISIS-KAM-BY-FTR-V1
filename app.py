import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import io
import re
import time 
from pdf2image import convert_from_bytes
import pytesseract
import gspread
from google.oauth2.service_account import Credentials

# Import Library AI
import google.generativeai as genai
from groq import Groq

# Konfigurasi Halaman
st.set_page_config(page_title="KAM Analyzer Pro (Auto-Fallback)", layout="wide", page_icon="🚀")
st.title("🚀 KAM Analyzer Pro (Auto-Fallback AI)")
st.markdown("Unggah dokumen KAM Anda. Sistem otomatis menggunakan **Gemini** terlebih dahulu, dan akan berpindah ke **Groq** jika terkena limit.")

# --- SIDEBAR: PENGATURAN API KEY MANUAL ---
with st.sidebar:
    st.header("⚙️ Pengaturan API Key")
    st.write("Masukkan Key Anda di bawah ini:")
    gemini_key_input = st.text_input("🔑 API Key Google Gemini (Utama)", type="password")
    groq_key_input = st.text_input("🔑 API Key Groq (Cadangan)", type="password")
    
    st.info("💡 Sistem akan mencoba Gemini dulu. Jika limit (Error 429), otomatis pindah pakai Llama-3 (Groq).")

# --- MENGAMBIL URL DARI SECRETS (HANYA UNTUK SPREADSHEET) ---
try:
    SHEET_URL = st.secrets["SPREADSHEET_URL"]
except KeyError:
    SHEET_URL = None

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

def parse_filename(filename):
    clean = re.sub(r'\.pdf$', '', filename, flags=re.IGNORECASE).strip()
    parts = clean.rsplit(' ', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], parts[1]
    return clean, "-"

def calculate_readability(text):
    if not text.strip(): return {"Word Count": 0, "Sentence Count": 0, "Flesch Reading Ease": 0, "Gunning Fog": 0, "FK Grade": 0}
    return {
        "Word Count": textstat.lexicon_count(text),
        "Sentence Count": textstat.sentence_count(text),
        "Flesch Reading Ease": textstat.flesch_reading_ease(text),
        "Gunning Fog": textstat.gunning_fog(text),
        "FK Grade": textstat.flesch_kincaid_grade(text)
    }

def interpret_flesch(score):
    if score >= 60: return "Mudah (Standar)"
    elif score >= 30: return "Sulit (Formal)"
    else: return "Sangat Sulit (Akademis)"

def interpret_grade(score):
    if score < 10: return "Menengah (SMA)"
    elif score <= 16: return "Lanjut (Sarjana)"
    else: return "Pakar (Auditor)"

def interpret_similarity(score):
    if score >= 0.75: return "🔴 Sangat Mirip (Indikasi Boilerplate)"
    elif score >= 0.40: return "🟡 Kemiripan Sedang"
    else: return "🟢 Unik"

def calculate_similarity(texts):
    try:
        vectorizer = TfidfVectorizer() 
        tfidf_matrix = vectorizer.fit_transform(texts)
        return cosine_similarity(tfidf_matrix)
    except ValueError: return None

# --- FUNGSI GENERATE DENGAN AUTO-FALLBACK ---
def generate_ai_with_fallback(prompt, gemini_key, groq_key):
    if not gemini_key and not groq_key:
        return "⚠️ Error: Anda belum memasukkan API Key sama sekali."
    
    # 1. Coba pakai Google Gemini terlebih dahulu
    if gemini_key:
        try:
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            return model.generate_content(prompt).text
        except Exception as e:
            error_msg = str(e).lower()
            # Jika tidak ada key Groq, kembalikan error Gemini
            if not groq_key:
                return f"❌ Error Gemini: {e} (Groq Key tidak diisi untuk cadangan)"
            # Jika error (termasuk 429 Quota), diam-diam lanjut ke blok Groq di bawah
            pass

    # 2. Jika Gemini gagal (atau tidak diisi), pakai Groq
    if groq_key:
        try:
            client = Groq(api_key=groq_key)
            chat = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile"
            )
            return chat.choices[0].message.content
        except Exception as e:
            return f"❌ Error Groq: {str(e)}"
            
    return "❌ Gagal memproses AI."

# --- INISIALISASI SESSION STATE ---
if 'is_processed' not in st.session_state:
    st.session_state.update({
        'is_processed': False,
        'df_readability': pd.DataFrame(),
        'df_boilerplate_db': pd.DataFrame(),
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

if not SHEET_URL:
    st.warning("⚠️ **Link Spreadsheet belum diatur di Secrets!** Fitur sinkronisasi otomatis tidak akan berfungsi penuh.")

uploaded_files = st.file_uploader("📂 Pilih Dokumen PDF (Contoh: ACES 2023.pdf)", type=['pdf'], accept_multiple_files=True)

# --- TOMBOL EKSEKUSI ---
if st.button("⚡ PROSES SELURUH ANALISIS ⚡", use_container_width=True, type="primary"):
    if not gemini_key_input and not groq_key_input:
        st.error("⚠️ Proses dihentikan. Masukkan minimal satu API Key (Gemini atau Groq) di menu sebelah kiri.")
    elif not uploaded_files or len(uploaded_files) < 2:
        st.error("⚠️ Mohon unggah minimal 2 dokumen untuk analisis perbandingan.")
    else:
        st.session_state['doc_names'] = [f.name for f in uploaded_files]
        documents = {}
        
        with st.spinner("⏳ [1/4] Mengekstrak teks & memecah Nama/Tahun..."):
            results_readability = []
            for file in uploaded_files:
                text = extract_text_from_pdf(file.getvalue())
                documents[file.name] = text
                
                metrics = calculate_readability(text)
                emiten, tahun = parse_filename(file.name)
                metrics['Nama Emiten'] = emiten
                metrics['Tahun KAM'] = tahun
                metrics['Filename Asli'] = file.name
                results_readability.append(metrics)
                
            df_read = pd.DataFrame(results_readability)
            df_read['Interpretasi Flesch'] = df_read['Flesch Reading Ease'].apply(interpret_flesch)
            df_read['Interpretasi Gunning Fog'] = df_read['Gunning Fog'].apply(interpret_grade)
            df_read['Interpretasi FK Grade'] = df_read['FK Grade'].apply(interpret_grade)
            
            cols = ['Nama Emiten', 'Tahun KAM', 'Filename Asli', 'Word Count', 'Sentence Count', 'Flesch Reading Ease', 'Interpretasi Flesch', 'Gunning Fog', 'Interpretasi Gunning Fog', 'FK Grade', 'Interpretasi FK Grade']
            st.session_state['df_readability'] = df_read[cols]
            
        with st.spinner("⏳ [2/4] Menganalisis kemiripan dokumen (Boilerplate)..."):
            doc_texts = list(documents.values())
            sim_matrix = calculate_similarity(doc_texts)
            st.session_state['sim_matrix'] = sim_matrix
            
            boilerplate_insights = []
            boilerplate_db = []
            
            if sim_matrix is not None:
                st.session_state['df_sim'] = pd.DataFrame(sim_matrix, index=st.session_state['doc_names'], columns=st.session_state['doc_names'])
                doc_names = st.session_state['doc_names']
                for i in range(len(doc_names)):
                    for j in range(i + 1, len(doc_names)):
                        score = sim_matrix[i][j]
                        interp = interpret_similarity(score)
                        boilerplate_insights.append(f"{doc_names[i]} vs {doc_names[j]} : {score:.2%} -> {interp}")
                        
                        emiten_a, tahun_a = parse_filename(doc_names[i])
                        emiten_b, tahun_b = parse_filename(doc_names[j])
                        boilerplate_db.append({
                            "Dokumen A": doc_names[i],
                            "Emiten A": emiten_a, "Tahun A": tahun_a,
                            "Dokumen B": doc_names[j],
                            "Emiten B": emiten_b, "Tahun B": tahun_b,
                            "Persentase Mirip": f"{score:.2%}",
                            "Interpretasi": interp
                        })
            
            st.session_state['boilerplate_insights'] = "\n".join(boilerplate_insights)
            st.session_state['df_boilerplate_db'] = pd.DataFrame(boilerplate_db)

        with st.spinner(f"⏳ [3/4] AI sedang menyusun ringkasan (Mencoba Gemini, bersiap Groq)..."):
            summaries = {}
            for name, text in documents.items():
                prompt_summary = f"""Anda adalah gabungan seorang Auditor Senior dan Analis Ekuitas Pasar Modal. Ekstrak Key Audit Matters (KAM) berikut dengan singkat dan tajam:
1. Fokus Audit & Alasan: (Apa yang disorot dan kenapa auditor menganggap ini krusial?)
2. Respons Auditor: (Langkah apa yang dilakukan auditor?)
3. Radar Risiko: (Secara fundamental, apa ancaman dari poin ini terhadap likuiditas, laba, atau kelangsungan usaha emiten?)

Teks KAM:
{text}"""
                summaries[name] = generate_ai_with_fallback(prompt_summary, gemini_key_input, groq_key_input)
                time.sleep(2) # Jeda aman
            st.session_state['ai_summaries'] = summaries
            
            combined_texts = ""
            for name, text in documents.items(): combined_texts += f"\n\n### Dokumen: {name}\n{text}\n"
            
            prompt_comp = f"""Anda adalah pakar Gabungan (Auditor Senior & Analis Ekuitas Spesialis IHSG). Analisis komparatif dari kumpulan dokumen Key Audit Matters (KAM) berikut untuk mencari anomali fundamental yang bisa menggerakkan harga saham. 

Gunakan format baku berikut:

1. Persamaan:
(Jelaskan isu apa saja yang sama/berulang antar dokumen. Apakah ini memang risiko wajar di sektornya, atau sekadar template/boilerplate dari tahun ke tahun?)

2. Perbedaan Signifikan:
(Jelaskan secara tajam perbedaan isi KAM antar dokumen atau antar tahun. Adakah akun spesifik, metodologi, atau peringatan auditor yang muncul di satu laporan tapi tidak ada di laporan lain?)

3. Dampak yang Mungkin Terjadi (Market Reaction):
(Berikan analisis YoY/komparatif. Contoh: Karena laporan terbaru menyoroti masalah ini, sentimen pasar akan bergeser menjadi...)

4. Informasi Penting Lainnya (Insight Pelaku Pasar):
(Berikan peringatan spesifik dari temuan ini bagi:
- Investor Institusi / Smart Money: Akumulasi atau Distribusi?
- Ritel: Risiko jebakan psikologis/FOMO?
- Bandar / Market Maker: Momen kerek harga atau buang barang?)

Teks Dokumen:
{combined_texts}"""
            st.session_state['ai_comparison'] = generate_ai_with_fallback(prompt_comp, gemini_key_input, groq_key_input)

        with st.spinner("⏳ [4/4] Menyusun memori data..."):
            st.session_state['is_processed'] = True

# --- AREA HASIL, DOWNLOAD, & SPREADSHEET SYNC ---
if st.session_state['is_processed']:
    st.success("✅ Analisis Berhasil!")
    
    # PERSIAPAN DATAFRAME FINAL
    df_final = st.session_state['df_readability'].copy()
    df_final['AI Summary'] = df_final['Filename Asli'].map(st.session_state['ai_summaries'])
    df_final['Interpretasi Boilerplate (Keseluruhan)'] = ""
    df_final.loc[0, 'Interpretasi Boilerplate (Keseluruhan)'] = st.session_state['boilerplate_insights']
    df_final['AI Comparison Analysis'] = ""
    df_final.loc[0, 'AI Comparison Analysis'] = st.session_state['ai_comparison']
    
    df_boiler = st.session_state['df_boilerplate_db'].copy()

    # --- MEMBUAT FILE EXCEL DI MEMORI UNTUK DOWNLOAD ---
    output_excel = io.BytesIO()
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer: 
        df_final.to_excel(writer, index=False, sheet_name='Laporan Utama')
        df_boiler.to_excel(writer, index=False, sheet_name='Data Boilerplate')
    excel_data = output_excel.getvalue()

    st.divider()
    st.header("📤 Ekspor & Sinkronisasi Data")
    
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        if st.button("🚀 KIRIM DATA KE SPREADSHEET", type="primary", use_container_width=True):
            if not SHEET_URL:
                st.error("⚠️ Link Spreadsheet belum diatur di Streamlit Secrets!")
            else:
                try:
                    with st.spinner("Menghubungkan ke Google Server..."):
                        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
                        skey = st.secrets["gcp_service_account"]
                        credentials = Credentials.from_service_account_info(skey, scopes=scopes)
                        client = gspread.authorize(credentials)
                        
                        sheet = client.open_by_url(SHEET_URL)
                        
                        # 1. Kirim ke Laporan Utama
                        try:
                            ws_utama = sheet.worksheet("Laporan Utama")
                        except gspread.exceptions.WorksheetNotFound:
                            ws_utama = sheet.add_worksheet(title="Laporan Utama", rows="1000", cols="20")
                        
                        if not ws_utama.get_all_values():
                            ws_utama.append_row(df_final.columns.tolist())
                        
                        df_final_clean = df_final.fillna("").astype(str)
                        ws_utama.append_rows(df_final_clean.values.tolist())
                        
                        # 2. Kirim ke Data Boilerplate
                        try:
                            ws_boiler = sheet.worksheet("Data Boilerplate")
                        except gspread.exceptions.WorksheetNotFound:
                            ws_boiler = sheet.add_worksheet(title="Data Boilerplate", rows="1000", cols="10")
                            
                        if not ws_boiler.get_all_values():
                            ws_boiler.append_row(df_boiler.columns.tolist())
                        
                        df_boiler_clean = df_boiler.fillna("").astype(str)
                        ws_boiler.append_rows(df_boiler_clean.values.tolist())
                        
                        st.success("✅ BOOM! Data berhasil ditambahkan ke Spreadsheet Anda!")
                except Exception as e:
                    st.error(f"❌ Gagal mengirim: {e}. Pastikan Service Account email sudah di-invite sebagai Editor di Sheet Anda.")

    with col_btn2:
        st.download_button(
            label="📥 DOWNLOAD HASIL (EXCEL)",
            data=excel_data,
            file_name="Hasil_Analisis_KAM.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="secondary",
            use_container_width=True
        )

    st.divider()
    
    st.header("📊 1. Analisis Keterbacaan")
    st.dataframe(df_final.drop(columns=['AI Summary', 'Interpretasi Boilerplate (Keseluruhan)', 'AI Comparison Analysis']), use_container_width=True)
    
    st.header("🔍 2. Kemiripan Teks (Boilerplate Log)")
    st.dataframe(df_boiler, use_container_width=True)
        
    st.header("🤖 3. Hasil AI (Fundamental & Market Insight)")
    col_sum, col_comp = st.columns([1, 1])
    with col_sum:
        st.subheader("Ringkasan Per Dokumen")
        for name, summary in st.session_state['ai_summaries'].items():
            with st.expander(f"📄 Ringkasan: {name}"): st.write(summary)
    with col_comp:
        st.subheader("⚖️ Analisis Perbandingan Keseluruhan")
        st.info(st.session_state['ai_comparison'])
