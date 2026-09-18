from PIL import Image
import streamlit as st
from utils import load_model, process_safety_detection

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Construction Safety Detection",
    page_icon="👷",
    layout="wide",
)

# Judul dan Deskripsi Utama
st.title("👷 Construction Safety Detection App")
st.write(
    "Aplikasi deteksi alat pelindung diri (APD) dan keselamatan konstruksi"
    " berbasis YOLO dan Streamlit dengan Label Pekerja."
)

# Sidebar untuk Pengaturan Model
st.sidebar.header("⚙️ Pengaturan Model")
model_file = st.sidebar.text_input(
    "Nama File Model", value="model_best_10_epoch.pt"
)
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.05
)

# Memuat Model
try:
    model = load_model(model_file)
except Exception as e:
    st.sidebar.error(
        f"Gagal memuat model '{model_file}'. Pastikan file ada di repository."
        f" Error: {e}"
    )
    st.stop()

# Form Unggah Gambar
uploaded_file = st.file_uploader(
    "Pilih gambar untuk dianalisis...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ Gambar Asli")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🔍 Hasil Deteksi & Label Pekerja")
        with st.spinner("Sedang memproses deteksi dan pelabelan..."):
            # Pemanggilan fungsi pemrosesan backend dari utils.py
            (
                res_rgb,
                summary_data,
                complete_count,
                incomplete_count,
                total_persons,
            ) = process_safety_detection(image, model, conf_threshold)

            # Tampilkan Gambar Terannotasi
            st.image(res_rgb, use_container_width=True)

            # Tampilkan Ringkasan Analisis Keselamatan
            st.markdown("### 📊 Analisis Keselamatan Pekerja:")

            if total_persons == 0:
                st.info("Tidak ada 'orang' yang terdeteksi.")
            else:
                col_a, col_b = st.columns(2)
                col_a.metric("Total Orang Lengkap", complete_count)
                col_b.metric("Total Orang Tidak Lengkap", incomplete_count)

                st.markdown("---")

                for data in summary_data:
                    color_style = (
                        "color: green;"
                        if "Lengkap" in data["status"] and "Tidak" not in data["status"]
                        else "color: red;"
                    )
                    st.markdown(
                        f"- **{data['worker']}** (Akurasi: **{data['conf']:.2f}**) - Posisi"
                        f" [{data['position']}] -> Status: <span"
                        f" style='{color_style}'>**{data['status']}**</span>",
                        unsafe_allow_html=True,
                    )