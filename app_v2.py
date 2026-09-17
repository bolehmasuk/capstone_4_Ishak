import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Construction Safety Detection",
    page_icon="👷",
    layout="wide",
)


# Load Model YOLO (Menggunakan cache agar model tidak dimuat ulang terus-menerus)
@st.cache_resource
def load_model(model_path):
  model = YOLO(model_path)
  return model


# Judul dan Deskripsi
st.title("👷 Construction Safety Detection App")
st.write(
    "Aplikasi deteksi alat pelindung diri (APD) dan keselamatan konstruksi"
    " berbasis YOLO dan Streamlit dengan Label Pekerja."
)

# Sidebar untuk Pengaturan
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

# Unggah Gambar
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
      # Konversi gambar PIL ke array NumPy
      img_array = np.array(image)

      # Jalankan inferensi YOLO
      results = model.predict(img_array, conf=conf_threshold)

      # Ambil hasil plotting standar dari YOLO (untuk helm, vest, dll)
      res_plotted = results[0].plot()

      # --- CUSTOM DRAWING: Tambahkan Label "Person 1", "Person 2", dst. ---
      boxes = results[0].boxes
      persons = []
      helmets = []
      vests = []

      for box in boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id].lower()
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy()

        if "person" in class_name:
          persons.append({"box": xyxy, "conf": conf})
        elif "helmet" in class_name or "head" in class_name:
          helmets.append({"box": xyxy, "conf": conf})
        elif "vest" in class_name:
          vests.append({"box": xyxy, "conf": conf})

      complete_count = 0
      incomplete_count = 0
      summary_data = []

      # Salin gambar yang sudah di-plot YOLO untuk ditimpa kotak khusus Person
      annotated_img = res_plotted.copy()

      # Ganti bagian perulangan pengecekan has_helmet dan has_vest dengan kode ini:

      for i, p in enumerate(persons, start=1):
        px1, py1, px2, py2 = map(int, p["box"])
        has_helmet = False
        has_vest = False

        # Diperluas: Cek helm di area atas (0% s.d. 50% dari tinggi badan)
        for h in helmets:
          hx1, hy1, hx2, hy2 = h["box"]
          # Cek apakah titik tengah horizontal helm berada di dalam rentang badan orang
          if px1 <= (hx1 + hx2) / 2 <= px2 and py1 <= hy2 <= (py1 + (py2 - py1) * 0.5):
            has_helmet = True
            break

        # Diperluas: Cek vest di area badan tengah hingga bawah (15% s.d. 95% dari tinggi badan)
        for v in vests:
          vx1, vy1, vx2, vy2 = v["box"]
          # Cek apakah titik tengah horizontal vest berada di dalam rentang badan orang
          if px1 <= (vx1 + vx2) / 2 <= px2 and (py1 + (py2 - py1) * 0.15) <= vy2:
            has_vest = True
            break

        # Tentukan Status
        if has_helmet and has_vest:
          status = "Lengkap (Helm & Vest)"
          complete_count += 1
          box_color = (0, 255, 0)  # Hijau untuk lengkap
        else:
          missing = []
          if not has_helmet:
            missing.append("Helm")
          if not has_vest:
            missing.append("Vest")
          status = f"Tidak Lengkap (Kurang: {', '.join(missing)})"
          incomplete_count += 1
          box_color = (0, 0, 255)  # Merah untuk tidak lengkap

        # Gambar Bounding Box khusus Orang dengan nomor urut
        cv2.rectangle(
            annotated_img, (px1, py1), (px2, py2), box_color, 3
        )  # Ketebalan garis 3

        # Label teks di atas kotak orang
        label_text = f"Orang ke-{i} ({p['conf']:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        # Hitung ukuran teks untuk latar belakang label
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, font, font_scale, thickness
        )
        cv2.rectangle(
            annotated_img,
            (px1, py1 - text_height - 10),
            (px1 + text_width + 10, py1),
            box_color,
            -1,
        )
        cv2.putText(
            annotated_img,
            label_text,
            (px1 + 5, py1 - 5),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
        )

        # Simpan ringkasan posisi
        center_x = int((px1 + px2) / 2)
        center_y = int((py1 + py2) / 2)
        summary_data.append({
            "worker": f"Orang ke-{i}",
            "conf": p["conf"],
            "status": status,
            "position": f"Koordinat Pusat: (X: {center_x}, Y: {center_y})",
        })

      # Konversi BGR (OpenCV) ke RGB (Streamlit)
      res_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
      st.image(res_rgb, use_container_width=True)

      # --- TAMPILKAN ANALISIS KESELAMATAN ---
      st.markdown("### 📊 Analisis Keselamatan Pekerja:")

      if len(persons) == 0:
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