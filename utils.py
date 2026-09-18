import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO


@st.cache_resource
def load_model(model_path):
    """Memuat model YOLO dengan caching Streamlit agar tidak dimuat ulang setiap rerun."""
    model = YOLO(model_path)
    return model


def process_safety_detection(image, model, conf_threshold):
    """
    Memproses gambar untuk deteksi keselamatan kerja (APD).
    
    Returns:
        res_rgb (np.ndarray): Gambar hasil plotting/annotasi dalam format RGB.
        summary_data (list): Detail status keselamatan per pekerja.
        complete_count (int): Jumlah pekerja dengan APD lengkap.
        incomplete_count (int): Jumlah pekerja dengan APD tidak lengkap.
        total_persons (int): Total orang yang terdeteksi.
    """
    # Konversi gambar PIL ke array NumPy
    img_array = np.array(image)

    # Jalankan inferensi YOLO
    results = model.predict(img_array, conf=conf_threshold)

    # Ambil hasil plotting standar dari YOLO
    res_plotted = results[0].plot()

    boxes = results[0].boxes
    persons = []
    helmets = []
    vests = []

    # Kelompokkan hasil deteksi berdasarkan kelas
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

    # Salin gambar hasil plot YOLO untuk ditimpa kotak khusus Person
    annotated_img = res_plotted.copy()

    # Evaluasi setiap orang terdeteksi
    for i, p in enumerate(persons, start=1):
        px1, py1, px2, py2 = map(int, p["box"])
        has_helmet = False
        has_vest = False

        # Cek helm di area atas (0% s.d. 50% dari tinggi badan)
        for h in helmets:
            hx1, hy1, hx2, hy2 = h["box"]
            if px1 <= (hx1 + hx2) / 2 <= px2 and py1 <= hy2 <= (py1 + (py2 - py1) * 0.5):
                has_helmet = True
                break

        # Cek vest di area badan tengah hingga bawah (15% s.d. 95% dari tinggi badan)
        for v in vests:
            vx1, vy1, vx2, vy2 = v["box"]
            if px1 <= (vx1 + vx2) / 2 <= px2 and (py1 + (py2 - py1) * 0.15) <= vy2:
                has_vest = True
                break

        # Tentukan Status Keselamatan
        if has_helmet and has_vest:
            status = "Lengkap (Helm & Vest)"
            complete_count += 1
            box_color = (0, 255, 0)  # Hijau (BGR)
        else:
            missing = []
            if not has_helmet:
                missing.append("Helm")
            if not has_vest:
                missing.append("Vest")
            status = f"Tidak Lengkap (Kurang: {', '.join(missing)})"
            incomplete_count += 1
            box_color = (0, 0, 255)  # Merah (BGR)

        # Gambar Bounding Box khusus Orang
        cv2.rectangle(annotated_img, (px1, py1), (px2, py2), box_color, 3)

        # Buat teks label
        label_text = f"Orang ke-{i} ({p['conf']:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        (text_width, text_height), _ = cv2.getTextSize(
            label_text, font, font_scale, thickness
        )
        
        # Background label
        cv2.rectangle(
            annotated_img,
            (px1, py1 - text_height - 10),
            (px1 + text_width + 10, py1),
            box_color,
            -1,
        )
        # Teks label
        cv2.putText(
            annotated_img,
            label_text,
            (px1 + 5, py1 - 5),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
        )

        # Catat koordinat dan ringkasan
        center_x = int((px1 + px2) / 2)
        center_y = int((py1 + py2) / 2)
        summary_data.append({
            "worker": f"Orang ke-{i}",
            "conf": p["conf"],
            "status": status,
            "position": f"Koordinat Pusat: (X: {center_x}, Y: {center_y})",
        })

    # Konversi dari BGR (OpenCV) ke RGB (Streamlit)
    res_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    return res_rgb, summary_data, complete_count, incomplete_count, len(persons)