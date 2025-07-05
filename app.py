import sys
import os
import cv2
import time
import numpy as np
from datetime import datetime
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QWidget,
    QVBoxLayout, QTabWidget, QListWidget, QFileDialog, QHBoxLayout, QMessageBox,
    QSplitter, QSizePolicy, QComboBox
)

import mediapipe as mp

# ==== Cấu hình mặc định ====
logo_path = ""
frame_overlay_path = ""
show_logo = True
show_frame = True
frame_width, frame_height = 1920, 1080
capture_delay = 5
show_time = True
flip_camera = True
save_dir = "captured_images"
os.makedirs(save_dir, exist_ok=True)

# ==== Hàm lấy danh sách độ phân giải hỗ trợ ====

def get_supported_resolutions():
    # Thử các độ phân giải phổ biến, trả về list theo thứ tự từ cao đến thấp
    resolutions = [
        (3840, 2160),  # 4K UHD
        (4096, 2160),  # DCI 4K
        (3200, 1800),  # QHD+
        (2560, 1440),  # QHD (2K)
        (2560, 1600),  # WQXGA
        (2048, 1536),  # QXGA
        (1920, 1200),  # WUXGA
        (1920, 1080),  # FHD
        (1600, 1200),  # UXGA
        (1440, 1080),  # HD+
        (1366, 768),   # WXGA
        (1280, 1024),  # SXGA
        (1280, 960),
        (1280, 800),   # WXGA
        (1280, 720),   # HD
        (1024, 768),   # XGA
        (800, 600),    # SVGA
        (640, 480),    # VGA
        (320, 240),    # QVGA
    ]
    cap = cv2.VideoCapture(0)
    supported = []
    for w, h in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if abs(real_w - w) < 20 and abs(real_h - h) < 20:
            supported.append(f"{real_w}x{real_h}")
    cap.release()
    # Loại trùng và giữ thứ tự
    unique = []
    for r in supported:
        if r not in unique:
            unique.append(r)
    return unique if unique else ["1280x720", "640x480"]

# ==== Hàm overlay PNG (có alpha) ====
def overlay_image_alpha(bg, overlay, x, y):
    if overlay.shape[2] == 3:  # Nếu không có alpha, tạo alpha trắng toàn bộ
        b, g, r = cv2.split(overlay)
        a = np.ones_like(b, dtype=np.uint8) * 255
        overlay = cv2.merge((b, g, r, a))
    elif overlay.shape[2] < 3:
        return bg

    b, g, r, a = cv2.split(overlay)
    overlay_rgb = cv2.merge((b, g, r))
    mask = cv2.merge((a, a, a)) / 255.0

    h, w = overlay.shape[:2]
    bg_h, bg_w = bg.shape[:2]
    if y + h > bg_h: h = bg_h - y
    if x + w > bg_w: w = bg_w - x
    overlay_rgb = overlay_rgb[:h, :w]
    mask = mask[:h, :w]

    roi = bg[y:y + h, x:x + w]
    blended = roi * (1 - mask) + overlay_rgb * mask
    bg[y:y + h, x:x + w] = blended.astype(np.uint8)
    return bg


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def count_fingers(hand_landmarks, hand_label):
    tips = [8, 12, 16, 20]
    count = 0
    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            count += 1
    if hand_label == "Right":
        if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
            count += 1
    else:
        if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
            count += 1
    return count

class CameraTab(QWidget):
    def __init__(self):
        super().__init__()
        self.logo_path = logo_path
        self.frame_path = frame_overlay_path
        self.show_logo = show_logo
        self.show_frame = show_frame
        self.last_frame = None  # Lưu frame mới nhất để scale lại khi resize

        self.init_ui()
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
        self.capturing = False
        self.capture_start_time = None

    def init_ui(self):
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.setScaledContents(False)
        self.image_label.setMinimumSize(1, 1)      # Cho phép thu nhỏ tối đa
        self.image_label.setMaximumSize(16777215, 16777215)  # Không giới hạn tối đa

        self.capture_button = QPushButton("Chụp ngay")
        self.capture_button.clicked.connect(self.manual_capture)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.capture_button)
        layout.setStretch(0, 1)  # Ưu tiên chiếm chỗ cho image_label
        layout.setStretch(1, 0)
        self.setLayout(layout)
        self.setMinimumSize(1, 1)
        self.setMaximumSize(16777215, 16777215)


    def update_frame(self):
        global frame_width, frame_height
        ret, frame_raw = self.cap.read()
        if not ret:
            return

        frame = cv2.resize(frame_raw.copy(), (frame_width, frame_height))
        if flip_camera:
            frame = cv2.flip(frame, 1)

        if self.show_frame and self.frame_path and os.path.exists(self.frame_path):
            overlay = cv2.imread(self.frame_path, cv2.IMREAD_UNCHANGED)
            if overlay is not None:
                if overlay.shape[0] != frame_height or overlay.shape[1] != frame_width:
                    overlay = cv2.resize(overlay, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
                if flip_camera:
                    overlay = cv2.flip(overlay, 1)
                frame = overlay_image_alpha(frame, overlay, 0, 0)

        if self.show_logo and self.logo_path and os.path.exists(self.logo_path):
            logo_img = cv2.imread(self.logo_path, cv2.IMREAD_UNCHANGED)
            if logo_img is not None:
                max_logo_w = frame_width // 4
                max_logo_h = frame_height // 4
                if logo_img.shape[1] > max_logo_w or logo_img.shape[0] > max_logo_h:
                    logo_img = cv2.resize(logo_img, (max_logo_w, max_logo_h), interpolation=cv2.INTER_AREA)
                if flip_camera:
                    logo_img = cv2.flip(logo_img, 1)
                    x = 20
                else:
                    x = frame_width - logo_img.shape[1] - 20
                y = 20
                frame = overlay_image_alpha(frame, logo_img, x, y)

        frame_rgb = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        finger_count = 0

        if results.multi_hand_landmarks:
            for handLms, handInfo in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handInfo.classification[0].label
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                finger_count = count_fingers(handLms, label)

        if not self.capturing and finger_count == 2:
            self.capturing = True
            self.capture_start_time = time.time()
        elif self.capturing and finger_count == 4:
            self.capturing = False
            self.capture_start_time = None

        if self.capturing:
            elapsed = time.time() - self.capture_start_time
            remaining = max(0, int(capture_delay - elapsed))
            cv2.putText(frame, f"Chup sau: {remaining}s", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            if elapsed >= capture_delay:
                self.capture_image(frame_raw)
                self.capturing = False

        cv2.putText(frame, f"Gesture: {finger_count} ngon", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        self.last_frame = frame.copy()
        self.display_scaled_frame(self.last_frame)

    def display_scaled_frame(self, frame):
        h, w, ch = frame.shape
        label_w = self.image_label.width()
        label_h = self.image_label.height()
        if label_w == 0 or label_h == 0:
            return
        scale = min(label_w / w, label_h / h)
        show_w = int(w * scale)
        show_h = int(h * scale)
        frame_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(frame_display.data, w, h, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            show_w, show_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)

    def resizeEvent(self, event):
        if self.last_frame is not None:
            self.display_scaled_frame(self.last_frame)
        super().resizeEvent(event)

    def manual_capture(self):
        ret, frame_raw = self.cap.read()
        if ret:
            self.capture_image(frame_raw)

    def capture_image(self, frame_raw):
        frame_clean = cv2.resize(frame_raw.copy(), (frame_width, frame_height))
        if self.show_frame and self.frame_path and os.path.exists(self.frame_path):
            overlay = cv2.imread(self.frame_path, cv2.IMREAD_UNCHANGED)
            if overlay is not None:
                if overlay.shape[0] != frame_height or overlay.shape[1] != frame_width:
                    overlay = cv2.resize(overlay, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
                frame_clean = overlay_image_alpha(frame_clean, overlay, 0, 0)
        if self.show_logo and self.logo_path and os.path.exists(self.logo_path):
            logo_img = cv2.imread(self.logo_path, cv2.IMREAD_UNCHANGED)
            if logo_img is not None:
                max_logo_w = frame_width // 4
                max_logo_h = frame_height // 4
                if logo_img.shape[1] > max_logo_w or logo_img.shape[0] > max_logo_h:
                    logo_img = cv2.resize(logo_img, (max_logo_w, max_logo_h), interpolation=cv2.INTER_AREA)
                x = frame_width - logo_img.shape[1] - 20
                y = 20
                frame_clean = overlay_image_alpha(frame_clean, logo_img, x, y)
        if show_time:
            time_str = datetime.now().strftime("%d/%m/%Y")
            cv2.putText(frame_clean, time_str, (frame_width - 200, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f"user_{timestamp}.jpg")
        cv2.imwrite(filename, frame_clean)
        print(f"📸 Đã lưu ảnh: {filename}")

class GalleryTab(QWidget):
    def __init__(self):
        super().__init__()
        self.list_widget = QListWidget()
        self.list_widget.itemClicked.connect(self.display_image)
        self.selected_image_path = None

        self.image_label = QLabel("Chọn ảnh để xem")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.refresh_button = QPushButton("Làm mới")
        self.refresh_button.clicked.connect(self.load_images)
        self.refresh_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.download_button = QPushButton("Tải ảnh")
        self.download_button.clicked.connect(self.download_image)
        self.delete_button = QPushButton("Xoá ảnh")
        self.delete_button.clicked.connect(self.delete_image)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.list_widget)
        left_layout.addWidget(self.refresh_button)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.image_label)
        button_row = QHBoxLayout()
        button_row.addWidget(self.download_button)
        button_row.addWidget(self.delete_button)
        right_layout.addLayout(button_row)

        splitter = QSplitter()
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(1, 2)

        main_layout = QHBoxLayout()
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

        self.load_images()

    def load_images(self):
        self.list_widget.clear()
        for file in sorted(os.listdir(save_dir)):
            if file.endswith(".jpg"):
                self.list_widget.addItem(file)

    def display_image(self, item):
        image_path = os.path.join(save_dir, item.text())
        pixmap = QPixmap(image_path).scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)
        self.selected_image_path = image_path

    def download_image(self):
        if self.selected_image_path:
            dest_path, _ = QFileDialog.getSaveFileName(self, "Lưu ảnh", os.path.basename(self.selected_image_path), "Images (*.jpg)")
            if dest_path:
                try:
                    with open(self.selected_image_path, "rb") as fsrc, open(dest_path, "wb") as fdst:
                        fdst.write(fsrc.read())
                    QMessageBox.information(self, "Thành công", "Ảnh đã được tải xuống.")
                except Exception as e:
                    QMessageBox.critical(self, "Lỗi", str(e))

    def delete_image(self):
        if self.selected_image_path:
            reply = QMessageBox.question(self, "Xác nhận xoá", "Bạn có chắc muốn xoá ảnh này?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                try:
                    os.remove(self.selected_image_path)
                    self.selected_image_path = None
                    self.image_label.setText("Chọn ảnh để xem")
                    self.load_images()
                except Exception as e:
                    QMessageBox.critical(self, "Lỗi", str(e))

class SettingsTab(QWidget):
    def __init__(self):
        super().__init__()
        from PyQt6.QtWidgets import QSpinBox, QDoubleSpinBox, QCheckBox
        BUTTON_WIDTH = 240

        layout = QVBoxLayout()
        layout.addWidget(QLabel("🎥 Cài đặt camera"))
        
        # Độ phân giải
        self.resolution_label = QLabel("Độ phân giải:")
        self.resolution_combo = QComboBox()
        self.resolutions = get_supported_resolutions()
        self.resolution_combo.addItems(self.resolutions)
        self.resolution_combo.setFixedWidth(BUTTON_WIDTH)
        # Chọn mặc định khớp với cấu hình
        current_res = f"{frame_width}x{frame_height}"
        idx = self.resolution_combo.findText(current_res)
        self.resolution_combo.setCurrentIndex(idx if idx >= 0 else 0)
        layout.addWidget(self.resolution_label)
        layout.addWidget(self.resolution_combo)
        
        self.flip_checkbox = QCheckBox("Lật camera")
        self.flip_checkbox.setChecked(flip_camera)
        layout.addWidget(self.flip_checkbox)

        # ==== Preview group ====
        layout.addWidget(QLabel("🖼️ Logo và khung ảnh"))
        self.preview_layout = QHBoxLayout()

        self.logo_button = QPushButton("Chọn logo")
        self.logo_button.setFixedWidth(BUTTON_WIDTH)
        self.logo_button.clicked.connect(self.select_logo)
        self.frame_button = QPushButton("Chọn khung ảnh")
        self.frame_button.setFixedWidth(BUTTON_WIDTH)
        self.frame_button.clicked.connect(self.select_frame)
        self.toggle_logo_button = QPushButton("Ẩn logo")
        self.toggle_logo_button.setFixedWidth(BUTTON_WIDTH)
        self.logo_visible = True
        self.toggle_logo_button.clicked.connect(self.toggle_logo)
        self.toggle_frame_button = QPushButton("Ẩn khung ảnh")
        self.toggle_frame_button.setFixedWidth(BUTTON_WIDTH)
        self.frame_visible = True
        self.toggle_frame_button.clicked.connect(self.toggle_frame)

        btn_vbox = QVBoxLayout()
        btn_vbox.addWidget(self.logo_button)
        btn_vbox.addWidget(self.frame_button)
        btn_vbox.addWidget(self.toggle_logo_button)
        btn_vbox.addWidget(self.toggle_frame_button)
        btn_vbox.addStretch()

        btn_widget = QWidget()
        btn_widget.setLayout(btn_vbox)
        self.preview_layout.addWidget(btn_widget)

        self.preview_label = QLabel()
        self.preview_label.setStyleSheet("background: #222; border: 1px solid #666;")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview_label.setMinimumSize(200, 150)
        self.preview_label.setMaximumSize(10000, 10000)
        self.preview_layout.addWidget(self.preview_label, stretch=2)
        layout.addLayout(self.preview_layout)

        layout.addWidget(QLabel("⏱️ Delay chụp khi dùng hand sign"))
        self.delay_spin = QSpinBox()
        self.delay_spin.setRange(2, 15)
        self.delay_spin.setValue(capture_delay)
        self.delay_spin.setFixedWidth(BUTTON_WIDTH)
        layout.addWidget(self.delay_spin)

        layout.addWidget(QLabel("🖼️ Cài đặt ảnh đã chụp"))
        self.display_size_label = QLabel("Hiển thị ảnh: vừa với khung")
        layout.addWidget(self.display_size_label)

        layout.addWidget(QLabel("📏 Hỗ trợ canh chỉnh khung chụp"))
        self.guideline_checkbox = QCheckBox("Hiện đường hỗ trợ")
        layout.addWidget(self.guideline_checkbox)

        layout.addWidget(QLabel("🤖 Cài đặt mô hình"))
        self.hand_count_spin = QSpinBox()
        self.hand_count_spin.setRange(1, 2)
        self.hand_count_spin.setValue(1)
        self.hand_count_spin.setFixedWidth(BUTTON_WIDTH)
        layout.addWidget(QLabel("Số tay nhận diện:"))
        layout.addWidget(self.hand_count_spin)

        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.1, 1.0)
        self.confidence_spin.setSingleStep(0.1)
        self.confidence_spin.setValue(0.7)
        self.confidence_spin.setFixedWidth(BUTTON_WIDTH)
        layout.addWidget(QLabel("Độ tin cậy:"))
        layout.addWidget(self.confidence_spin)

        self.apply_button = QPushButton("Lưu & Áp dụng")
        self.apply_button.setFixedWidth(BUTTON_WIDTH)
        layout.addWidget(self.apply_button)

        self.setLayout(layout)
        self.update_preview()
    
    def resizeEvent(self, event):
        self.update_preview()
        super().resizeEvent(event)

    def update_preview(self):
        w = max(200, self.preview_label.width())
        h = max(150, self.preview_label.height())
        bg = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 220
        if self.frame_visible and frame_overlay_path and os.path.exists(frame_overlay_path):
            overlay = cv2.imread(frame_overlay_path, cv2.IMREAD_UNCHANGED)
            if overlay is not None:
                bg = overlay_image_alpha(bg, overlay, 0, 0)
        if self.logo_visible and logo_path and os.path.exists(logo_path):
            logo_img = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
            if logo_img is not None:
                x = frame_width - logo_img.shape[1] - 20
                y = 20
                bg = overlay_image_alpha(bg, logo_img, x, y)
        preview = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
        preview = cv2.resize(preview, (w, h))
        qimg = QImage(preview.data, preview.shape[1], preview.shape[0], QImage.Format.Format_RGB888)
        self.preview_label.setPixmap(QPixmap.fromImage(qimg))

    def select_logo(self):
        global logo_path
        path, _ = QFileDialog.getOpenFileName(self, "Chọn logo", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            logo_path = path
            self.update_preview()

    def select_frame(self):
        global frame_overlay_path
        path, _ = QFileDialog.getOpenFileName(self, "Chọn khung ảnh", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            frame_overlay_path = path
            self.update_preview()

    def toggle_logo(self):
        self.logo_visible = not self.logo_visible
        self.toggle_logo_button.setText("Hiện logo" if not self.logo_visible else "Ẩn logo")
        global show_logo
        show_logo = self.logo_visible
        self.update_preview()

    def toggle_frame(self):
        self.frame_visible = not self.frame_visible
        self.toggle_frame_button.setText("Hiện khung ảnh" if not self.frame_visible else "Ẩn khung ảnh")
        global show_frame
        show_frame = self.frame_visible
        self.update_preview()


class MainWindow(QMainWindow):
    def apply_settings(self):
        settings_tab = self.settings_tab
        global frame_width, frame_height, capture_delay, flip_camera, hands, logo_path, frame_overlay_path, show_logo, show_frame

        # Độ phân giải
        selected_res = settings_tab.resolution_combo.currentText()
        try:
            w, h = map(int, selected_res.split('x'))
            self.camera_tab.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.camera_tab.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            # Đọc lại giá trị thực tế
            real_w = int(self.camera_tab.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            real_h = int(self.camera_tab.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_width, frame_height = real_w, real_h
            settings_tab.resolution_label.setText(f"Độ phân giải thực tế: {real_w}x{real_h}")
        except Exception as e:
            print("Resolution set error:", e)

        capture_delay = settings_tab.delay_spin.value()
        flip_camera = settings_tab.flip_checkbox.isChecked()
        try:
            hands.close()
        except:
            pass
        hand_count = settings_tab.hand_count_spin.value()
        confidence = settings_tab.confidence_spin.value()
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=hand_count,
            min_detection_confidence=confidence
        )
        show_logo = settings_tab.logo_visible
        show_frame = settings_tab.frame_visible

        # Đồng bộ camera_tab
        self.camera_tab.logo_path = logo_path
        self.camera_tab.frame_path = frame_overlay_path
        self.camera_tab.show_logo = show_logo
        self.camera_tab.show_frame = show_frame
        self.camera_tab.update_frame()



    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand Gesture Capture")
        self.tabs = QTabWidget()
        self.camera_tab = CameraTab()
        self.tabs.addTab(self.camera_tab, "📷 Camera")
        self.tabs.addTab(GalleryTab(), "🖼️ Ảnh đã chụp")
        self.tabs.addTab(SettingsTab(), "⚙️ Cài đặt")
        self.setCentralWidget(self.tabs)
        self.settings_tab = self.tabs.widget(2)
        self.settings_tab.apply_button.clicked.connect(self.apply_settings)
        self.apply_settings()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())