import sys
import os
import json
import time
import socket
import threading
from datetime import datetime

import numpy as np
import cv2
import mediapipe as mp
import pyrealsense2 as rs

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QGroupBox, QGridLayout, QScrollArea, QProgressBar
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

# wk 로직 import
import scan_1216 as wk  

# =========================
# 저장 폴더
# =========================
SCAN_DIR = "scan_records"
os.makedirs(SCAN_DIR, exist_ok=True)

# =========================
# 카메라→로봇 변환행렬
# =========================
TRANSFORMATION_MATRIX = np.array([
    [0.01665455, 0.97822465, 0.02832482, 377.40874423],
    [0.99141691, -0.02245248, -0.00914220,   8.52529268],
    [-0.01228417, 0.00438147, -0.97705115, 390.87641972],
], dtype=float)
R_CAM2ROB = TRANSFORMATION_MATRIX[:, :3]
T_CAM2ROB = TRANSFORMATION_MATRIX[:, 3]

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 200

try:
    import serial
except ImportError:
    serial = None

ARDUINO_PORT = "COM5"
ARDUINO_BAUD = 115200
ARDUINO_TIMEOUT = 5.0

# =========================
# 로봇 좌표 변환 로직
# =========================
def cam_point_to_robot(point_cam_m: np.ndarray) -> np.ndarray:
    p_cam_mm = point_cam_m * 1000.0
    return R_CAM2ROB @ p_cam_mm + T_CAM2ROB

def cam_normal_to_robot(normal_cam: np.ndarray) -> np.ndarray:
    n = R_CAM2ROB @ normal_cam
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return n / norm

def frame_from_normal_z(normal_robot: np.ndarray, world_up: np.ndarray = np.array([0.0, 0.0, 1.0])) -> np.ndarray:
    z = normal_robot / (np.linalg.norm(normal_robot) + 1e-9)
    ref = world_up
    if abs(np.dot(z, ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    x = np.cross(ref, z)
    x = x / (np.linalg.norm(x) + 1e-9)
    y = np.cross(z, x)
    return np.column_stack((x, y, z))

def euler_zyz_deg_from_R(R: np.ndarray) -> np.ndarray:
    beta = np.arccos(np.clip(R[2, 2], -1.0, 1.0))
    if abs(np.sin(beta)) < 1e-9:
        alpha = np.arctan2(R[1, 0], R[0, 0])
        gamma = 0.0
    else:
        alpha = np.arctan2(R[1, 2], R[0, 2])
        gamma = np.arctan2(R[2, 1], -R[2, 0])
    return np.array([np.degrees(alpha), np.degrees(beta), np.degrees(gamma)], dtype=float)

def pose_from_record(record: dict) -> np.ndarray:
    p_cam_m = np.array([record["X_m"], record["Y_m"], record["Z_m"]], dtype=float)
    n_cam = np.array([record["nx"], record["ny"], record["nz"]], dtype=float)
    p_robot_mm = cam_point_to_robot(p_cam_m)
    n_robot = -cam_normal_to_robot(n_cam)
    R_tool = frame_from_normal_z(n_robot)
    A_deg, B_deg, C_deg = euler_zyz_deg_from_R(R_tool)
    x_mm, y_mm, z_mm = p_robot_mm
    return np.array([x_mm, y_mm, z_mm, A_deg, B_deg, C_deg], dtype=float)

def load_all_poses(jsonl_path: str):
    poses = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            poses.append(pose_from_record(json.loads(line)))
    return poses

# =========================
# 아두이노 컨트롤러
# =========================
class ArduinoController:
    def __init__(self):
        self.ser = None
        self.lock = threading.Lock()
    def open(self) -> bool:
        if serial is None: return False
        if self.ser and self.ser.is_open: return True
        try:
            self.ser = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=1.0)
            time.sleep(2.0)
            self.ser.reset_input_buffer()
            return True
        except:
            self.ser = None
            return False
    def run_once(self) -> bool:
        with self.lock:
            if not self.open(): return False
            try:
                self.ser.write(b"1")
                self.ser.flush()
            except: return False
            t0 = time.time()
            while True:
                try: line = self.ser.readline().decode(errors="ignore").strip()
                except: line = ""
                if "DONE" in line: return True
                if time.time() - t0 > ARDUINO_TIMEOUT: return False

# =========================
# 서버 스레드
# =========================
class RobotServerThread(QThread):
    sig_status = pyqtSignal(str)
    sig_done_idx = pyqtSignal(int)
    def __init__(self, jsonl_path, arduino, host=SERVER_HOST, port=SERVER_PORT, parent=None):
        super().__init__(parent)
        self.jsonl_path = jsonl_path
        self.arduino = arduino
        self.host, self.port = host, port
        self.running = True
        self._sock = None
    def stop(self):
        self.running = False
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(("127.0.0.1", self.port))
            s.close()
        except: pass
        try: self._sock.close()
        except: pass
    def run(self):
        try: poses = load_all_poses(self.jsonl_path)
        except Exception as e:
            self.sig_status.emit(f"[SERVER] 로드 실패: {e}")
            return
        if not poses:
            self.sig_status.emit("[SERVER] 포인트 0개")
            return
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._sock.bind((self.host, self.port))
            self._sock.listen(1)
        except Exception as e:
            self.sig_status.emit(f"[SERVER] Bind 실패: {e}")
            return
        self.sig_status.emit(f"[SERVER] READY (pts={len(poses)})")
        while self.running:
            try: conn, addr = self._sock.accept()
            except: break
            if not self.running: break
            self.sig_status.emit(f"[SERVER] CONNECT {addr}")
            idx = 0
            try:
                while self.running:
                    data = conn.recv(1024)
                    if not data: break
                    cmd = data.decode(errors="ignore").strip().lower()
                    if cmd == "shot":
                        if idx >= len(poses):
                            conn.sendall("END\r\n".encode())
                            self.sig_status.emit("[SERVER] END")
                            break
                        msg = ",".join(f"{v:.6f}" for v in poses[idx]) + "\r\n"
                        conn.sendall(msg.encode())
                        self.sig_status.emit(f"[SERVER] SHOT idx={idx}")
                        idx += 1
                    elif cmd == "reached":
                        done = idx - 1
                        self.sig_status.emit(f"[SERVER] REACHED -> INJECT")
                        if self.arduino.run_once():
                            conn.sendall("DONE\r\n".encode())
                            if done >= 0: self.sig_done_idx.emit(done)
                            self.sig_status.emit(f"[SERVER] DONE idx={done}")
                        else:
                            conn.sendall("ERROR\r\n".encode())
                            self.sig_status.emit(f"[SERVER] ARDUINO FAIL")
                    elif cmd in ("quit", "exit"):
                        conn.sendall("BYE\r\n".encode())
                        break
            except: pass
            finally: conn.close()
        try: self._sock.close()
        except: pass

# =========================
# 메인 윈도우
# =========================
class MainWindow(QMainWindow):
    def __init__(self, patients):
        super().__init__()
        self.setWindowTitle("SkinDepth AI Injector")
        self.resize(1300, 750)
        self.patients = patients
        self.current_patient_idx = 0
        self.frozen_background = None
        self.frozen_pixmap = None

        # RealSense
        self.rs_pipeline = None
        self.rs_align = None
        self.rs_depth_scale = 0.001
        self.last_intr = None

        # Scanning Logic Variables
        self.is_scanning = False
        self.scan_frame_count = 0
        self.scan_max_frames = 90  # wk와 동일하게 90프레임(3초) 누적
        self.scan_samples = []     # 랜드마크별 좌표 리스트: [ [pt, pt..], ... ]
        
        # Display
        self.freeze_view = False
        self.frozen_pixmap = None
        self.injection_points = []
        self.done_indices = set()

        # Paths
        self.latest_jsonl_path = None
        
        # Hardware
        self.arduino = ArduinoController()
        self.server_thread = None

        # FaceMesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, # 동영상 모드로 변경 (안정성 증가)
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # UI Layout
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # Left
        left = QVBoxLayout()
        self.lbl_status = QLabel("대기 중")
        left.addWidget(self.lbl_status)

        self.btn_cam_start = QPushButton("카메라 시작")
        self.btn_cam_start.clicked.connect(self.on_cam_start)
        left.addWidget(self.btn_cam_start)

        self.btn_cam_stop = QPushButton("카메라 정지")
        self.btn_cam_stop.clicked.connect(self.on_cam_stop)
        left.addWidget(self.btn_cam_stop)

        # 수정됨: 스캔 시작 버튼
        self.btn_scan = QPushButton(f"스캔 시작 (3초간 가만히 유지)")
        self.btn_scan.clicked.connect(self.start_scan_mode)
        left.addWidget(self.btn_scan)

        self.btn_server = QPushButton("주사 시작")
        self.btn_server.clicked.connect(self.on_start_server)
        left.addWidget(self.btn_server)

        self.btn_server_stop = QPushButton("서버 정지")
        self.btn_server_stop.clicked.connect(self.on_stop_server)
        left.addWidget(self.btn_server_stop)

        info = QGroupBox("진행 상황")
        g = QGridLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("%p%")  # 퍼센트 표시

        g.addWidget(QLabel("진행률"), 2, 0)
        g.addWidget(self.progress, 2, 1)
        info.setLayout(g)
        g.addWidget(QLabel("전체 포인트: "), 0, 0)
        self.lbl_total = QLabel("0")
        g.addWidget(self.lbl_total, 0, 1)
        g.addWidget(QLabel("완료 포인트: "), 1, 0)
        self.lbl_done = QLabel("0")
        g.addWidget(self.lbl_done, 1, 1)
        left.addWidget(info)
        left.addStretch()
        root.addLayout(left, 2)

        # Center
        center = QVBoxLayout()
        self.view = QLabel("카메라를 켜고 환자를 선택하세요.")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumHeight(520)
        center.addWidget(self.view)
        root.addLayout(center, 6)

        # Right
        right = QVBoxLayout()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        v = QVBoxLayout(content)

        self.patient_buttons = []
        for idx, p in enumerate(self.patients):
            b = QPushButton(f"{p['name']} ({p['id']})")
            b.setCheckable(True)
            b.clicked.connect(lambda _, i=idx: self.select_patient(i))
            v.addWidget(b)
            self.patient_buttons.append(b)

        v.addStretch()
        scroll.setWidget(content)
        right.addWidget(scroll, 3)
        
        # --- ✅ 환자 상세 정보 박스 추가 ---
        detail = QGroupBox("환자 상세 정보")
        dv = QVBoxLayout(detail)
        
        self.lbl_p_name = QLabel("-")     # 이름/ID
        self.lbl_p_meta = QLabel("")      # 나이/성별/최근일
        self.lbl_p_warning = QLabel("")   # 주의사항
        self.lbl_p_note = QLabel("")      # 메모
        self.lbl_p_history = QLabel("")   # 시술 이력

        # 줄바꿈/자동 높이
        for lb in (self.lbl_p_warning, self.lbl_p_note, self.lbl_p_history):
            lb.setWordWrap(True)

        dv.addWidget(self.lbl_p_name)
        dv.addWidget(self.lbl_p_meta)
        dv.addSpacing(6)
        dv.addWidget(self.lbl_p_warning)
        dv.addWidget(self.lbl_p_note)
        dv.addWidget(self.lbl_p_history)

        right.addWidget(detail, 7)

        root.addLayout(right, 2)

        if self.patients: self.select_patient(0)

        # =========================
        # StyleSheet (QSS)
        # =========================
        self.setStyleSheet("""
        * {
            font-family: "Consolas";
            font-size: 16px;
        }
                           
        QMainWindow { background: #0B1220; }

        QLabel { color: #E5E7EB; }
        QLabel#Title { font-size: 14px; }

        QGroupBox {
            color: #9CA3AF;
            border: 1px solid #1F2937;
            border-radius: 12px;
            margin-top: 10px;
            padding: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 6px;
        }

        QScrollArea { border: none; background: transparent; }
        QScrollArea > QWidget > QWidget { background: transparent; }

        QPushButton {
            background: #111827;
            border: 1px solid #1F2937;
            border-radius: 10px;
            padding: 10px 12px;
            color: #E5E7EB;
            font-size: 12px;
        }
        QPushButton:hover { background: #172554; }
        QPushButton:pressed { background: #0B1220; }

        QPushButton:checked {
            border: 1px solid #2563EB;
            background: #0F172A;
        }

        QPushButton#Primary {
            background: #2563EB;
            border: none;
            font-weight: 700;
        }
        QPushButton#Primary:hover { background: #1D4ED8; }

        QPushButton#Danger {
            background: #B91C1C;
            border: none;
            font-weight: 700;
        }
        QPushButton#Danger:hover { background: #991B1B; }
        
        QProgressBar {
            border: 1px solid #1F2937;
            border-radius: 8px;
            background: #111827;
            text-align: center;
            color: #E5E7EB;
            height: 18px;
        }
        QProgressBar::chunk {
            background-color: #2563EB;
            border-radius: 8px;
        }
        """)

    # ---------------- Camera ----------------
    def on_cam_start(self):
        if self.rs_pipeline is None:
            try:
                self.rs_pipeline = rs.pipeline()
                cfg = rs.config()
                cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                prof = self.rs_pipeline.start(cfg)
                self.rs_align = rs.align(rs.stream.color)
                self.rs_depth_scale = prof.get_device().first_depth_sensor().get_depth_scale()
            except Exception as e:
                self.lbl_status.setText(f"카메라 에러: {e}")
                return
        self.freeze_view = False
        self.timer.start(30)
        self.lbl_status.setText("카메라 실행 중")

    def on_cam_stop(self):
        self.timer.stop()
        if self.rs_pipeline:
            self.rs_pipeline.stop()
            self.rs_pipeline = None
        self.lbl_status.setText("카메라 정지")

    # ---------------- Scan Loop (Accumulate) ----------------
    def start_scan_mode(self):
        if self.rs_pipeline is None:
            self.frozen_background = None
            self.lbl_status.setText("카메라를 먼저 켜주세요.")
            return
        
        # 초기화
        self.is_scanning = True
        self.scan_frame_count = 0
        NUM_LM = wk.NUM_LM if hasattr(wk, "NUM_LM") else 478
        self.scan_samples = [[] for _ in range(NUM_LM)] # 478개 리스트 초기화
        
        self.freeze_view = False
        self.lbl_status.setText("스캔 중... 움직이지 마세요.")

    def update_frame(self):
        # 1. 스캔 완료되어 프리즈 상태면 그거 보여주고 리턴
        if self.freeze_view and self.frozen_background is not None:
            disp = self.frozen_background.copy()
            h, w, _ = disp.shape

            # 주사점(노란/빨간)만 매번 다시 그림 -> done_indices 변화가 즉시 반영됨
            for i, (xn, yn) in enumerate(self.injection_points):
                cx, cy = int(xn * w), int(yn * h)
                color = (0, 0, 255) if i in self.done_indices else (0, 255, 255)
                cv2.circle(disp, (cx, cy), 5, color, -1)

            rgb_disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb_disp.data, w, h, w * 3, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(
                self.view.width(), self.view.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.view.setPixmap(pix)
            return

        # 2. 프레임 획득
        frames = self.rs_pipeline.poll_for_frames()
        if not frames: return
        frames = self.rs_align.process(frames)
        c_frame = frames.get_color_frame()
        d_frame = frames.get_depth_frame()
        if not c_frame or not d_frame: return

        color_img = np.asanyarray(c_frame.get_data())
        depth_img = np.asanyarray(d_frame.get_data())
        h, w, _ = color_img.shape
        intr = c_frame.profile.as_video_stream_profile().intrinsics
        self.last_intr = intr  # 나중에 투영할 때 사용

        # 3. FaceMesh 수행
        rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        
        disp = color_img.copy()

        # 4. 스캔 모드 로직 (데이터 누적)
        if self.is_scanning and res.multi_face_landmarks:
            self.scan_frame_count += 1
            lmks = res.multi_face_landmarks[0].landmark
            
            # 모든 랜드마크에 대해 깊이 추출 및 저장
            for idx, lm in enumerate(lmks):
                if idx >= len(self.scan_samples): break
                
                px, py = int(lm.x * w), int(lm.y * h)
                if 0 <= px < w and 0 <= py < h:
                    d_raw = depth_img[py, px]
                    if d_raw > 0:
                        d_m = d_raw * self.rs_depth_scale
                        # wk 설정 범위 체크
                        if wk.MIN_DEPTH_M <= d_m <= wk.MAX_DEPTH_M:
                            # 3D 변환 후 저장
                            X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [px, py], d_m)
                            self.scan_samples[idx].append([X, Y, Z])

            cv2.putText(disp, f"SCANNING {self.scan_frame_count}/{self.scan_max_frames}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            # 목표 프레임 도달 시 스캔 종료 및 처리
            if self.scan_frame_count >= self.scan_max_frames:
                self.is_scanning = False
                self.process_scan_data(color_img) # 처리 함수 호출

        # 5. UI 그리기 (랜드마크, 주사점 등)
        if res.multi_face_landmarks:
            # 랜드마크 점 찍기
            for lm in res.multi_face_landmarks[0].landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(disp, (cx, cy), 1, (0, 255, 0), -1)

        # 생성된 주사 경로가 있으면 표시
        for i, (xn, yn) in enumerate(self.injection_points):
            cx, cy = int(xn * w), int(yn * h)
            color = (0, 0, 255) if i in self.done_indices else (0, 255, 255)
            cv2.circle(disp, (cx, cy), 5, color, -1)

        # 화면 업데이트
        rgb_disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb_disp.data, w, h, w*3, QImage.Format_RGB888)
        self.frozen_pixmap = QPixmap.fromImage(qimg).scaled(self.view.width(), self.view.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.view.setPixmap(self.frozen_pixmap)
        
        # 스캔 완료 직후라면 화면을 이 상태로 고정
        if self.latest_jsonl_path and not self.is_scanning and self.scan_frame_count >= self.scan_max_frames:
            if not self.freeze_view:
                self.frozen_background = disp.copy()  # <- 배경 저장(이후 점만 갱신)
            self.freeze_view = True


    # ---------------- Process Data (wk logic) ----------------
    def process_scan_data(self, last_color_img):
        """누적된 데이터를 wk 로직으로 정제하고 저장"""
        self.lbl_status.setText("데이터 처리 중...")
        QApplication.processEvents()

        NUM_LM = len(self.scan_samples)
        fused_pts = np.zeros((NUM_LM, 3), dtype=np.float64)
        fused_mask = np.zeros((NUM_LM,), dtype=bool)

        # 1. Median Filtering (wk와 동일)
        min_samples = wk.MIN_SAMPLES_PER_POINT
        for idx, samples in enumerate(self.scan_samples):
            if len(samples) < min_samples:
                continue
            arr = np.array(samples)
            z_vals = arr[:, 2]
            z_med = np.median(z_vals)
            # Z값 기준 outlier 제거 (2cm 이내)
            good_indices = np.abs(z_vals - z_med) < 0.02
            arr_good = arr[good_indices]
            
            if arr_good.shape[0] > 0:
                fused_pts[idx] = arr_good.mean(axis=0)
                fused_mask[idx] = True

        if fused_mask.sum() < 50:
            self.lbl_status.setText("실패: 유효 포인트가 너무 적음. 다시 스캔하세요.")
            self.freeze_view = False
            return

        # 2. wk 파이프라인 호출 (안전마스크 -> 법선 -> 리샘플링)
        # wk_1cm_margin_3.py 의 로직을 그대로 사용
        safe_mask = wk.build_safety_mask_uv(fused_pts, fused_mask)
        normals = wk.estimate_normals(fused_pts, safe_mask, k=20)
        
        # 3. 1cm 간격 리샘플링
        resampled_pts, resampled_normals = wk.resample_zigzag_points(
            fused_pts, fused_mask, safe_mask, normals,
            row_step_mm=5.0, spacing_mm=10.0
        )

        if len(resampled_pts) == 0:
            self.lbl_status.setText("실패: 안전 영역 내 경로 생성 불가.")
            self.freeze_view = False
            return

        # 4. 파일 저장
        patient = self.patients[self.current_patient_idx]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 이미지 저장
        img_name = f"{patient['id']}_{ts}.png"
        img_path = os.path.join(SCAN_DIR, img_name)
        cv2.imwrite(img_path, last_color_img)
        
        # JSONL 저장 (wk 함수 활용)
        jsonl_name = f"{patient['id']}_{ts}_path.jsonl"
        jsonl_path = os.path.join(SCAN_DIR, jsonl_name)
        
        wk.export_resampled_path_to_jsonl(
            fused_pts, fused_mask, out_path=jsonl_path,
            row_step_mm=5.0, spacing_mm=10.0
        )
        self.latest_jsonl_path = jsonl_path

        # 5. UI용 포인트 생성 (Project back to 2D)
        h, w, _ = last_color_img.shape
        self.injection_points = []
        for p in resampled_pts:
            u, v = rs.rs2_project_point_to_pixel(self.last_intr, [float(p[0]), float(p[1]), float(p[2])])
            self.injection_points.append((u / w, v / h))

        self.done_indices.clear()
        self.lbl_total.setText(str(len(self.injection_points)))
        self.lbl_done.setText("0")
        self.update_progress()
        self.lbl_status.setText(f"스캔 완료! {len(resampled_pts)} 포인트 저장됨.")
        print(f"[UI] Scan processed. Saved to {jsonl_path}")


    # ---------------- Server Control ----------------
    def on_start_server(self):
        if not self.latest_jsonl_path:
            self.lbl_status.setText("먼저 스캔을 완료하세요.")
            return
        if self.server_thread and self.server_thread.isRunning(): return
        
        self.server_thread = RobotServerThread(self.latest_jsonl_path, self.arduino)
        self.server_thread.sig_status.connect(self.set_server_status)
        self.server_thread.sig_done_idx.connect(self.mark_done)
        self.server_thread.start()

    def on_stop_server(self):
        if self.server_thread:
            self.server_thread.stop()
            self.server_thread.wait()
        self.lbl_status.setText("서버 정지")

    def set_server_status(self, msg):
        self.lbl_status.setText(msg)

    def mark_done(self, idx):
        self.done_indices.add(idx)
        self.lbl_done.setText(str(len(self.done_indices)))
        self.update_progress()

    def select_patient(self, idx):
        self.current_patient_idx = idx

        for i, b in enumerate(self.patient_buttons):
            b.setChecked(i == idx)
        
        p = self.patients[idx]

        if hasattr(self, "lbl_p_name"):
            self.lbl_p_name.setText(f"{p.get('name', '-') } ({p.get('id', '-')})")
            self.lbl_p_meta.setText(
                f"{p.get('age', '-')}세 · {p.get('gender', '-')} · 최근 시술일: {p.get('last_date', '-')}"
            )
            self.lbl_p_warning.setText(f"주의사항: {p.get('warning', '-')}")
            self.lbl_p_note.setText(f"메모: {p.get('note', '-')}")
            self.lbl_p_history.setText(f"이전 시술: {p.get('history', '-')}")

        self.lbl_status.setText(f"환자: {self.patients[idx]['name']}")
        self.injection_points = []
        self.done_indices.clear()
        self.lbl_total.setText("0")
        self.lbl_done.setText("0")
        self.progress.setValue(0)
        self.latest_jsonl_path = None
        self.freeze_view = False

    def closeEvent(self, e):
        self.on_stop_server()
        self.on_cam_stop()
        super().closeEvent(e)

    def update_progress(self):
        try:
            total = int(self.lbl_total.text())
        except:
            total = 0
        done = len(self.done_indices)

        if total <= 0:
            self.progress.setValue(0)
            return

        pct = int(round(done * 100 / total))
        self.progress.setValue(pct)


if __name__ == "__main__":
    pts = [
                {
            "name": "이수환",
            "id": "P001",
            "age": 32,
            "gender": "여",
            "warning": "눈 주변 멍 잘 듦, 좌측 볼 압통",
            "note": "알레르기 없음",
            "history": "2024-10-12 스킨부스터 2cc",
            "last_date": "2024-12-11",
        },
        {
            "name": "정희윤",
            "id": "P002",
            "age": 28,
            "gender": "여",
            "warning": "리도카인 민감, 마취 시간 충분히 필요",
            "note": "갑상선 약 복용 중 (용량 변경 시 확인)",
            "history": "2024-09-01 턱선 필러",
            "last_date": "2024-12-09",
        },
        {
            "name": "황원건",
            "id": "P003",
            "age": 35,
            "gender": "여",
            "warning": "코 주변 통증 민감, 출혈 많았던 이력",
            "note": "고혈압 약 복용, 시술 전 혈압 체크 필수",
            "history": "2024-08-20 이마 보톡스",
            "last_date": "2024-12-01",
        },
    ]
    app = QApplication(sys.argv)
    w = MainWindow(pts)
    w.show()
    sys.exit(app.exec())