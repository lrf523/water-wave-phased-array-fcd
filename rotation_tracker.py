#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import csv
import time
import math
import threading
import queue
from collections import deque
import numpy as np
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from scipy.optimize import least_squares
import platform

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

DEG = "\N{DEGREE SIGN}"


# =====================================================================
# 小球检测：尺度匹配的差分高斯(DoG)斑点检测
#
# 为什么不再用 goodFeaturesToTrack + 光流：
#   背景是 FCD 的棋盘格光栅，整幅画面处处高对比角点，光流会牢牢咬住背景；
#   而棋盘格本身不旋转，于是既跟不住小球、也拟合不出旋转中心。
#
# DoG 的道理：棋盘格空间周期只有几个像素，小球有几十像素。用与球同尺度的
# 高斯低通一模糊，棋盘格被平成均值直接消失，小球还在；再减掉更大尺度的
# 背景，小球就是响应的极值点。亚像素位置用响应加权质心。
# =====================================================================
def detect_ball(gray, guess, ball_radius, search_radius, dark_ball=True):
    """在 guess 附近找小球，返回 (x, y, 信噪比)；找不到返回 None。
    guess 为 None 时全画面搜索（跟丢后的兜底）。"""
    img = np.asarray(gray, dtype=np.float32)
    H, W = img.shape[:2]
    r = max(2.0, float(ball_radius))

    if guess is None:
        x0, y0, x1, y1 = 0, 0, W, H
    else:
        gx, gy = guess
        pad = max(int(search_radius), int(3 * r))
        x0 = max(0, int(gx) - pad); x1 = min(W, int(gx) + pad + 1)
        y0 = max(0, int(gy) - pad); y1 = min(H, int(gy) + pad + 1)
    win = img[y0:y1, x0:x1]
    if win.size == 0 or min(win.shape) < 5:
        return None

    small = cv2.GaussianBlur(win, (0, 0), 0.6 * r)      # 抹掉棋盘格，留住小球
    large = cv2.GaussianBlur(win, (0, 0), 3.0 * r)      # 局部背景
    resp = (large - small) if dark_ball else (small - large)

    m = int(np.ceil(1.5 * r))                            # 收掉模糊造成的边缘假响应
    if resp.shape[0] > 2 * m + 3 and resp.shape[1] > 2 * m + 3:
        core = resp[m:-m, m:-m]; oy, ox = m, m
    else:
        core = resp; oy = ox = 0
    if core.size == 0:
        return None

    iy, ix = np.unravel_index(int(np.argmax(core)), core.shape)
    peak = float(core[iy, ix])
    iy += oy; ix += ox

    k = max(2, int(round(r)))                            # 亚像素：响应加权质心
    ys0, ys1 = max(0, iy - k), min(resp.shape[0], iy + k + 1)
    xs0, xs1 = max(0, ix - k), min(resp.shape[1], ix + k + 1)
    patch = resp[ys0:ys1, xs0:xs1]
    wgt = np.clip(patch - patch.min(), 0, None)
    tot = float(wgt.sum())
    if tot <= 1e-12:
        return None
    yy, xx = np.mgrid[ys0:ys1, xs0:xs1]
    cx = float((wgt * xx).sum() / tot) + x0
    cy = float((wgt * yy).sum() / tot) + y0
    return cx, cy, peak / (float(np.std(resp)) + 1e-9)


def _orbit_gate(x, y, center_xy, fit_radius, tol_lo=0.5, tol_hi=1.6):
    """已知圆心/半径时的物理先验校验：候选点到圆心的距离必须落在合理的
    半径带内，才认为它是球而不是背景上的固定伪特征（棋盘格褶皱、水槽
    边缘卡口这些跟球完全不同心、也不在轨道半径上）。center_xy 为 None
    时（还没拟合出圆心）永远放行，不做限制。"""
    if center_xy is None or fit_radius is None or fit_radius <= 1.0:
        return True
    d = math.hypot(x - center_xy[0], y - center_xy[1])
    return tol_lo * fit_radius <= d <= tol_hi * fit_radius


def concentric_residuals(center, trajectories):
    cx, cy = center
    res = []
    for traj in trajectories:
        dist = np.sqrt((traj[:, 0] - cx) ** 2 + (traj[:, 1] - cy) ** 2)
        res.append(dist - np.mean(dist))
    return np.concatenate(res)


# =====================================================================
# 离线视频分析：拿一整段 mp4，把小球的轨迹、旋转中心、角速度一次性算完
#
# 跟实时相机那一路的核心区别：
#   实时那一路要“边采边看”，角速度是在线滑动窗口回归，圆心也是边跑边收敛；
#   离线这一路视频已经完整躺在硬盘上了，没必要假装实时——先把全部帧的小球
#   位置都找出来，再用全部轨迹一次性做鲁棒圆拟合，最后对展开后的角度做整体
#   线性回归，这样给出的角速度和圆心都是全局最优解，比在线版本稳得多，也
#   适合直接当实验数据引用（带 R^2、圆拟合残差这些可以放进论文/报告的量）。
# =====================================================================
def track_ball_in_video(path, roi, ball_radius, search_radius, dark_ball=True,
                         max_jump_factor=5.0, progress_cb=None, cancel_flag=None):
    """逐帧离线追踪整段视频里的小球。

    guess 用上一帧位置 + 简单匀速外推；如果探测结果偏离预测太远（大概率是
    棋盘格纹理或涡旋核花纹造成的误检），丢弃这次探测，改用外推位置占位，
    并标记为不可信（valid=False），不会污染后面的圆拟合和角速度回归。

    返回 dict: fps, frame_idx, t, x, y, score, valid（均为 np.ndarray）。
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pos = (roi[0] + roi[2] / 2.0, roi[1] + roi[3] / 2.0)
    vel = (0.0, 0.0)
    lost_streak = 0

    frame_idx_l, t_l, x_l, y_l, score_l, valid_l = [], [], [], [], [], []
    idx = 0
    max_jump = max_jump_factor * max(ball_radius, search_radius * 0.5)

    while True:
        if cancel_flag is not None and cancel_flag():
            break
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        guess = (pos[0] + vel[0], pos[1] + vel[1])
        hit = detect_ball(gray, guess, ball_radius, search_radius, dark_ball)
        if hit is not None:
            hx, hy, score = hit
            if math.hypot(hx - guess[0], hy - guess[1]) > max_jump:
                hit = None      # 跳变太大，多半是棋盘格/涡旋核花纹的误检，丢掉

        valid = False
        if hit is not None:
            hx, hy, score = hit
            vel = (hx - pos[0], hy - pos[1])
            pos = (hx, hy)
            valid = True
            lost_streak = 0
        else:
            lost_streak += 1
            score = 0.0
            if lost_streak > 10:
                # 连续跟丢：绝不做"全画面盲搜"。实测过，这个棋盘格+涡旋核花纹的
                # 背景里，全画面 DoG 响应最强的地方是水槽内壁上固定不动的卡口
                # 花纹（跟球完全不同心也不移动），一盲搜就会死锁在那上面，表现
                # 出来就是"怎么追都在追背景"。改成以最后已知位置为中心，逐步
                # 放大局部搜索半径找回，范围有界，够不到那些远处的背景伪特征。
                widen = 1 + min(4, (lost_streak - 10) // 8)
                hit_wide = detect_ball(gray, pos, ball_radius, search_radius * widen, dark_ball)
                if hit_wide is not None:
                    pos = (hit_wide[0], hit_wide[1])
                    vel = (0.0, 0.0)
                    valid = True
                    score = hit_wide[2]
                    lost_streak = 0
                else:
                    pos = guess
            else:
                pos = guess                    # 短暂跟丢：用匀速外推占位，不当真实测量点

        frame_idx_l.append(idx)
        t_l.append(idx / fps)
        x_l.append(pos[0])
        y_l.append(pos[1])
        score_l.append(score)
        valid_l.append(valid)

        idx += 1
        if progress_cb is not None:
            progress_cb(idx, n_frames)

    cap.release()
    return {
        "fps": fps,
        "frame_idx": np.array(frame_idx_l, dtype=int),
        "t": np.array(t_l, dtype=float),
        "x": np.array(x_l, dtype=float),
        "y": np.array(y_l, dtype=float),
        "score": np.array(score_l, dtype=float),
        "valid": np.array(valid_l, dtype=bool),
    }


def fit_circle_robust(traj_xy, n_iter=4, sigma_clip=2.5):
    """对一条轨迹做带异常点剔除的最小二乘圆拟合（迭代 sigma-clipping）。

    单条轨迹时 concentric_residuals 最小化的就是半径方差，即标准几何圆拟合，
    不依赖任何背景特征点。

    返回 (cx, cy, r_mean, r_std, inlier_mask)。
    """
    pts = np.asarray(traj_xy, dtype=np.float64)
    if len(pts) < 8:
        raise ValueError("有效轨迹点太少，无法拟合圆（至少需要 8 个点）")

    inlier = np.ones(len(pts), dtype=bool)
    cx, cy = pts.mean(axis=0)
    for _ in range(n_iter):
        sub = pts[inlier]
        if len(sub) < 8:
            break
        res = least_squares(concentric_residuals, x0=(cx, cy), args=([sub],))
        cx, cy = float(res.x[0]), float(res.x[1])
        d_all = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        r_mean, r_std = d_all[inlier].mean(), d_all[inlier].std()
        if r_std < 1e-6:
            break
        new_inlier = np.abs(d_all - r_mean) < max(sigma_clip * r_std, 1.5)
        if np.array_equal(new_inlier, inlier):
            break
        inlier = new_inlier

    d_all = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    return cx, cy, float(d_all[inlier].mean()), float(d_all[inlier].std()), inlier


def compute_angular_velocity(t, x, y, cx, cy, win_s=1.5):
    """给定轨迹和已知圆心，算角速度。

    全局角速度：把展开后的角度 theta(t) 做整体线性回归，斜率就是角速度，
    截距无所谓；用残差估计斜率的标准误，R^2 衡量“转速是否均匀”。
    这比逐帧差分稳得多——逐帧差分对亚像素噪声极敏感，整体回归把噪声平均掉了。

    另外算一条滑动窗口的瞬时角速度曲线，用来目视检查转速沿途是否均匀、
    有没有卡顿或加减速——physically 这正是判断"是否匀速圆周运动"假设是否
    成立的诊断图。
    """
    t = np.asarray(t, dtype=np.float64)
    theta = np.unwrap(np.arctan2(np.asarray(y) - cy, np.asarray(x) - cx))

    A = np.vstack([t, np.ones_like(t)]).T
    (omega_mean, theta0), *_ = np.linalg.lstsq(A, theta, rcond=None)
    theta_fit = A @ np.array([omega_mean, theta0])
    ss_res = float(np.sum((theta - theta_fit) ** 2))
    ss_tot = float(np.sum((theta - theta.mean()) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    dof = max(1, len(t) - 2)
    resid_std = math.sqrt(ss_res / dof)
    t_var = float(np.sum((t - t.mean()) ** 2)) + 1e-12
    omega_stderr = resid_std / math.sqrt(t_var)

    win_t, win_omega = [], []
    i0 = 0
    for i1 in range(len(t)):
        while t[i1] - t[i0] > win_s:
            i0 += 1
        if i1 - i0 >= 4:
            tt = t[i0:i1 + 1] - t[i0]
            th = theta[i0:i1 + 1]
            s, _ = np.polyfit(tt, th, 1)
            win_t.append(t[i1])
            win_omega.append(s)

    return {
        "theta": theta,
        "omega_mean": float(omega_mean),
        "omega_stderr": float(omega_stderr),
        "theta0": float(theta0),
        "r2": float(r2),
        "win_t": np.array(win_t),
        "win_omega": np.array(win_omega),
    }


def export_annotated_video(path, out_path, data, cx, cy, r_mean, ball_radius, progress_cb=None):
    """把追踪结果叠加回原视频（球位置圈出、拟合圆心、轨迹尾迹），方便肉眼核验
    追踪对不对，而不是只信一堆数字。"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or data["fps"]
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    trail = deque(maxlen=150)
    n = len(data["x"])
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or idx >= n:
            break
        x, y, valid = data["x"][idx], data["y"][idx], data["valid"][idx]
        trail.append((int(x), int(y)))

        color = (0, 255, 0) if valid else (0, 165, 255)
        cv2.circle(frame, (int(x), int(y)), max(3, int(ball_radius)), color, 2)
        cv2.drawMarker(frame, (int(x), int(y)), color, cv2.MARKER_CROSS, 16, 1)
        if len(trail) > 1:
            cv2.polylines(frame, [np.array(trail, dtype=np.int32)], False, (0, 200, 255), 1)
        cv2.drawMarker(frame, (int(cx), int(cy)), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.circle(frame, (int(cx), int(cy)), int(r_mean), (255, 120, 0), 1)

        vw.write(frame)
        idx += 1
        if progress_cb is not None:
            progress_cb(idx, n)

    cap.release()
    vw.release()


class RotationTrackerUI:
    def __init__(self, camera=None):
        self.camera = camera
        self.monitoring = False
        self._stop_worker = False
        self.selected_area = None

        self._track_lock = threading.RLock()
        self._fit_lock = threading.Lock()

        # 小球追踪状态（取代原来的特征点云 + 光流）
        self.ball_pos = None                 # 当前小球位置 (x, y)，全画面坐标
        self.ball_traj = deque(maxlen=3000)  # (t, x, y)
        self.ball_radius = 15.0
        self.fit_radius = None               # 圆拟合出的回转半径
        self.fit_rms = None

        self._ball_vel = (0.0, 0.0)

        self.center_xy = None
        self.angle_measuring = False
        self.angle_center_xy = None
        self.angle_point_ids = []
        self.angle_point_state = {}

        self._accum_angle = 0.0
        self._speed_display = 0.0
        self._angle_hist = deque(maxlen=6000)

        self.frame_q = queue.Queue(maxsize=2)
        self.state_q = queue.Queue(maxsize=5)

        # 可调参数
        self.radius_var = tk.DoubleVar(value=15.0)
        self.search_var = tk.DoubleVar(value=60.0)
        self.dark_var = tk.BooleanVar(value=True)
        self.speed_win_var = tk.DoubleVar(value=1.0)

        self._build_gui()
        self.root.after(20, self._ui_pump)

    def _build_gui(self):
        self.root = tk.Toplevel()
        self.root.title("相机实时角速度与旋转中心追踪")
        self.root.geometry("450x650")
        self.root.attributes("-topmost", True)

        self.side = ttk.Frame(self.root, padding=10)
        self.side.pack(fill=tk.BOTH, expand=True)

        self.status_lbl = ttk.Label(self.side, text="状态: 请先框选小球", foreground="blue")
        self.status_lbl.pack(anchor="w", pady=5)

        self.position_lbl = ttk.Label(self.side, text="旋转中心: N/A")
        self.position_lbl.pack(anchor="w", pady=5)

        self.radius_lbl = ttk.Label(self.side, text="回转半径: N/A")
        self.radius_lbl.pack(anchor="w", pady=2)

        self.angle_lbl = ttk.Label(self.side, text=f"当前累计角度: 0.0000{DEG}", font=("", 12, "bold"))
        self.angle_lbl.pack(anchor="w", pady=5)

        self.speed_lbl = ttk.Label(self.side, text=f"实时角速度: 0.00000{DEG}/s", font=("", 12, "bold"), foreground="red")
        self.speed_lbl.pack(anchor="w", pady=5)

        cfg = ttk.LabelFrame(self.side, text="小球与搜索参数", padding=6)
        cfg.pack(fill=tk.X, pady=6)

        def _row(r, text, var, tip):
            ttk.Label(cfg, text=text).grid(row=r, column=0, sticky="e", padx=(0, 4), pady=2)
            ttk.Entry(cfg, textvariable=var, width=7, justify="center").grid(row=r, column=1, sticky="w")
            ttk.Label(cfg, text=tip, foreground="#666").grid(row=r, column=2, sticky="w", padx=6)

        _row(0, "小球半径(px):", self.radius_var, "框选时按框大小自动填")
        _row(1, "搜索半径(px):", self.search_var, "两帧间小球最多跑多远")
        _row(2, "测速窗口(s):", self.speed_win_var, "角速度用最近这么久回归")
        ttk.Checkbutton(cfg, text="小球比背景暗", variable=self.dark_var).grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(4, 0))

        ttk.Button(self.side, text="1. 从相机画面框选小球", command=self._start_area_selection).pack(fill=tk.X, pady=10)

        self.toggle_btn = ttk.Button(self.side, text="2. 开始追踪小球", command=self._toggle_monitoring)
        self.toggle_btn.pack(fill=tk.X, pady=5)

        self.angle_measure_btn = ttk.Button(self.side, text="3. 锁定中心并开始测角速度", command=self._toggle_angle_measurement)
        self.angle_measure_btn.pack(fill=tk.X, pady=5)

        ttk.Button(self.side, text="重置角度", command=self._reset_angle).pack(fill=tk.X, pady=5)

        self.preview_canvas = tk.Canvas(self.side, bg="black", height=300)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True, pady=10)
        self._preview_img_item = None

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _start_area_selection(self):
        if not self.camera or not self.camera.is_opened:
            messagebox.showerror("错误", "相机未连接！请先在主界面连接相机。")
            return

        import mvsdk
        try:
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self.camera.hCamera, 1000)
            mvsdk.CameraImageProcess(self.camera.hCamera, pRawData, self.camera.pFrameBuffer, FrameHead)
            mvsdk.CameraReleaseImageBuffer(self.camera.hCamera, pRawData)
            if platform.system() == "Windows":
                mvsdk.CameraFlipFrameBuffer(self.camera.pFrameBuffer, FrameHead, 1)

            # 🌟 核心修复：自动识别黑白/彩色相机，防止 reshape 崩溃
            channels = 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3
            w, h = FrameHead.iWidth, FrameHead.iHeight
            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(self.camera.pFrameBuffer)

            if channels == 1:
                frame_raw = np.frombuffer(frame_data, dtype=np.uint8).reshape((h, w))
                frame = cv2.cvtColor(frame_raw, cv2.COLOR_GRAY2BGR) # 转为彩色方便画框
            else:
                frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((h, w, 3))

            title = "Select the BALL (drag a box around it, ENTER to confirm)"
            roi = cv2.selectROI(title, frame, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow(title)

            if roi[2] < 4 or roi[3] < 4:
                messagebox.showwarning("警告", "框选区域太小或已取消。")
                return

            # 🌟 框的大小就是小球直径：框多大、球就按多大去匹配
            bx, by = roi[0] + roi[2] / 2.0, roi[1] + roi[3] / 2.0
            r = max(3.0, min(roi[2], roi[3]) / 2.0)
            self.radius_var.set(round(r, 1))
            self.search_var.set(round(max(30.0, 4 * r), 1))
            self.selected_area = roi
            self.ball_pos = (bx, by)
            self._ball_vel = (0.0, 0.0)
            self._reset_angle()

            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hit = detect_ball(gray_full, self.ball_pos, r, self.search_var.get(), self.dark_var.get())
            if hit is None:
                self.status_lbl.config(
                    text="状态: 框选完成，但这帧没检出小球，试试调半径或取消『小球比背景暗』",
                    foreground="red")
            else:
                self.ball_pos = (hit[0], hit[1])
                self.status_lbl.config(
                    text=f"状态: 已锁定小球 ({hit[0]:.1f}, {hit[1]:.1f}) 半径{r:.0f}px 信噪比{hit[2]:.1f}",
                    foreground="green")
        except Exception as e:
            messagebox.showerror("相机错误", f"抓取图像失败: {e}")

    def _toggle_monitoring(self):
        if not self.monitoring:
            if self.ball_pos is None:
                messagebox.showwarning("警告", "请先框选小球！")
                return
            self.monitoring = True
            self._stop_worker = False
            self.toggle_btn.config(text="停止追踪")
            threading.Thread(target=self._worker_loop, daemon=True).start()
        else:
            self.monitoring = False
            self._stop_worker = True
            self.toggle_btn.config(text="2. 开始追踪小球")

    def _toggle_angle_measurement(self):
        if self.angle_measuring:
            self.angle_measuring = False
            self.angle_measure_btn.config(text="3. 锁定中心并开始测角速度")
            return

        with self._fit_lock:
            if self.center_xy is None:
                messagebox.showwarning("警告", "尚未拟合出稳定的旋转中心，请稍等或重新框选！")
                return
            self.angle_center_xy = self.center_xy.copy()

        with self._track_lock:
            if self.ball_pos is None:
                messagebox.showwarning("警告", "还没跟上小球，无法测角！")
                return
            theta = math.degrees(math.atan2(self.ball_pos[1] - self.angle_center_xy[1],
                                            self.ball_pos[0] - self.angle_center_xy[0]))
            self.angle_point_state = {"ball": {"last_theta": theta, "accum": 0.0}}


        self._accum_angle = 0.0
        self._angle_hist.clear()
        self.angle_measuring = True
        self.angle_measure_btn.config(text="停止测角速度")

    def _reset_angle(self):
        self._accum_angle = 0.0
        self._speed_display = 0.0
        self._angle_hist.clear()
        with self._track_lock:
            self.ball_traj.clear()
        with self._fit_lock:
            self.center_xy = None
            self.fit_radius = None
        self.position_lbl.config(text="旋转中心: N/A")
        self.radius_lbl.config(text="回转半径: N/A")
        self.angle_lbl.config(text=f"当前累计角度: 0.0000{DEG}")
        self.speed_lbl.config(text=f"实时角速度: 0.00000{DEG}/s")

    def _worker_loop(self):
        import mvsdk
        lost = 0

        while not self._stop_worker and self.monitoring:
            t_now = time.perf_counter()
            try:
                pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self.camera.hCamera, 1000)
                mvsdk.CameraImageProcess(self.camera.hCamera, pRawData, self.camera.pFrameBuffer, FrameHead)
                mvsdk.CameraReleaseImageBuffer(self.camera.hCamera, pRawData)
                if platform.system() == "Windows":
                    mvsdk.CameraFlipFrameBuffer(self.camera.pFrameBuffer, FrameHead, 1)

                # 🌟 自动识别黑白/彩色相机，防止 reshape 崩溃
                channels = 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3
                frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(self.camera.pFrameBuffer)

                if channels == 1:
                    gray = np.frombuffer(frame_data, dtype=np.uint8).reshape(
                        (FrameHead.iHeight, FrameHead.iWidth))
                    full_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                else:
                    full_frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(
                        (FrameHead.iHeight, FrameHead.iWidth, 3))
                    gray = cv2.cvtColor(full_frame, cv2.COLOR_BGR2GRAY)

                # ---------- 只盯小球，不碰背景 ----------
                # 实测过：这套棋盘格+涡旋核花纹的背景里，一旦不加约束地信任
                # 检测结果，DoG 响应偶尔会被水槽内壁固定卡口花纹或涡旋核附近
                # 的褶皱抢走峰值——而且这些背景伪特征是静止的、每帧都在同一
                # 个地方给出稳定响应，一旦被锁上就再也甩不掉，表现出来就是
                # “怎么追都在追背景”。用两道物理约束过滤掉这类假检测：
                #   1) 运动一致性——真球两帧间的位移不可能突然跳变一大截；
                #   2) 轨道半径一致性——一旦圆心拟合出来了，球到圆心的距离
                #      应该稳定在回转半径附近，背景伪特征通常离得老远。
                r = max(3.0, float(self.radius_var.get()))
                sr = max(10.0, float(self.search_var.get()))
                dark = bool(self.dark_var.get())

                with self._fit_lock:
                    center_now, radius_now = self.center_xy, self.fit_radius

                guess = (self.ball_pos[0] + self._ball_vel[0], self.ball_pos[1] + self._ball_vel[1])
                hit = detect_ball(gray, guess, r, sr, dark)
                if hit is not None:
                    hx, hy, _ = hit
                    jump_ok = math.hypot(hx - guess[0], hy - guess[1]) <= 5.0 * max(r, sr * 0.5)
                    if not (jump_ok and _orbit_gate(hx, hy, center_now, radius_now)):
                        hit = None      # 跳变太大或偏离已知轨道半径，判定为背景伪特征，丢掉

                if hit is None:
                    lost += 1
                    if lost == 15:
                        self._push_state(status="状态: 跟丢了，正在附近逐步扩大范围找回…")
                    if lost > 15:
                        # 不做全画面盲搜：以最后已知位置为中心逐步放大局部搜索半径，
                        # 范围始终有界，够不到水槽内壁那些远处的固定背景伪特征；
                        # 找到候选后仍然要求它落在轨道半径带内才接受。
                        widen = 1 + min(4, (lost - 15) // 8)
                        hit_wide = detect_ball(gray, self.ball_pos, r, sr * widen, dark)
                        if hit_wide is not None and _orbit_gate(hit_wide[0], hit_wide[1], center_now, radius_now):
                            hit = hit_wide
                    if hit is None:
                        time.sleep(0.005)
                        continue
                    self._push_state(status="状态: 已找回小球，继续追踪")
                lost = 0

                bx, by, score = hit
                with self._track_lock:
                    self._ball_vel = (bx - self.ball_pos[0], by - self.ball_pos[1])
                    self.ball_pos = (bx, by)
                    self.ball_traj.append((t_now, bx, by))

                # ---------- 旋转中心：用小球自己的轨迹做同心圆拟合 ----------
                # 原来的 concentric_residuals 照用，只是喂进去的从"一堆光流点的轨迹"
                # 换成"小球一条轨迹"——单条轨迹时它就是在最小化半径的方差，
                # 也就是标准的几何圆拟合，完全不依赖背景特征。
                with self._track_lock:
                    traj = np.array([[p[1], p[2]] for p in self.ball_traj], dtype=np.float64)
                if len(traj) > 30:
                    guess = self.center_xy if self.center_xy is not None else traj.mean(axis=0)
                    try:
                        res = least_squares(concentric_residuals, x0=guess, args=([traj],))
                        cx, cy = float(res.x[0]), float(res.x[1])
                        d = np.hypot(traj[:, 0] - cx, traj[:, 1] - cy)
                        th = np.unwrap(np.arctan2(traj[:, 1] - cy, traj[:, 0] - cx))
                        span_deg = abs(math.degrees(th[-1] - th[0]))
                        # 转过的弧太短时圆心是外推出来的，不可信，先不更新
                        if span_deg > 20.0 and d.mean() > 2.0:
                            with self._fit_lock:
                                self.center_xy = res.x
                                self.fit_radius = float(d.mean())
                                self.fit_rms = float(d.std())
                            self._push_state(
                                position=f"旋转中心: ({cx:.1f}, {cy:.1f})",
                                radius_text=f"回转半径: {self.fit_radius:.1f} px "
                                            f"(拟合残差 {self.fit_rms:.2f} px, 已扫过 {span_deg:.0f}°)")
                        elif self.center_xy is None:
                            self._push_state(
                                position=f"旋转中心: 待定 (才扫过 {span_deg:.0f}°，需 >20°)")
                    except Exception:
                        pass

                # ---------- 角度累计与角速度（沿用原来的逻辑，单点版）----------
                if self.angle_measuring and self.angle_center_xy is not None:
                    theta = math.degrees(math.atan2(by - self.angle_center_xy[1],
                                                    bx - self.angle_center_xy[0]))
                    st = self.angle_point_state.get("ball")
                    if st is not None:
                        delta = (theta - st["last_theta"] + 180) % 360 - 180
                        st["accum"] += delta
                        st["last_theta"] = theta
                        self._accum_angle = st["accum"]
                        self._angle_hist.append((t_now, self._accum_angle))

                        win = max(0.2, float(self.speed_win_var.get()))
                        history = [p for p in self._angle_hist if p[0] >= t_now - win]
                        if len(history) > 5:
                            t_arr = np.array([p[0] for p in history])
                            a_arr = np.array([p[1] for p in history])
                            slope, _ = np.polyfit(t_arr - t_arr[0], a_arr, 1)
                            self._speed_display = slope

                        self._push_state(
                            angle_text=f"当前累计角度: {self._accum_angle:.2f}{DEG}",
                            speed_text=f"实时角速度: {self._speed_display:.3f}{DEG}/s"
                        )

                # ---------- 预览：整幅画面缩小后画球、轨迹、圆心 ----------
                sc = 640.0 / max(full_frame.shape[0], full_frame.shape[1])
                sc = min(1.0, sc)
                preview = cv2.resize(full_frame, None, fx=sc, fy=sc) if sc < 1.0 else full_frame.copy()
                p = (int(bx * sc), int(by * sc))
                cv2.circle(preview, p, max(3, int(r * sc)), (0, 255, 0), 2)
                cv2.drawMarker(preview, p, (0, 255, 0), cv2.MARKER_CROSS, 16, 1)
                if len(traj) > 1:
                    pts = (traj * sc).astype(np.int32)
                    cv2.polylines(preview, [pts], False, (0, 200, 255), 1)
                with self._fit_lock:
                    c = self.center_xy
                    rad = self.fit_radius
                if c is not None:
                    cp = (int(c[0] * sc), int(c[1] * sc))
                    if rad:
                        cv2.circle(preview, cp, max(2, int(rad * sc)), (255, 120, 0), 1)
                        # 半透明提示环：这是背景伪特征过滤用的轨道半径容许带
                        # (0.5~1.6倍回转半径)，落在带外的检测会被判定为误检丢弃
                        cv2.circle(preview, cp, max(2, int(0.5 * rad * sc)), (120, 120, 120), 1)
                        cv2.circle(preview, cp, max(2, int(1.6 * rad * sc)), (120, 120, 120), 1)
                    cv2.drawMarker(preview, cp, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

                self._push_frame(preview)

            except Exception:
                time.sleep(0.05)

    def _push_state(self, **kwargs):
        try: self.state_q.put_nowait(kwargs)
        except queue.Full: pass

    def _push_frame(self, frame):
        try:
            while not self.frame_q.empty(): self.frame_q.get_nowait()
            self.frame_q.put_nowait(frame)
        except queue.Full: pass

    def _ui_pump(self):
        try:
            while True:
                upd = self.state_q.get_nowait()
                if 'status' in upd: self.status_lbl.config(text=upd['status'], foreground="blue")
                if 'position' in upd: self.position_lbl.config(text=upd['position'])
                if 'radius_text' in upd: self.radius_lbl.config(text=upd['radius_text'])
                if 'angle_text' in upd: self.angle_lbl.config(text=upd['angle_text'])
                if 'speed_text' in upd: self.speed_lbl.config(text=upd['speed_text'])
        except queue.Empty: pass

        try:
            frame = self.frame_q.get_nowait()
            cw, ch = self.preview_canvas.winfo_width(), self.preview_canvas.winfo_height()
            if cw > 10 and ch > 10:
                frame_resized = cv2.resize(frame, (cw, ch))
                rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                tk_img = ImageTk.PhotoImage(Image.fromarray(rgb))
                if self._preview_img_item is None:
                    self._preview_img_item = self.preview_canvas.create_image(0, 0, anchor="nw", image=tk_img)
                else:
                    self.preview_canvas.itemconfig(self._preview_img_item, image=tk_img)
                self.preview_canvas.image = tk_img
        except queue.Empty: pass

        self.root.after(30, self._ui_pump)

    def _on_close(self):
        self._stop_worker = True
        self.monitoring = False
        self.root.destroy()


# =====================================================================
# 视频文件角速度分析窗口
#
# 流程：选文件 -> 第一帧框小球 -> 后台线程跑完整段视频(track_ball_in_video)
# -> 鲁棒圆拟合(fit_circle_robust) -> 角度展开+整体线性回归(compute_angular_velocity)
# -> 显示数字结果 + 三联图 -> 可选导出 CSV / 导出标注视频核验。
# =====================================================================
class VideoAnalysisDialog:
    def __init__(self, parent, video_path=None):
        self.parent = parent
        self.video_path = video_path
        self.roi = None
        self.result = None
        self._cancel = False

        self.top = tk.Toplevel(parent)
        self.top.title("视频文件角速度分析")
        self.top.geometry("760x1000")

        self._build_ui()
        if video_path:
            self.path_var.set(video_path)

        self.top.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        pad = dict(padx=6, pady=4)

        file_frame = ttk.LabelFrame(self.top, text="视频文件", padding=8)
        file_frame.pack(fill=tk.X, **pad)
        self.path_var = tk.StringVar(value="")
        ttk.Entry(file_frame, textvariable=self.path_var, width=60).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(file_frame, text="浏览…", command=self._browse).pack(side=tk.LEFT, padx=4)

        cfg = ttk.LabelFrame(self.top, text="小球与搜索参数", padding=8)
        cfg.pack(fill=tk.X, **pad)

        self.radius_var = tk.DoubleVar(value=18.0)
        self.search_var = tk.DoubleVar(value=70.0)
        self.dark_var = tk.BooleanVar(value=True)
        self.win_var = tk.DoubleVar(value=1.5)

        def _row(r, text, var, tip):
            ttk.Label(cfg, text=text).grid(row=r, column=0, sticky="e", padx=(0, 4), pady=2)
            ttk.Entry(cfg, textvariable=var, width=8, justify="center").grid(row=r, column=1, sticky="w")
            ttk.Label(cfg, text=tip, foreground="#666").grid(row=r, column=2, sticky="w", padx=6)

        _row(0, "小球半径(px):", self.radius_var, "框选时按框大小自动填")
        _row(1, "搜索半径(px):", self.search_var, "两帧间小球最多跑多远")
        _row(2, "瞬时角速度窗口(s):", self.win_var, "诊断图用最近这么久回归瞬时角速度")
        ttk.Checkbutton(cfg, text="小球比背景暗", variable=self.dark_var).grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(4, 0))

        btns = ttk.Frame(self.top)
        btns.pack(fill=tk.X, **pad)
        ttk.Button(btns, text="1. 框选小球(第一帧)", command=self._select_roi).pack(side=tk.LEFT, padx=4)
        self.run_btn = ttk.Button(btns, text="2. 开始分析整段视频", command=self._run_analysis, state=tk.DISABLED)
        self.run_btn.pack(side=tk.LEFT, padx=4)
        self.export_csv_btn = ttk.Button(btns, text="导出CSV", command=self._export_csv, state=tk.DISABLED)
        self.export_csv_btn.pack(side=tk.LEFT, padx=4)
        self.export_video_btn = ttk.Button(btns, text="导出标注视频(核验用)", command=self._export_video, state=tk.DISABLED)
        self.export_video_btn.pack(side=tk.LEFT, padx=4)

        self.progress = ttk.Progressbar(self.top, mode="determinate", maximum=100)
        self.progress.pack(fill=tk.X, **pad)

        self.status_lbl = ttk.Label(self.top, text="状态: 请选择视频文件并框选小球", foreground="blue")
        self.status_lbl.pack(anchor="w", **pad)

        self.summary_lbl = ttk.Label(self.top, text="", justify=tk.LEFT, font=("", 11))
        self.summary_lbl.pack(anchor="w", fill=tk.X, **pad)

        plot_frame = ttk.Frame(self.top)
        plot_frame.pack(fill=tk.BOTH, expand=True, **pad)
        self.fig = Figure(figsize=(7.2, 6.5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _browse(self):
        initial = os.path.dirname(self.path_var.get()) if self.path_var.get() else os.getcwd()
        path = filedialog.askopenfilename(
            title="选择小球运动视频",
            initialdir=initial,
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")]
        )
        if path:
            self.path_var.set(path)

    def _select_roi(self):
        path = self.path_var.get().strip()
        if not path or not os.path.isfile(path):
            messagebox.showerror("错误", "请先选择一个有效的视频文件。")
            return
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            messagebox.showerror("错误", "读取视频第一帧失败。")
            return

        title = "框选小球 (拖框选中，回车确认，Esc取消)"
        roi = cv2.selectROI(title, frame, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(title)
        if roi[2] < 4 or roi[3] < 4:
            messagebox.showwarning("警告", "框选区域太小或已取消。")
            return

        self.roi = roi
        r = max(3.0, min(roi[2], roi[3]) / 2.0)
        self.radius_var.set(round(r, 1))
        self.search_var.set(round(max(30.0, 4 * r), 1))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        guess = (roi[0] + roi[2] / 2.0, roi[1] + roi[3] / 2.0)
        hit = detect_ball(gray, guess, r, self.search_var.get(), self.dark_var.get())
        if hit is None:
            self.status_lbl.config(
                text="状态: 框选完成，但第一帧没检出小球，试试调半径或取消『小球比背景暗』",
                foreground="red")
        else:
            self.status_lbl.config(
                text=f"状态: 已锁定小球，半径≈{r:.0f}px，信噪比{hit[2]:.1f}，可以开始分析",
                foreground="green")
        self.run_btn.config(state=tk.NORMAL)

    def _run_analysis(self):
        path = self.path_var.get().strip()
        if not path or self.roi is None:
            messagebox.showwarning("警告", "请先完成第 1 步框选小球。")
            return
        self.run_btn.config(state=tk.DISABLED)
        self.export_csv_btn.config(state=tk.DISABLED)
        self.export_video_btn.config(state=tk.DISABLED)
        self._cancel = False
        self.progress.config(value=0)
        self.status_lbl.config(text="状态: 正在逐帧追踪…", foreground="blue")
        threading.Thread(target=self._analysis_thread, args=(path,), daemon=True).start()

    def _analysis_thread(self, path):
        try:
            def progress_cb(i, n):
                pct = 100.0 * i / max(1, n)
                self.top.after(0, lambda: self.progress.config(value=pct))

            data = track_ball_in_video(
                path, self.roi,
                ball_radius=float(self.radius_var.get()),
                search_radius=float(self.search_var.get()),
                dark_ball=bool(self.dark_var.get()),
                progress_cb=progress_cb,
                cancel_flag=lambda: self._cancel,
            )
            if self._cancel:
                self.top.after(0, lambda: self.status_lbl.config(text="状态: 已取消", foreground="orange"))
                return

            valid = data["valid"]
            n_valid = int(valid.sum())
            if n_valid < 8:
                raise RuntimeError(f"只探测到 {n_valid} 个有效帧，太少了，无法拟合圆。检查半径/搜索半径/明暗设置。")

            valid_idx = np.where(valid)[0]
            pts = np.stack([data["x"][valid_idx], data["y"][valid_idx]], axis=1)
            cx, cy, r_mean, r_std, inlier = fit_circle_robust(pts)
            good_idx = valid_idx[inlier]

            ang = compute_angular_velocity(
                data["t"][good_idx], data["x"][good_idx], data["y"][good_idx],
                cx, cy, win_s=float(self.win_var.get())
            )

            self.result = dict(
                path=path, data=data, cx=cx, cy=cy, r_mean=r_mean, r_std=r_std,
                good_idx=good_idx, n_valid=n_valid, ang=ang,
                ball_radius=float(self.radius_var.get()),
            )
            self.top.after(0, self._show_results)
        except Exception as e:
            err = str(e)
            self.top.after(0, lambda: messagebox.showerror("分析出错", err))
            self.top.after(0, lambda: self.status_lbl.config(text=f"状态: 出错 - {err}", foreground="red"))
        finally:
            self.top.after(0, lambda: self.run_btn.config(state=tk.NORMAL))

    def _show_results(self):
        r = self.result
        data, ang = r["data"], r["ang"]
        omega = ang["omega_mean"]
        omega_err = ang["omega_stderr"]
        direction = "顺时针" if omega < 0 else "逆时针"
        period = 2 * math.pi / abs(omega) if abs(omega) > 1e-9 else float("inf")

        n_total = len(data["valid"])
        n_valid = r["n_valid"]
        n_inlier = len(r["good_idx"])

        summary = (
            f"共 {n_total} 帧，有效探测 {n_valid} 帧({100*n_valid/n_total:.1f}%)，"
            f"圆拟合内点 {n_inlier} 帧\n"
            f"旋转中心: ({r['cx']:.1f}, {r['cy']:.1f}) px    "
            f"回转半径: {r['r_mean']:.1f} ± {r['r_std']:.2f} px\n"
            f"角速度: {omega:.5f} ± {omega_err:.5f} rad/s "
            f"({math.degrees(omega):.3f}{DEG}/s)，{direction}\n"
            f"周期: {period:.2f} s    线性拟合 R^2 = {ang['r2']:.5f}"
        )
        self.summary_lbl.config(text=summary)
        self.status_lbl.config(text="状态: 分析完成", foreground="green")
        self.export_csv_btn.config(state=tk.NORMAL)
        self.export_video_btn.config(state=tk.NORMAL)

        self.fig.clear()
        ax1 = self.fig.add_subplot(2, 2, 1)
        ax2 = self.fig.add_subplot(2, 2, 2)
        ax3 = self.fig.add_subplot(2, 1, 2)

        valid = data["valid"]
        ax1.plot(data["x"][valid], data["y"][valid], '.', ms=2, color="#4C72B0", label="轨迹")
        th = np.linspace(0, 2 * np.pi, 200)
        ax1.plot(r["cx"] + r["r_mean"] * np.cos(th), r["cy"] + r["r_mean"] * np.sin(th),
                 '-', color="#DD8452", lw=1.2, label="拟合圆")
        ax1.plot([r["cx"]], [r["cy"]], '+', color="red", ms=12, mew=2, label="旋转中心")
        ax1.set_title("轨迹与拟合圆 (像素坐标)")
        ax1.set_xlabel("x (px)"); ax1.set_ylabel("y (px)")
        ax1.invert_yaxis()
        ax1.set_aspect("equal", adjustable="datalim")
        ax1.legend(fontsize=8, loc="best")

        t_good = data["t"][r["good_idx"]]
        theta_deg = np.degrees(ang["theta"])
        fit_deg = np.degrees(ang["omega_mean"] * t_good + ang["theta0"])
        ax2.plot(t_good, theta_deg, '.', ms=2, color="#4C72B0", label="展开角度")
        ax2.plot(t_good, fit_deg, '-', color="#DD8452", lw=1.2, label=f"线性拟合 (R^2={ang['r2']:.4f})")
        ax2.set_title("展开角度 vs 时间")
        ax2.set_xlabel("t (s)"); ax2.set_ylabel(f"角度 ({DEG})")
        ax2.legend(fontsize=8, loc="best")

        ax3.plot(ang["win_t"], np.degrees(ang["win_omega"]), '-', color="#55A868", lw=1.2, label="瞬时角速度(滑动窗口)")
        ax3.axhline(math.degrees(omega), color="red", lw=1, ls="--",
                    label=f"整体平均 = {math.degrees(omega):.3f}{DEG}/s")
        ax3.set_title("瞬时角速度 vs 时间 (用来看转速均不均匀)")
        ax3.set_xlabel("t (s)"); ax3.set_ylabel(f"角速度 ({DEG}/s)")
        ax3.legend(fontsize=8, loc="best")

        self.fig.tight_layout()
        self.canvas.draw()

    def _export_csv(self):
        if self.result is None:
            return
        out = filedialog.asksaveasfilename(
            title="导出追踪数据CSV", defaultextension=".csv",
            initialfile=os.path.splitext(os.path.basename(self.result["path"]))[0] + "_track.csv",
            filetypes=[("CSV文件", "*.csv")]
        )
        if not out:
            return
        data, ang, good_idx = self.result["data"], self.result["ang"], self.result["good_idx"]
        theta_full = np.full(len(data["x"]), np.nan)
        theta_full[good_idx] = ang["theta"]
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["frame_idx", "t_s", "x_px", "y_px", "score", "valid",
                        "used_in_fit", "theta_unwrapped_rad"])
            used = np.zeros(len(data["x"]), dtype=bool)
            used[good_idx] = True
            for i in range(len(data["x"])):
                w.writerow([data["frame_idx"][i], f"{data['t'][i]:.5f}",
                            f"{data['x'][i]:.3f}", f"{data['y'][i]:.3f}",
                            f"{data['score'][i]:.3f}", int(data["valid"][i]),
                            int(used[i]),
                            "" if np.isnan(theta_full[i]) else f"{theta_full[i]:.6f}"])
            w.writerow([])
            w.writerow(["center_x_px", f"{self.result['cx']:.3f}"])
            w.writerow(["center_y_px", f"{self.result['cy']:.3f}"])
            w.writerow(["radius_px", f"{self.result['r_mean']:.3f}"])
            w.writerow(["radius_std_px", f"{self.result['r_std']:.3f}"])
            w.writerow(["omega_rad_per_s", f"{ang['omega_mean']:.6f}"])
            w.writerow(["omega_stderr_rad_per_s", f"{ang['omega_stderr']:.6f}"])
            w.writerow(["r_squared", f"{ang['r2']:.6f}"])
        messagebox.showinfo("完成", f"已导出: {out}")

    def _export_video(self):
        if self.result is None:
            return
        out = filedialog.asksaveasfilename(
            title="导出标注视频", defaultextension=".mp4",
            initialfile=os.path.splitext(os.path.basename(self.result["path"]))[0] + "_annotated.mp4",
            filetypes=[("MP4视频", "*.mp4")]
        )
        if not out:
            return
        self.status_lbl.config(text="状态: 正在导出标注视频…", foreground="blue")
        threading.Thread(target=self._export_video_thread, args=(out,), daemon=True).start()

    def _export_video_thread(self, out):
        try:
            r = self.result

            def cb(i, n):
                pct = 100.0 * i / max(1, n)
                self.top.after(0, lambda: self.progress.config(value=pct))

            export_annotated_video(r["path"], out, r["data"], r["cx"], r["cy"],
                                    r["r_mean"], r["ball_radius"], progress_cb=cb)
            self.top.after(0, lambda: self.status_lbl.config(text="状态: 标注视频已导出", foreground="green"))
            self.top.after(0, lambda: messagebox.showinfo("完成", f"已导出: {out}"))
        except Exception as e:
            err = str(e)
            self.top.after(0, lambda: messagebox.showerror("导出出错", err))

    def _on_close(self):
        self._cancel = True
        self.top.destroy()


# =====================================================================
# 独立运行入口：不用开 main_gui，也不用接相机，直接拿视频文件跑角速度分析。
#   python3 rotation_tracker.py [视频路径]
# 不给路径参数时，默认找脚本上一级目录下的“小球运动.mp4”。
# =====================================================================
if __name__ == "__main__":
    default_video = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "小球运动.mp4")
    )
    video_path = sys.argv[1] if len(sys.argv) > 1 else default_video
    if not os.path.isfile(video_path):
        video_path = None

    root = tk.Tk()
    root.withdraw()
    dlg = VideoAnalysisDialog(root, video_path=video_path)
    dlg.top.protocol("WM_DELETE_WINDOW", lambda: (dlg._on_close(), root.destroy()))
    root.mainloop()
