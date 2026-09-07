"""
实时渲染引擎 - 纯Python实现
支持动态显示水位场、位移场、自旋场等
采用双线程架构彻底解决相机采集与计算的冲突
修复 OpenCV 跨线程 GUI 死锁问题
增加 GPU/CPU 自动降级与算法版本自适应机制
"""
import numpy as np
import cv2
import threading
import time
from datetime import datetime
import os
import traceback
import sys

_CN_FONT_CANDIDATES = [
    "C:/Windows/Fonts/simhei.ttf",          # 黑体，Windows 必装
    "C:/Windows/Fonts/msyh.ttc",            # 微软雅黑
    "C:/Windows/Fonts/simsun.ttc",          # 宋体
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]


_CN_FONT_PATH = None            # 探到一次就记住


def _find_cn_font_path():
    """挑一个真的有汉字字形的字体。先试常见路径，再扫系统字体目录，
    用 '水' 字量一下宽度来确认字形存在（没有字形时宽度为 0）。"""
    global _CN_FONT_PATH
    if _CN_FONT_PATH is not None:
        return _CN_FONT_PATH or None
    try:
        from PIL import ImageFont
    except ImportError:
        _CN_FONT_PATH = ""
        return None

    import glob
    paths = list(_CN_FONT_CANDIDATES)
    for pat in ("/usr/share/fonts/truetype/*/*.tt[cf]",
                "/usr/share/fonts/opentype/*/*.tt[cf]",
                "/usr/share/fonts/*/*.tt[cf]"):
        paths.extend(sorted(glob.glob(pat))[:40])

    for path in paths:
        if not os.path.exists(path):
            continue
        try:
            f = ImageFont.truetype(path, 20)
            box = f.getbbox("水")
            if box and (box[2] - box[0]) > 0:      # 确实有这个汉字
                _CN_FONT_PATH = path
                return path
        except Exception:
            continue
    _CN_FONT_PATH = ""
    return None


def _load_cn_font(size):
    """找一个能画中文的字体；找不到就返回 None，调用方回退英文。"""
    path = _find_cn_font_path()
    if not path:
        return None
    try:
        from PIL import ImageFont
        return ImageFont.truetype(path, size)
    except Exception:
        return None


class LiveRenderer:
    """实时渲染器 - 动态显示FCD解调结果"""

    # 显示分辨率上限：所有着色/统计都在缩小后的场上做，
    # 大 ROI 时能把渲染线程的开销压掉一个量级（着色是 O(像素数)）
    MAX_DISPLAY_W = 1000
    MAX_DISPLAY_H = 800

    # 只保留有物理含义的模式。删掉的三个：
    #   涡量  —— FCD 的 (u,v) ∝ ∇h，∇×∇h ≡ 0，而且重建高度时用的最小二乘梯度积分
    #            正好把旋度丢掉，画出来只有解调噪声（就是你看到"啥也没有"的那个）
    #   归一化位移 —— 就是位移场去掉幅度信息，同一份物理量
    #   速度场 —— 代码里直接 return 位移场，完全重复
    MODES = ["height", "gradient", "amplitude", "phase", "ampphase"]
    MODE_NAMES = {
        "height":    "0 瞬时水面高度 h (mm)",
        "gradient":  "1 水面坡度场 (色相=倾斜方向, 明度=坡度大小)",
        "amplitude": "2 锁相振幅 |A| (mm)",
        "phase":     "3 锁相相位 arg(A) (色环一圈=2pi)",
        "ampphase":  "4 振幅+相位 (色相=相位, 明度=振幅)",
    }
    MODE_ASCII = {          # 找不到中文字体时的回退
        "height":    "0 Instant height h (mm)",
        "gradient":  "1 Surface slope (hue=dir, val=|grad|)",
        "amplitude": "2 Lock-in amplitude |A| (mm)",
        "phase":     "3 Lock-in phase arg(A)",
        "ampphase":  "4 Amplitude + Phase",
    }
    # 需要锁相结果才能画的模式
    LOCKIN_MODES = ("amplitude", "phase", "ampphase")

    def __init__(self, fps=15, window_name="FCD Live Monitor"):
        self.fps = fps
        self.window_name = window_name
        self.is_running = False
        self.render_thread = None
        self.paused = False
        
        self.display_mode = "height"
        self.auto_range = True
        self.vmin = None
        self.vmax = None
        
        self.current_h = None
        self.current_u = None
        self.current_v = None
        self.current_phase = None
        self.current_amp = None
        
        self.frame_count = 0
        self.last_update_time = 0
        self.fps_display = 0

        self.mm_per_pixel = 0.12

        # 🌟 重绘缓存：数据/显示设置没变就不重新着色，
        # 省下来的 CPU 直接还给解调线程（这是卡顿的主要来源之一）
        self.data_version = 0
        self._amp_range = (0.0, 0.0)
        self._h_range = (0.0, 0.0)
        self._label_cache = {}
        self._last_render_key = None
        self._last_frame_img = None

    def _create_trackbars(self):
        cv2.createTrackbar('Mode', self.window_name, 0, len(self.MODES) - 1, lambda x: None)
        cv2.createTrackbar('Auto', self.window_name, 1, 1, lambda x: None)
        cv2.createTrackbar('Min', self.window_name, 2, 100, lambda x: None)
        cv2.createTrackbar('Max', self.window_name, 98, 100, lambda x: None)
        
    def set_physical_params(self, mm_per_pixel):
        self.mm_per_pixel = mm_per_pixel
        
    def update_data(self, h, u=None, v=None, phase=None, amp=None):
        if h is None: return
        self.current_h = h
        self.current_u = u
        self.current_v = v
        self.current_phase = phase
        self.current_amp = amp
        self.frame_count += 1
        self.data_version += 1
        # 量程只在拿到新数据时统计一次，别让渲染线程每 30ms 重扫一遍全场
        if amp is not None and amp.size > 0:
            self._amp_range = (float(amp.min()), float(amp.max()))
        if h.size > 0:
            self._h_range = (float(h.min()), float(h.max()))

        current_time = time.time()
        if self.last_update_time > 0:
            dt = current_time - self.last_update_time
            if dt > 0:
                self.fps_display = 0.9 * self.fps_display + 0.1 * (1.0 / dt)
        self.last_update_time = current_time
        
    def _read_trackbars_safely(self):
        try:
            idx = cv2.getTrackbarPos('Mode', self.window_name)
            if idx < len(self.MODES):
                self.display_mode = self.MODES[idx]
                
            self.auto_range = cv2.getTrackbarPos('Auto', self.window_name) == 1
            
            if not self.auto_range:
                min_val = cv2.getTrackbarPos('Min', self.window_name) / 100.0
                max_val = cv2.getTrackbarPos('Max', self.window_name) / 100.0
                lo, hi = (self._amp_range if self.display_mode in self.LOCKIN_MODES
                          else self._h_range)       # 用更新数据时算好的量程，别每 15ms 重扫全场
                if hi > lo:
                    self.vmin = lo + (hi - lo) * min_val
                    self.vmax = lo + (hi - lo) * max_val
        except Exception:
            pass

    def _fit(self, data, wrapped=False):
        """把物理场先缩到显示分辨率再着色。
        对已经小于上限的场直接返回，不产生额外拷贝。
        wrapped=True 用于相位这类带 ±π 跳变的量——面积平均会把 +3.1 和 -3.1
        平成 0，凭空造出假条纹，所以改用最近邻抽样。"""
        if data is None or data.size == 0: return data
        rows, cols = data.shape[:2]
        scale = min(self.MAX_DISPLAY_W / cols, self.MAX_DISPLAY_H / rows, 1.0)
        if scale >= 1.0: return data
        # 场往往是切片视图，cv2 要求连续内存，这里顺手转成连续 float32
        return cv2.resize(np.ascontiguousarray(data, dtype=np.float32),
                          (max(1, int(cols * scale)), max(1, int(rows * scale))),
                          interpolation=cv2.INTER_NEAREST if wrapped else cv2.INTER_AREA)

    def _percentile_pair(self, data, p_low=2, p_high=98):
        """大场用隔点抽样估分位数：视觉上看不出差别，耗时降 4 倍。"""
        sample = data[::2, ::2] if data.size > 250000 else data
        return np.percentile(sample, p_low), np.percentile(sample, p_high)

    def _normalize_to_uint8(self, data):
        if data is None or data.size == 0: return np.zeros((100, 100), dtype=np.uint8)
        if self.auto_range:
            vmin, vmax = self._percentile_pair(data)
        else:
            vmin = self.vmin if self.vmin is not None else np.min(data)
            vmax = self.vmax if self.vmax is not None else np.max(data)
        if vmax - vmin < 1e-6: vmax = vmin + 1e-6
        normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)
        return (normalized * 255).astype(np.uint8)

    def _cn_label(self, text, size=22):
        """把中文渲染成一小条白字黑底位图并缓存；叠加时用 max 混合即可。
        cv2.putText 画不了中文，只能借 PIL。"""
        key = (text, size)
        if key in self._label_cache:
            return self._label_cache[key]
        img = None
        try:
            from PIL import Image, ImageDraw
            font = _load_cn_font(size)
            if font is not None:
                probe = ImageDraw.Draw(Image.new("L", (8, 8)))
                x0, y0, x1, y1 = probe.textbbox((0, 0), text, font=font)
                pil = Image.new("RGB", (max(1, x1 - x0 + 4), max(1, y1 - y0 + 4)), (0, 0, 0))
                ImageDraw.Draw(pil).text((2 - x0, 2 - y0), text, font=font, fill=(255, 255, 255))
                img = np.array(pil)[:, :, ::-1].copy()          # RGB → BGR
        except Exception:
            img = None
        self._label_cache[key] = img
        return img

    def _blit_label(self, img, text, org, size=22, ascii_fallback=None, scale=0.7):
        """把中文标签贴到 img 的 (x, y) 左上角；没有中文字体时回退 ASCII。"""
        label = self._cn_label(text, size)
        x, y = org
        if label is None:
            cv2.putText(img, ascii_fallback or text, (x, y + size),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
            return
        lh = min(label.shape[0], img.shape[0] - y)
        lw = min(label.shape[1], img.shape[1] - x)
        if lh <= 0 or lw <= 0:
            return
        roi = img[y:y + lh, x:x + lw]
        np.maximum(roi, label[:lh, :lw], out=roi)

    def _waiting_img(self, text="Waiting for lock-in..."):
        img = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.putText(img, text, (120, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        return img

    def _render_height(self, h):
        """瞬时水面高度场：这一帧的水面起伏(mm)，FCD 的直接输出。"""
        h = self._fit(h)
        if h is None or h.size == 0: return self._waiting_img("Waiting for FCD data...")
        return cv2.applyColorMap(self._normalize_to_uint8(h), cv2.COLORMAP_JET)

    def _render_gradient(self, u, v):
        """水面坡度场：(u,v) 是 FCD 解出的像素位移，正比于水面梯度 ∇h。
        色相 = 水面倾斜的方向，明度 = 倾斜的大小。
        注意这是空间上的倾斜方向，不是波的相位——它在任何局部极值/鞍点
        周围都会转满一圈，跟拓扑荷无关，别拿它看涡旋。"""
        if u is None or v is None or u.size == 0:
            return self._waiting_img("Waiting for FCD data...")
        u = self._fit(u); v = self._fit(v)
        mag = np.sqrt(u**2 + v**2)
        angle = np.arctan2(v, -u)
        mag_max = self._percentile_pair(mag)[1] if self.auto_range else (self.vmax if self.vmax else 1.0)
        mag_norm = np.clip(mag / (mag_max + 1e-12), 0, 1)

        hsv = np.zeros((mag.shape[0], mag.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = ((angle + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
        hsv[..., 1] = 255
        hsv[..., 2] = (mag_norm * 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _render_amplitude(self, amp):
        """|A|：该点的振动幅度（mm），锁相解出来的包络，不是某一瞬间的水面高度。"""
        amp = self._fit(amp)
        if amp is None or amp.size == 0: return self._waiting_img()
        return cv2.applyColorMap(self._normalize_to_uint8(amp), cv2.COLORMAP_JET)

    def _render_phase(self, phase):
        """arg(A)：相对固定时间原点的相位滞后，色环一圈 = 2π。
        涡旋的拓扑荷就看这张图绕核心一圈换了几轮颜色。"""
        phase = self._fit(phase, wrapped=True)
        if phase is None or phase.size == 0: return self._waiting_img()
        hsv = np.zeros((phase.shape[0], phase.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = ((phase + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
        hsv[..., 1] = 255
        hsv[..., 2] = 255
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _render_ampphase(self, phase, amp):
        """色相=相位、明度=振幅：涡旋核心振幅趋零会自然发黑，好找奇点。"""
        if phase is None or amp is None: return self._waiting_img()
        phase = self._fit(phase, wrapped=True)
        amp = self._fit(amp)
        if phase.size == 0 or amp.size == 0: return self._waiting_img()
        amp_max = self._percentile_pair(amp)[1] if self.auto_range else (self.vmax if self.vmax else 1.0)
        amp_norm = np.clip(amp / (amp_max + 1e-12), 0, 1)

        hsv = np.zeros((phase.shape[0], phase.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = ((phase + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
        hsv[..., 1] = 255
        hsv[..., 2] = (amp_norm * 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _add_overlay(self, img):
        h, w = img.shape[:2]
        mode_text = self.MODE_NAMES.get(self.display_mode, self.display_mode)
        mode_ascii = self.MODE_ASCII.get(self.display_mode, self.display_mode)
        
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
        
        self._blit_label(img, mode_text, (10, 6), size=24, ascii_fallback=mode_ascii)
        cv2.putText(img, f"{self.fps_display:.1f} FPS", (w - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if self.display_mode in self.LOCKIN_MODES and self.current_amp is not None:
            cv2.putText(img, f"Amp: {self._amp_range[0]:.4f} ~ {self._amp_range[1]:.4f} mm",
                        (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        elif self.current_h is not None and self.current_h.size > 0:
            cv2.putText(img, f"h: {self._h_range[0]:.4f} ~ {self._h_range[1]:.4f} mm",
                        (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        self._blit_label(img, "m 切换模式   a 自动量程   +/- 调量程   空格 暂停   q 退出",
                         (10, h - 26), size=16,
                         ascii_fallback="m:mode a:auto +/-:range SPACE:pause q:quit", scale=0.45)
        return img
    
    def render_one_frame(self):
        if self.current_h is None or self.current_h.size == 0:
            return self._waiting_img("Waiting for FCD data...")
        if self.display_mode in self.LOCKIN_MODES and (self.current_amp is None or self.current_phase is None):
            # 锁相要攒够约一个驱动周期的帧才出结果
            return self._waiting_img("Waiting for lock-in (need ~1 drive period)...")
        try:
            if self.display_mode == "height": img = self._render_height(self.current_h)
            elif self.display_mode == "gradient": img = self._render_gradient(self.current_u, self.current_v)
            elif self.display_mode == "amplitude": img = self._render_amplitude(self.current_amp)
            elif self.display_mode == "phase": img = self._render_phase(self.current_phase)
            else: img = self._render_ampphase(self.current_phase, self.current_amp)
            return self._add_overlay(img)
        except Exception as e:
            img = np.zeros((600, 800, 3), dtype=np.uint8)
            cv2.putText(img, f"Render Error: {str(e)[:50]}", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return img
    
    def show(self):
        self._read_trackbars_safely()
        # 只有数据或显示设置真的变了才重新着色，否则直接复用上一帧位图
        render_key = (self.data_version, self.display_mode, self.auto_range, self.vmin, self.vmax)
        if render_key != self._last_render_key or self._last_frame_img is None:
            self._last_frame_img = self.render_one_frame()
            self._last_render_key = render_key
        cv2.imshow(self.window_name, self._last_frame_img)
        key = cv2.waitKey(15) & 0xFF
        if key == ord('q') or key == 27: return False
        elif key == ord('m'):
            try: cv2.setTrackbarPos('Mode', self.window_name,
                                    (self.MODES.index(self.display_mode) + 1) % len(self.MODES))
            except ValueError: pass
        elif key == ord('a'): cv2.setTrackbarPos('Auto', self.window_name, 0 if self.auto_range else 1)
        elif key == ord('+') or key == ord('='): cv2.setTrackbarPos('Min', self.window_name, min(cv2.getTrackbarPos('Min', self.window_name) + 5, 98))
        elif key == ord('-') or key == ord('_'): cv2.setTrackbarPos('Min', self.window_name, max(cv2.getTrackbarPos('Min', self.window_name) - 5, 0))
        elif key == 32: return "pause"
        return True
    
    def start(self):
        if self.is_running: return
        self.is_running = True
        def render_loop():
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 1024, 768)
            self._create_trackbars()
            while self.is_running:
                try:
                    result = self.show()
                    if result is False: break
                    elif result == "pause":
                        self.paused = True
                        while self.paused and self.is_running:
                            k = cv2.waitKey(100) & 0xFF
                            if k == 32: self.paused = False
                            elif k in (ord('q'), 27): self.is_running = False; break
                except Exception: time.sleep(0.1)
            try: cv2.destroyWindow(self.window_name)
            except: pass
        self.render_thread = threading.Thread(target=render_loop, daemon=True)
        self.render_thread.start()
        
    def stop(self):
        self.is_running = False
        self.paused = False
        
    def save_snapshot(self, save_dir):
        if self.current_h is None: return None
        filename = os.path.join(save_dir, f"live_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(filename, self.render_one_frame())
        return filename

# ====================================================================
# LiveProcessThread (采集与 GPU/CPU 自适应计算双线程架构)
# ====================================================================

try:
    import mvsdk
    import platform
    MVS_AVAILABLE = True
except ImportError:
    MVS_AVAILABLE = False

class LiveProcessThread(threading.Thread):
    def __init__(self, camera, fcd_core, renderer, fps=15, roi_size=512,
                 drive_period_ms=150.0, lockin_periods=4.0):
        super().__init__(daemon=True)
        self.camera = camera
        self.fcd_core = fcd_core
        self.renderer = renderer
        self.target_fps = fps
        self.is_running = False
        self.paused = False

        self.process_fps = 0
        self.process_count = 0
        self.Iref = None

        # 🌟 ROI：FCD 的耗时随像素数超线性增长（解调 O(N²logN)、积分器 O(N³)），
        # 整幅 5MP 图实时解调一帧要几秒，只有取中心 ROI 才能把延迟压到几十毫秒。
        self.roi_size = int(roi_size) if roi_size else 0
        self.crop_box = None          # (x1, x2, y1, y2)，抓图线程直接按它裁

        # 每帧都重算的常量：参考图的 float32 副本、mm/pixel 标定
        self._Iref32 = None
        self._mm_per_pixel = None

        # 🌟 锁相解调：喇叭是被单一驱动频率驱动的，所以水面每一点都在做
        # h(x,y,t) = |A|·cos(2πft + φ(x,y)) 的简谐振动。在驱动频率上做锁相
        # 就能同时拿到振幅 |A| 和相位 φ；涡旋的拓扑荷 = φ 绕核心一圈的缠绕数。
        self.drive_freq = 1000.0 / float(drive_period_ms) if drive_period_ms and drive_period_ms > 0 else 0.0
        self.lockin_periods = float(lockin_periods)
        self._lockin_t0 = None            # 固定的时间原点，保证相位图不逐帧乱转
        self._buf_t = []
        self._buf_h = []
        self._lockin_ready = False

        self.latest_frame = None
        self.frame_lock = threading.Lock()

    def _resolve_crop(self, shape, requested):
        """确定解调区域：优先用上层框选出来的框（越界会被夹到画面内），
        没框选就按 roi_size 在画面正中取一块方形 ROI，roi_size<=0 才用整幅。"""
        rows, cols = shape
        x1, x2, y1, y2 = (int(v) for v in requested)
        if x2 > x1 and y2 > y1:
            x1 = max(0, min(x1, cols - 1)); x2 = max(x1 + 1, min(x2, cols))
            y1 = max(0, min(y1, rows - 1)); y2 = max(y1 + 1, min(y2, rows))
            if x2 - x1 >= 16 and y2 - y1 >= 16:
                print(f"🔲 使用框选区域: X[{x1}:{x2}] Y[{y1}:{y2}] ({x2-x1}x{y2-y1})，原图 {cols}x{rows}")
                return (x1, x2, y1, y2)
            print(f"⚠️ 框选区域越界或过小 ({requested})，改用中心 ROI。")

        if self.roi_size <= 0:
            print(f"⚠️ 未框选区域且 ROI=0，将对整幅 {cols}x{rows} 图像解调，会很慢。")
            return None

        s = min(self.roi_size, rows, cols)
        s -= s % 2
        ox, oy = (cols - s) // 2, (rows - s) // 2
        print(f"🔲 未框选区域，使用中心 ROI: {s}x{s} @ ({ox},{oy})，原图 {cols}x{rows}")
        return (ox, ox + s, oy, oy + s)

    def _lockin_update(self, h_field, t_now):
        """在驱动频率上锁相，返回 (|A|, arg(A))；帧数不够时返回 (None, None)。

        逐点做一次四参数最小二乘：

            h_k ≈ a + b·τ_k + c·cos(ω(t_k-t0)) + d·sin(ω(t_k-t0)),  ω = 2πf

        直流 a、慢漂移 b 和正弦一起解，所以窗口不是整数个周期、采样不均匀
        都不会把直流/漂移漏进振幅里（单纯投影到 e^{-iωt} 会漏）。
        复振幅 A = c - i·d，|A| 是振幅、arg(A) 是相位。

        四个基函数只依赖时间，法方程 G=XᵀX 是全图共用的 4×4 小矩阵，
        逐点只需要 Xᵀh 这四个累加量，所以代价和普通锁相差不多。
        相位原点 t0 取线程启动时刻并固定不变，否则窗口一滑整张相位图就乱转；
        而 a、b 用的 τ 是窗口内中心化的时间，避免跑久了 t 很大导致病态。
        """
        if self.drive_freq <= 0:
            return None, None

        if self._buf_h and self._buf_h[-1].shape != h_field.shape:
            self._buf_h.clear(); self._buf_t.clear()

        self._buf_h.append(np.ascontiguousarray(h_field, dtype=np.float32))
        self._buf_t.append(t_now)

        win = self.lockin_periods / self.drive_freq
        while len(self._buf_t) > 2 and (t_now - self._buf_t[0]) > win:
            self._buf_t.pop(0); self._buf_h.pop(0)

        n = len(self._buf_t)
        span = self._buf_t[-1] - self._buf_t[0]
        if n < 6 or span < 0.9 / self.drive_freq:
            return None, None              # 还没攒够一个驱动周期

        t = np.asarray(self._buf_t, dtype=np.float64)
        phase_arg = 2.0 * np.pi * self.drive_freq * (t - self._lockin_t0)
        tau = t - t.mean()                 # 中心化，保证 [1, τ] 不病态
        X = np.stack([np.ones(n), tau, np.cos(phase_arg), np.sin(phase_arg)], axis=1)

        G = X.T @ X
        try:
            if np.linalg.cond(G) > 1e8:
                return None, None          # 窗口太短，正弦基和线性基分不开
            Ginv = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            return None, None

        Xf = X.astype(np.float32)
        rhs = np.zeros((4,) + h_field.shape, dtype=np.float32)
        for hk, xk in zip(self._buf_h, Xf):
            rhs += xk[:, None, None] * hk[None, :, :]

        coef = np.tensordot(Ginv.astype(np.float32), rhs, axes=1)   # (4, H, W)
        A = coef[2] - 1j * coef[3]

        if not self._lockin_ready:
            self._lockin_ready = True
            fpp = n / (span * self.drive_freq)      # 每个驱动周期实际采到几帧
            print(f"🔒 锁相已就绪: 驱动 {self.drive_freq:.3f} Hz, "
                  f"窗口 {span:.2f}s / {n} 帧 ({fpp:.1f} 帧/周期)")
            if fpp < 3.0:
                print(f"⚠️ 每个驱动周期只采到 {fpp:.1f} 帧，低于 3 帧时相位会明显失真。"
                      f"建议加大驱动周期(降频)或提高相机帧率/缩小 ROI。")
        return np.abs(A).astype(np.float32), np.angle(A).astype(np.float32)

    def camera_grab_loop(self):
        print("📸 相机极速采集线程已启动...")
        while self.is_running:
            if self.paused:
                time.sleep(0.1)
                continue
            try:
                pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self.camera.hCamera, 1000)
                mvsdk.CameraImageProcess(self.camera.hCamera, pRawData, self.camera.pFrameBuffer, FrameHead)
                mvsdk.CameraReleaseImageBuffer(self.camera.hCamera, pRawData)
                
                if platform.system() == "Windows":
                    mvsdk.CameraFlipFrameBuffer(self.camera.pFrameBuffer, FrameHead, 1)
                
                channels = 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3
                w, h = FrameHead.iWidth, FrameHead.iHeight
                frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(self.camera.pFrameBuffer)
                Idef = np.frombuffer(frame_data, dtype=np.uint8).reshape((h, w, channels))
                
                if channels == 3: Idef = cv2.cvtColor(Idef, cv2.COLOR_BGR2GRAY)
                else: Idef = Idef.reshape((h, w))

                # 先裁 ROI 再拷贝：5MP 整幅拷贝 ~5MB/帧，裁完只剩几百 KB
                if self.crop_box is not None:
                    x1, x2, y1, y2 = self.crop_box
                    if h >= y2 and w >= x2:
                        Idef = Idef[y1:y2, x1:x2]

                with self.frame_lock:
                    self.latest_frame = Idef.copy()

            except mvsdk.CameraException as e:
                if e.error_code != -12: print(f"⚠️ 相机 SDK 抓图错误: {e}")
                time.sleep(0.001)
            except Exception:
                time.sleep(0.01)

    def run(self):
        self.is_running = True
        
        # 1. 检查并加载参考图像
        if self.fcd_core.ref_path and os.path.exists(self.fcd_core.ref_path):
            try:
                # 整幅读入参考图，裁切框统一在这里定，越界/过小都能兜住
                requested = tuple(self.fcd_core.crop)
                self.fcd_core.crop = (0, 0, 0, 0)
                full_ref = self.fcd_core._read_and_crop(self.fcd_core.ref_path)

                self.crop_box = self._resolve_crop(full_ref.shape, requested)
                if self.crop_box is not None:
                    x1, x2, y1, y2 = self.crop_box
                    self.Iref = full_ref[y1:y2, x1:x2]
                    self.fcd_core.crop = self.crop_box   # 让 core 与抓图线程用同一个框
                else:
                    self.Iref = full_ref
                self._Iref32 = np.ascontiguousarray(self.Iref, dtype=np.float32)
                # mm/pixel 只由参考图决定，是个常数——原来每帧都重算一次 FFT 找载波
                self._mm_per_pixel = self.fcd_core._estimate_mm_per_pixel(self.Iref)
                self.renderer.set_physical_params(self._mm_per_pixel)
                print(f"✅ 参考图像加载成功！解调尺寸 {self.Iref.shape[1]}x{self.Iref.shape[0]}, "
                      f"标定 {self._mm_per_pixel:.5f} mm/px")
            except Exception as e:
                print(f"❌ 加载参考图像失败: {e}")
                self.is_running = False
                return
        else:
            print("❌ 实时渲染失败：未找到参考图像！请在主界面选择有效的参考图像路径。")
            self.is_running = False
            return
            
        # 2. 启动相机流
        if self.camera and self.camera.is_opened and MVS_AVAILABLE:
            try:
                mvsdk.CameraSetTriggerMode(self.camera.hCamera, 0)
                mvsdk.CameraPlay(self.camera.hCamera)
                print("✅ 相机视频流已开启！")
            except Exception as e:
                print(f"⚠️ 开启相机流警告: {e}")
                
            self.grab_thread = threading.Thread(target=self.camera_grab_loop, daemon=True)
            self.grab_thread.start()
        else:
            print("⚠️ 相机未连接，将使用模拟测试数据进行渲染...")
            
        last_log_time = time.time()
        
        # ================= 智能环境探测 =================
        use_gpu = False
        has_gpu_methods = hasattr(self.fcd_core, '_fcd_demodulate_correct_gpu')
        
        if has_gpu_methods:
            try:
                import cupy as cp
                _ = cp.array([1.0]) 
                use_gpu = True
                print("🚀 检测到可用 GPU 及 GPU 算法，已开启 CUDA 硬件加速！")
            except Exception as e:
                print(f"⚠️ 未检测到可用 GPU ({e})。")
                print("🐌 已自动降级为标准 CPU 模式运行。")
        else:
            print("ℹ️ 当前算法库为标准 CPU 版本，使用 Numpy 进行解调。")
        # =================================================

        Iref_tensor = None      # GPU 路径下常驻显存的参考图

        # 锁相的固定时间原点
        self._lockin_t0 = time.time()
        if self.drive_freq <= 0:
            print("⚠️ 驱动周期无效，无法锁相解调，振幅/相位将不可用。")
        else:
            print(f"🔒 锁相解调: 驱动频率 {self.drive_freq:.3f} Hz "
                  f"(周期 {1000.0/self.drive_freq:.1f} ms)，积分窗口 {self.lockin_periods:g} 个周期")


        # 3. 核心计算循环
        while self.is_running:
            frame_start = time.time()
            Idef = None
            
            if self.camera and self.camera.is_opened and MVS_AVAILABLE:
                with self.frame_lock:
                    if self.latest_frame is not None:
                        Idef = self.latest_frame
                        self.latest_frame = None
            else:
                self._generate_test_data()
                self.process_count += 1
                time.sleep(0.05)
                continue
                
            if Idef is not None:
                try:
                    # ROI 已经在抓图线程裁好了，这里只做一次尺寸兜底
                    Idef_cropped = Idef
                    if Idef_cropped.shape != self.Iref.shape:
                        Idef_cropped = cv2.resize(Idef_cropped, (self.Iref.shape[1], self.Iref.shape[0]))

                    # 动态选择计算路径
                    if use_gpu and has_gpu_methods:
                        if Iref_tensor is None:          # 参考图常驻显存，不必每帧重传
                            Iref_tensor = cp.array(self.Iref, dtype=cp.float32)
                        Idef_tensor = cp.array(Idef_cropped, dtype=cp.float32)
                        u_tensor, v_tensor = self.fcd_core._fcd_demodulate_correct_gpu(Iref_tensor, Idef_tensor)
                        h_tensor = self.fcd_core._fftinvgrad_gpu(-u_tensor, -v_tensor)

                        u_px = cp.asnumpy(u_tensor)
                        v_px = cp.asnumpy(v_tensor)
                        h_field_raw = cp.asnumpy(h_tensor)
                    else:
                        Idef_np = Idef_cropped.astype(np.float32)
                        u_px, v_px = self.fcd_core._fcd_demodulate_correct(self._Iref32, Idef_np)
                        h_field_raw = self.fcd_core._fftinvgrad(-u_px, -v_px)

                    e = self.fcd_core.edge
                    if u_px.shape[0] > 2*e and u_px.shape[1] > 2*e:
                        u_crop = u_px[e:-e, e:-e]; v_crop = v_px[e:-e, e:-e]; h_field = h_field_raw[e:-e, e:-e]
                    else:
                        u_crop = u_px; v_crop = v_px; h_field = h_field_raw

                    if u_crop.size > 0:
                        mm_per_pixel = self._mm_per_pixel
                        h_field = h_field * (mm_per_pixel ** 2) / self.fcd_core.H

                        if hasattr(self.fcd_core, '_remove_background_surface'):
                            h_field = self.fcd_core._remove_background_surface(h_field, order=3)
                        else:
                            h_field = h_field - np.mean(h_field)

                        u_field = u_crop * mm_per_pixel
                        v_field = v_crop * mm_per_pixel

                        # 在驱动频率上锁相，拿到真正的振幅与相位
                        amp_field, phase_field = self._lockin_update(h_field, time.time())

                        self.renderer.update_data(h_field, u_field, v_field, phase_field, amp_field)
                        self.process_count += 1

                except Exception as e:
                    print(f"❌ FCD 计算发生致命错误:\n{e}\n{traceback.format_exc()}")
            else:
                time.sleep(0.002)
                
            elapsed = time.time() - frame_start
            if elapsed > 0:
                self.process_fps = 0.9 * self.process_fps + 0.1 * (1.0 / elapsed)
                
            if time.time() - last_log_time > 5:
                if self.process_count > 0:
                    mode_str = "GPU" if (use_gpu and has_gpu_methods) else "CPU"
                    print(f"⚡ FCD {mode_str} 实时解调中: {self.process_fps:.1f} FPS, 已处理帧数: {self.process_count}")
                last_log_time = time.time()
                
    def _generate_test_data(self):
        size = 200
        X, Y = np.meshgrid(np.linspace(-np.pi, np.pi, size), np.linspace(-np.pi, np.pi, size))
        h_test = np.sin(X) * np.cos(Y) * 0.5
        u_test = np.cos(X) * np.sin(Y) * 0.3
        self.renderer.update_data(h_test, u_test, -np.sin(X) * np.cos(Y) * 0.3, np.angle(h_test + 1j * u_test), np.abs(h_test + 1j * u_test))
                
    def stop(self):
        self.is_running = False