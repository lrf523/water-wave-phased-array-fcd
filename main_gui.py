import os
import json
import math
import colorsys
import importlib
import inspect
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import traceback
import time
import threading
import serial.tools.list_ports
import numpy as np
import scipy.signal as signal

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    from rotation_tracker import RotationTrackerUI
except ImportError:
    RotationTrackerUI = None

try:
    from speaker_controller import SpeakerArrayController
except ImportError:
    messagebox.showerror("导入错误", "找不到 speaker_controller.py，请确保它与此脚本在同一目录下！")
    exit()

try:
    from camera_controller import MindVisionCamera
except ImportError:
    MindVisionCamera = None

# 🌟 新增：实时渲染模块导入
try:
    from live_renderer import LiveRenderer, LiveProcessThread
    LIVE_RENDER_AVAILABLE = True
except ImportError:
    LIVE_RENDER_AVAILABLE = False

# =====================================================================
# 🌟 FCD 解调核心导入
# 旧代码写的是 `from fcd_core import FCDCore`，但目录里根本没有 fcd_core.py，
# 真正的实现在下面两个文件里（类名都叫 FCDCore）：
#     fcd_backend_gpu.py —— CuPy/CUDA 加速版，只有 *_gpu 算子
#     fcd_backend_syl.py —— Numpy/Scipy 纯 CPU 版，算子最全
# live_renderer 会先探测 _fcd_demodulate_correct_gpu 来决定走 GPU 还是 CPU 路径，
# 所以这里的规则是：cupy 真的能跑通才用 GPU 版，否则一律用 CPU 版，
# 避免“导入了 GPU 版但显卡不可用 → 回退 CPU 时找不到算子”的二次崩溃。
# =====================================================================
FCD_CORE_MODULES = ("fcd_core", "fcd_backend_gpu", "fcd_backend_syl")


def _cupy_runtime_ok():
    """cupy 装上了不等于能用，必须真的在显存里算一次才算数。"""
    try:
        import cupy as cp
        cp.asnumpy(cp.array([1.0]) + 1.0)
        return True
    except Exception:
        return False


def load_fcd_core():
    """解析可用的 FCDCore 类，返回 (类, 模块名, 失败原因列表)；全失败时类为 None。"""
    names = list(FCD_CORE_MODULES)
    if not _cupy_runtime_ok():
        names.remove("fcd_backend_gpu")
        names.append("fcd_backend_gpu")   # 无 GPU 时降到最后，仅作兜底
    errors = []
    for name in names:
        try:
            return getattr(importlib.import_module(name), "FCDCore"), name, errors
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    return None, None, errors


# =====================================================================
# 任意波形发生器 (修复版：安全读值 + 可反复预览)
# =====================================================================
# =====================================================================
# 任意波形发生器 —— NI I/O Trace 实测协议修正版
#
# 实测正确协议：
#   M0/M1/M3:
#       {board}#M0
#       {board}#M1
#       {board}#M3
#
#   发送波形头：
#       {board}#A{points:02X}{phase1:02X}...{phase8:02X}
#
#   数据行：
#       {board}#A{offset:02X}{16点 * %03X}
#
# 例如 64 点、相位 0,16,32,48,63,63,63,63：
#   0#M1
#   0#A40001020303F3F3F
#   0#A00...
#   0#A10...
#   0#A20...
#   0#A30...
#
# 注意：
#   1. 不再发送 0#M2A40...
#   2. 不自动发送 M3，因为实测 M3 可能恢复正弦
#   3. 默认 M 命令目标板为 0，即“所有板”
#   4. ## 参数和 E 使能仍按具体物理板发送，例如 1##、1#E08
# =====================================================================

class ArbitraryWaveformDialog:
    OP_NAMES = {
        0: "设为正弦模式(M0)",
        1: "设为任意模式(M1)",
        3: "同步时钟(M3)"
    }

    WAVE_TYPES = [
        "Custom (自定义公式)",
        "Sine (正弦)",
        "Triangle (三角波)",
        "Flat 2048 (静止平线)",
        "Move-Stop (一动一静)"
    ]

    def __init__(self, parent, main_app):
        self.top = tk.Toplevel(parent)
        self.top.title("任意波形发生器 - NI Trace协议修正版")
        self.top.geometry("1050x860")
        self.top.attributes("-topmost", True)

        self.main_app = main_app

        # M 命令默认走 0# 广播，这是你 LabVIEW 成功时的目标板“所有”
        self.m_board_var = tk.StringVar(value="0")

        # ## 参数和 E 使能需要发给具体物理板，默认从主界面取，否则用 1
        default_param_board = "1"
        try:
            btxt = self.main_app.board_var.get().split()[0]
            if btxt in ["1", "2", "3"]:
                default_param_board = btxt
        except Exception:
            pass

        self.param_board_var = tk.StringVar(value=default_param_board)

        self.start_var = tk.DoubleVar(value=0.0)
        self.end_var = tk.DoubleVar(value=6.28318)
        self.points_var = tk.IntVar(value=64)

        self.type_var = tk.StringVar(value="Custom (自定义公式)")
        self.formula_var = tk.StringVar(value="np.sin(x)")

        # 为了避免炸麦，默认 DAC 表幅度很小。
        # 如果要完全复刻 LabVIEW 满幅，可以手动改成 1.0。
        self.amp_var = tk.DoubleVar(value=0.05)

        # LabVIEW 示例相位：0,16,32,48,64,80,96,112
        # 64 点时会钳位成：0,16,32,48,63,63,63,63
        self.phase_vars = [
            tk.IntVar(value=0),
            tk.IntVar(value=16),
            tk.IntVar(value=32),
            tk.IntVar(value=48),
            tk.IntVar(value=64),
            tk.IntVar(value=80),
            tk.IntVar(value=96),
            tk.IntVar(value=112),
        ]

        # 激活输出用的 ## 和 E 参数
        # CH4 = 0x08。你之前主要测 CH4，所以这里默认 08。
        # 如果要开全部通道，可改 FF。
        self.enable_mask_var = tk.StringVar(value="08")

        # 3333 是你测试中过的标准安全驱动幅度。
        # 但任意波形容易炸麦，所以默认用 0400，更安全。
        self.drive_amp_hex_var = tk.StringVar(value="0400")

        self.period_var = tk.IntVar(value=1000)

        # 完整流程顺序。
        # 你 NI Trace 成功的核心是 0#M1 -> 0#A40 -> 数据行 -> ## -> E。
        self.order_var = tk.StringVar(value="M1 -> A表")

        self.setup_ui()
        self.plot_preview()

    # -----------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------
    def setup_ui(self):
        conf = ttk.LabelFrame(
            self.top,
            text="任意波形协议参数",
            padding=10
        )
        conf.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(conf, text="M命令目标板:").grid(row=0, column=0, padx=5, pady=3, sticky="e")
        ttk.Combobox(
            conf,
            textvariable=self.m_board_var,
            values=["0", "1", "2", "3"],
            width=6,
            state="readonly"
        ).grid(row=0, column=1, padx=5, pady=3, sticky="w")

        ttk.Label(conf, text="##/E激活板:").grid(row=0, column=2, padx=5, pady=3, sticky="e")
        ttk.Combobox(
            conf,
            textvariable=self.param_board_var,
            values=["1", "2", "3"],
            width=6,
            state="readonly"
        ).grid(row=0, column=3, padx=5, pady=3, sticky="w")

        ttk.Label(conf, text="点数(16-256):").grid(row=0, column=4, padx=5, pady=3, sticky="e")
        ttk.Entry(conf, textvariable=self.points_var, width=8).grid(row=0, column=5, padx=5, pady=3)

        ttk.Label(conf, text="DAC表幅度(0-1):").grid(row=0, column=6, padx=5, pady=3, sticky="e")
        ttk.Entry(conf, textvariable=self.amp_var, width=8).grid(row=0, column=7, padx=5, pady=3)

        ttk.Label(conf, text="波形类型:").grid(row=1, column=0, padx=5, pady=3, sticky="e")
        type_cb = ttk.Combobox(
            conf,
            textvariable=self.type_var,
            values=self.WAVE_TYPES,
            width=22,
            state="readonly"
        )
        type_cb.grid(row=1, column=1, columnspan=2, padx=5, pady=3, sticky="w")
        type_cb.bind("<<ComboboxSelected>>", lambda e: self.plot_preview())

        ttk.Label(conf, text="公式/周期数:").grid(row=1, column=3, padx=5, pady=3, sticky="e")
        ttk.Entry(conf, textvariable=self.formula_var, width=36).grid(
            row=1,
            column=4,
            columnspan=3,
            padx=5,
            pady=3,
            sticky="w"
        )

        ttk.Label(conf, text="Start X:").grid(row=2, column=0, padx=5, pady=3, sticky="e")
        ttk.Entry(conf, textvariable=self.start_var, width=8).grid(row=2, column=1, padx=5, pady=3)

        ttk.Label(conf, text="End X:").grid(row=2, column=2, padx=5, pady=3, sticky="e")
        ttk.Entry(conf, textvariable=self.end_var, width=8).grid(row=2, column=3, padx=5, pady=3)

        ttk.Label(conf, text="完整流程顺序:").grid(row=2, column=4, padx=5, pady=3, sticky="e")
        ttk.Combobox(
            conf,
            textvariable=self.order_var,
            values=["M1 -> A表", "A表 -> M1"],
            width=12,
            state="readonly"
        ).grid(row=2, column=5, padx=5, pady=3, sticky="w")

        act = ttk.LabelFrame(
            self.top,
            text="激活输出参数，会发送 ## 和 E；如只想装载波形，可只点“发送波形表”",
            padding=10
        )
        act.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(act, text="使能掩码(hex):").grid(row=0, column=0, padx=5, pady=3, sticky="e")
        ttk.Entry(act, textvariable=self.enable_mask_var, width=8).grid(row=0, column=1, padx=5, pady=3)

        ttk.Label(act, text="驱动幅度hex:").grid(row=0, column=2, padx=5, pady=3, sticky="e")
        ttk.Entry(act, textvariable=self.drive_amp_hex_var, width=8).grid(row=0, column=3, padx=5, pady=3)

        ttk.Label(act, text="period(ms):").grid(row=0, column=4, padx=5, pady=3, sticky="e")
        ttk.Entry(act, textvariable=self.period_var, width=8).grid(row=0, column=5, padx=5, pady=3)

        ttk.Label(
            act,
            text="提示：CH4=08，CH1=01，全部8通道=FF。炸麦时先把幅度hex降到0200/0400。",
            foreground="dimgray"
        ).grid(row=0, column=6, padx=10, pady=3, sticky="w")

        ph = ttk.LabelFrame(
            self.top,
            text="各通道相位差，单位：采样点；超过点数-1会自动钳位",
            padding=10
        )
        ph.pack(fill=tk.X, padx=10, pady=5)

        for i in range(8):
            ttk.Label(ph, text=f"CH{i+1}").grid(row=0, column=i, padx=10, pady=2)
            ttk.Entry(ph, textvariable=self.phase_vars[i], width=6).grid(row=1, column=i, padx=10, pady=2)

        btns = ttk.Frame(self.top)
        btns.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(
            btns,
            text="🔄 预览",
            command=self.plot_preview
        ).pack(side=tk.LEFT, padx=4)

        ttk.Button(
            btns,
            text="📤 仅发送波形表",
            command=self.send_waveform_only
        ).pack(side=tk.LEFT, padx=4)

        ttk.Button(
            btns,
            text="▶ 仅设为任意模式 M1",
            command=lambda: self.run_mode_op(1)
        ).pack(side=tk.LEFT, padx=4)

        ttk.Button(
            btns,
            text="🚀 发送并激活输出",
            command=self.send_and_activate
        ).pack(side=tk.LEFT, padx=8)

        ttk.Button(
            btns,
            text="🔙 切回正弦 M0",
            command=lambda: self.run_mode_op(0)
        ).pack(side=tk.LEFT, padx=4)

        ttk.Button(
            btns,
            text="⏱ M3同步/慎用",
            command=lambda: self.run_mode_op(3)
        ).pack(side=tk.LEFT, padx=4)

        ttk.Label(
            self.top,
            foreground="red",
            text=(
                "⚠️ 当前版本按 NI I/O Trace 修正：M1/M0/M3 不带 A40；"
                "发送波形头是 0#A40...，不是 0#M2A40...；"
                "M3 实测可能恢复正弦，默认不要用于任意波形播放。"
            )
        ).pack(padx=10, pady=3, anchor="w")

        self.fig, self.ax = plt.subplots(figsize=(9.2, 3.4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.top)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    # -----------------------------------------------------------------
    # 安全取值
    # -----------------------------------------------------------------
    def _safe_int(self, var, default=0):
        try:
            return int(var.get())
        except Exception:
            return int(default)

    def _safe_float(self, var, default=0.0):
        try:
            return float(var.get())
        except Exception:
            return float(default)

    def _safe_hex_byte(self, text, default=0):
        try:
            text = str(text).strip().replace("0x", "").replace("0X", "")
            return max(0, min(255, int(text, 16)))
        except Exception:
            return default

    def _safe_hex_word(self, text, default=0x0400):
        try:
            text = str(text).strip().replace("0x", "").replace("0X", "")
            return max(0, min(0xFFFF, int(text, 16)))
        except Exception:
            return default

    def _get_padded_points(self):
        points = self._safe_int(self.points_var, 64)
        points = max(16, min(256, points))

        if points % 16 != 0:
            points = ((points + 15) // 16) * 16

        return min(points, 256)

    # -----------------------------------------------------------------
    # 波形生成
    # -----------------------------------------------------------------
    def _generate_raw_array(self, w_type, formula, points, start_x, end_x):
        if "Flat" in w_type:
            return np.zeros(points)

        if "Move-Stop" in w_type:
            half = points // 2
            t = np.arange(half) / max(1, half)

            # 平滑一动一静，避免硬切炸麦
            active = np.sin(2 * np.pi * 1 * t)
            env = np.sin(np.pi * t) ** 2
            active = active * env

            silent = np.zeros(points - half)
            return np.concatenate([active, silent])

        if "Custom" in w_type:
            x = np.linspace(start_x, end_x, points)
            safe_dict = {
                "np": np,
                "signal": signal,
                "x": x
            }

            y = eval(formula, {"__builtins__": {}}, safe_dict)
            y = np.asarray(y, dtype=float)

            y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
            y = np.clip(y, -1e6, 1e6)

            max_val = np.max(np.abs(y))
            if max_val > 0:
                y = y / max_val

            return np.clip(y, -1.0, 1.0)

        x = np.linspace(0, 1, points, endpoint=False)

        try:
            cycles = float(formula)
        except Exception:
            cycles = 1.0

        if "Sine" in w_type:
            return np.sin(2 * np.pi * cycles * x)

        if "Triangle" in w_type:
            return signal.sawtooth(2 * np.pi * cycles * x, 0.5)

        return np.zeros(points)

    def _make_dac(self):
        points = self._get_padded_points()

        y = self._generate_raw_array(
            self.type_var.get(),
            self.formula_var.get(),
            points,
            self._safe_float(self.start_var, 0.0),
            self._safe_float(self.end_var, 1.0)
        )

        amp = max(0.0, min(1.0, self._safe_float(self.amp_var, 0.05)))

        # 注意：
        # 这里不再固定乘 0.2。
        # 安全性由默认 amp=0.05 和 ## 驱动幅度hex=0400 控制。
        dac = np.clip(
            np.round(np.asarray(y) * amp * 2047 + 2048),
            0,
            4095
        ).astype(np.uint16)

        if len(dac) < points:
            dac = np.concatenate([
                dac,
                np.full(points - len(dac), 2048, dtype=np.uint16)
            ])

        return dac[:points]

    # -----------------------------------------------------------------
    # 协议拼接：这是关键修正部分
    # -----------------------------------------------------------------
    def _phase_string(self, points):
        phases = []

        for v in self.phase_vars:
            p = self._safe_int(v, 0)
            p = max(0, min(p, points - 1))
            phases.append(p)

        return "".join(f"{p:02X}" for p in phases)

    def _mode_command(self, board, op):
        """
        NI Trace 实测：
            0#M1
            0#M0
            0#M3

        不带 A40 和相位。
        """
        return f"{board}#M{op}"

    def _wave_header_command(self, board, points):
        """
        NI Trace 实测发送波形头：
            0#A40001020303F3F3F

        不是：
            0#M2A40001020303F3F3F
        """
        return f"{board}#A{points:02X}{self._phase_string(points)}"

    def _wave_data_lines(self, board, dac):
        points = len(dac)
        lines = []

        for i in range(points // 16):
            chunk = dac[i * 16:(i + 1) * 16]
            line = f"{board}#A{16 * i:02X}" + "".join(f"{int(v):03X}" for v in chunk)
            lines.append(line)

        return lines

    def _build_std_params_command(self):
        """
        构造 ## 标准参数，用于激活具体物理板输出。

        每通道：
            amp(4hex) + phase(4hex) + period(8hex)

        默认只打开 enable_mask 对应的通道。
        """
        board = self._safe_int(self.param_board_var, 1)
        enable_mask = self._safe_hex_byte(self.enable_mask_var.get(), 0x08)
        amp_word = self._safe_hex_word(self.drive_amp_hex_var.get(), 0x0400)
        period = max(1, self._safe_int(self.period_var, 1000))

        payload = ""

        for ch in range(8):
            if enable_mask & (1 << ch):
                amp = amp_word
            else:
                amp = 0

            phase = 0
            payload += f"{amp:04X}{phase:04X}{period:08X}"

        return f"{board}##{payload}"

    def _enable_command(self):
        board = self._safe_int(self.param_board_var, 1)
        enable_mask = self._safe_hex_byte(self.enable_mask_var.get(), 0x08)
        return f"{board}#E{enable_mask:02X}"

    def _disable_command(self):
        board = self._safe_int(self.param_board_var, 1)
        return f"{board}#E00"

    # -----------------------------------------------------------------
    # 预览
    # -----------------------------------------------------------------
    def plot_preview(self):
        try:
            self.ax.clear()

            points = self._get_padded_points()
            y = self._generate_raw_array(
                self.type_var.get(),
                self.formula_var.get(),
                points,
                self._safe_float(self.start_var, 0.0),
                self._safe_float(self.end_var, 1.0)
            )

            amp = max(0.0, min(1.0, self._safe_float(self.amp_var, 0.05)))
            y = np.asarray(y) * amp

            for i, pv in enumerate(self.phase_vars):
                p = max(0, min(self._safe_int(pv, 0), points - 1))
                self.ax.plot(
                    np.arange(points),
                    np.roll(y, -p),
                    lw=1,
                    label=f"CH{i+1}(+{p})"
                )

            self.ax.legend(loc="upper right", fontsize=7, ncol=4)
            self.ax.set_title("Arbitrary Waveform Preview")
            self.ax.grid(True)
            self.ax.set_ylim(-1.05, 1.05)
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("预览错误", f"公式解析失败:\n{e}")

    # -----------------------------------------------------------------
    # 串口读写
    # -----------------------------------------------------------------
    def _poll_reply(self, ser, timeout=1.0):
        """
        读回包。
        0# 广播时可能出现：
            1#A
            2# Timeout
            3# Timeout

        这里不强制判断成功，只尽量收全并打印。
        """
        end_time = time.time() + timeout
        buf = b""

        while time.time() < end_time:
            try:
                n = ser.in_waiting
            except Exception:
                n = 0

            if n:
                buf += ser.read(n)
                end_time = time.time() + 0.2

            time.sleep(0.02)

        return buf.decode("ascii", errors="replace").strip()

    def _send_named_lines_thread(self, lines, title="任意波形发送"):
        ser = self.main_app.controller.ser

        try:
            self.main_app.log(f"\n〰️ [{title}] 共 {len(lines)} 行")

            # 只在开始清一次，避免把上一条延迟 ACK 全部打乱
            try:
                ser.reset_input_buffer()
            except Exception:
                pass

            for idx, item in enumerate(lines):
                if isinstance(item, tuple):
                    tag, line, wait_s, read_timeout = item
                else:
                    tag, line = f"行{idx+1}", item
                    wait_s, read_timeout = 0.03, 1.0

                shown = line if len(line) <= 100 else line[:100] + "..."

                ser.write((line + "\r\n").encode("ascii"))
                ser.flush()

                time.sleep(wait_s)
                reply = self._poll_reply(ser, timeout=read_timeout)

                self.main_app.log(f"  -> [{tag}] {shown}")
                self.main_app.log(f"  <- [{tag}] {reply if reply else '<无回包>'}")

                # LabVIEW 数据行间隔约 10ms
                time.sleep(0.01)

            self.main_app.log(f"✅ [{title}] 完成")

        except Exception as e:
            self.main_app.log(f"❌ [{title}] 异常: {e}")

    def _start_send_thread(self, lines, title):
        ctrl = self.main_app.controller

        if not ctrl or not ctrl.ser or not ctrl.ser.is_open:
            messagebox.showerror("错误", "请先在主界面连接串口!")
            return

        threading.Thread(
            target=self._send_named_lines_thread,
            args=(lines, title),
            daemon=True
        ).start()

    # -----------------------------------------------------------------
    # 按钮动作
    # -----------------------------------------------------------------
    def send_waveform_only(self):
        """
        仅发送波形表：
            0#A40...
            0#A00...
            0#A10...
            ...
        不切 M1，不写 ##，不使能。
        """
        try:
            board = self._safe_int(self.m_board_var, 0)
            dac = self._make_dac()
            points = len(dac)

            lines = []

            lines.append((
                "波形头 0#A40...",
                self._wave_header_command(board, points),
                0.10,
                1.0
            ))

            for idx, line in enumerate(self._wave_data_lines(board, dac)):
                lines.append((
                    f"数据行 {idx+1}/{points//16}",
                    line,
                    0.03,
                    1.0
                ))

            self._start_send_thread(lines, "仅发送波形表")

        except Exception as e:
            messagebox.showerror("错误", f"发送波形表失败:\n{e}")

    def run_mode_op(self, op):
        """
        M0/M1/M3：
            0#M0
            0#M1
            0#M3
        """
        try:
            board = self._safe_int(self.m_board_var, 0)

            if op == 3:
                if not messagebox.askyesno(
                    "确认 M3",
                    "你之前实测 M3 可能会恢复正弦。\n确定要发送 M3 吗？"
                ):
                    return

            line = self._mode_command(board, op)
            title = self.OP_NAMES.get(op, f"M{op}")

            self._start_send_thread(
                [(title, line, 0.10, 1.0)],
                title
            )

        except Exception as e:
            messagebox.showerror("错误", f"M{op} 发送失败:\n{e}")

    def send_and_activate(self):
        """
        完整流程，按你已经测通的 NI Trace 协议：

        默认顺序：
            0#M1
            0#A40...
            0#A00...
            0#A10...
            0#A20...
            0#A30...
            1##
            1#E08

        不自动发 M3。
        """
        try:
            m_board = self._safe_int(self.m_board_var, 0)

            dac = self._make_dac()
            points = len(dac)

            wave_lines = []

            wave_lines.append((
                "波形头 0#A40...",
                self._wave_header_command(m_board, points),
                0.10,
                1.0
            ))

            for idx, line in enumerate(self._wave_data_lines(m_board, dac)):
                wave_lines.append((
                    f"数据行 {idx+1}/{points//16}",
                    line,
                    0.03,
                    1.0
                ))

            m1_line = (
                "M1 任意模式",
                self._mode_command(m_board, 1),
                0.10,
                1.0
            )

            if self.order_var.get().startswith("M1"):
                lines = [m1_line] + wave_lines
            else:
                lines = wave_lines + [m1_line]

            # 写 ## 和 E，激活物理板输出
            lines.append((
                "## 参数激活",
                self._build_std_params_command(),
                0.10,
                1.0
            ))

            lines.append((
                "E 使能",
                self._enable_command(),
                0.10,
                1.0
            ))

            self._start_send_thread(lines, "发送并激活任意波形")

        except Exception as e:
            messagebox.showerror("错误", f"发送并激活失败:\n{e}")

    def stop_output(self):
        try:
            lines = [
                ("关使能", self._disable_command(), 0.10, 1.0)
            ]
            self._start_send_thread(lines, "停止输出")
        except Exception as e:
            messagebox.showerror("错误", f"停止失败:\n{e}")

# =====================================================================
# 🌟 贝塞尔涡旋相控阵解算 (对应 MATLAB 版第 2~3 节)
# =====================================================================
def compute_vortex_array(target_x, target_y, topo_charge, max_amp,
                         wavelength=6.2, alpha=0.08, radius=20.0, num_sources=24,
                         compensate=True):
    """正 24 边形相控阵的偏心涡旋解算。

    波源取正多边形各条边的中点。对第 i 个波源：
        r  = |源 - 目标|
        θ  = atan2(源.y - 目标.y, 源.x - 目标.x)
        相位 = l·θ - k·r        —— l 为拓扑荷，k = 2π/λ
        振幅 ∝ exp(α·r)（compensate=True 时）

    按 exp(α·r) 加权再整体归一化，使最大的一路正好等于“允许的最大振幅”
    （这是基础设计，始终生效：远的路上损耗大所以发得更强，各路传到目标点
    时幅度才一致；涡旋在正中心时 24 路距离相同，自然等幅）。

    compensate 就是界面上的「开启反射补偿」——涡旋中心越偏离水槽中心，
    强驱动的波要横穿整个水槽、打到对面壁上的反射越强，所以整体压一档：
        amps *= exp(-α · |涡旋中心偏心距|)
    勾选时施加这个全局系数（偏心 0 时系数为 1，与不勾完全相同），
    不勾则完全不做这一步。它只等比缩放 24 路，不改变各路之间的比例。

    相位以 1 号板 CH1（阵元 1）为零点：先减掉阵元 1 的相位再取模，
    所以 CH1 恒为 0，沿阵列绕一圈相位累计正好 l·2π。整体平移所有阵元的
    相位只相当于给整个阵列加一个共同的时间原点，不改变涡旋本身，
    扣掉纯粹是为了让通道表好读。相位映射到 0~1（1 表示 2π），与硬件输入格式一致。

    返回 (sources[N,2] cm, amps[N] 0~1, phases[N] 0~1)。
    """
    n = int(num_sources)
    k = 2.0 * np.pi / float(wavelength)
    ang = np.linspace(0.0, 2.0 * np.pi, n + 1)
    verts = float(radius) * np.column_stack([np.cos(ang), np.sin(ang)])
    sources = (verts[:-1] + verts[1:]) / 2.0          # 边中点即波源

    dx = sources[:, 0] - float(target_x)
    dy = sources[:, 1] - float(target_y)
    dist = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)

    amp_raw = np.exp(float(alpha) * dist)
    phase_raw = float(topo_charge) * theta - k * dist

    peak = float(amp_raw.max())
    amps = amp_raw / peak * float(max_amp) if peak > 0 else np.zeros(n)
    if compensate:
        # 偏心越大整体压得越狠，抑制横穿水槽后打在对面壁上的反射
        amps = amps * math.exp(-float(alpha) * math.hypot(float(target_x), float(target_y)))
    phases = np.mod(phase_raw - phase_raw[0], 2.0 * np.pi) / (2.0 * np.pi)
    # 浮点误差会让本该是 0 的相位从下方绕成 0.99999999，吸回 0，别让界面显示成 1.0
    phases[np.isclose(phases, 1.0, atol=1e-9)] = 0.0
    return sources, amps, phases


def vortex_reflection_factor(target_x, target_y, alpha=0.08):
    """「开启反射补偿」施加的全局衰减系数 exp(-α·偏心距)。"""
    return math.exp(-float(alpha) * math.hypot(float(target_x), float(target_y)))


def _phase_to_color(phase01):
    """相位 0~1 → HSV 色轮，观感与 MATLAB 的 colormap(hsv) 一致。"""
    r, g, b = colorsys.hsv_to_rgb(float(phase01) % 1.0, 0.95, 0.95)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


# ---------------------------------------------------------------------
# 下面三个是水槽/水波的物理常数，来自 MATLAB 模型。
# 换水槽或换驱动频率时改这里即可，界面上不再暴露给用户。
# ---------------------------------------------------------------------
VORTEX_WAVELENGTH_CM = 6.2      # 水波波长 λ (cm)，对应 5 Hz 驱动
VORTEX_ALPHA = 0.08             # 传播衰减系数 α
VORTEX_RADIUS_CM = 20.0         # 正 24 边形外接圆半径 (cm)


class VortexArrayPanel(ttk.LabelFrame):
    """24 通道模式下的涡旋操控面板。

    拖画布上的绿球就是在挪涡旋中心，24 个阵元的幅度/相位实时重算。
    「发送参数」把参数填进通道矩阵并按原路下发（下发时是否做 LUT 补正，
    完全听顶部那个「全局启用 LUT 补正」的）。
    """

    CANVAS = 232          # 画布边长 (px)
    VIEW = 22.0           # 视野半宽 (cm)，与 MATLAB 的 xlim/ylim 对齐

    def __init__(self, parent, app):
        super().__init__(parent, text="贝塞尔涡旋操控 (24阵元)", padding=8)
        self.app = app
        self.sources = self.amps = self.phases = None

        self.x_var = tk.DoubleVar(value=0.0)
        self.y_var = tk.DoubleVar(value=0.0)
        self.l_var = tk.IntVar(value=1)
        self.maxamp_var = tk.DoubleVar(value=0.25)
        self.compensate_var = tk.BooleanVar(value=True)

        self._build_ui()
        for v in (self.x_var, self.y_var, self.l_var, self.maxamp_var, self.compensate_var):
            v.trace_add("write", lambda *_: self.refresh())
        self.refresh()

    # ---------------- UI ----------------
    def _build_ui(self):
        self.canvas = tk.Canvas(self, width=self.CANVAS, height=self.CANVAS,
                                bg="#101820", highlightthickness=1,
                                highlightbackground="#556")
        self.canvas.grid(row=0, column=0, columnspan=4, pady=(0, 6))
        self.canvas.bind("<Button-1>", self._on_drag)
        self.canvas.bind("<B1-Motion>", self._on_drag)

        def row(r, l1, v1, l2, v2):
            ttk.Label(self, text=l1).grid(row=r, column=0, sticky="e", padx=(0, 3), pady=2)
            ttk.Entry(self, textvariable=v1, width=8, justify="center").grid(row=r, column=1, sticky="w")
            ttk.Label(self, text=l2).grid(row=r, column=2, sticky="e", padx=(10, 3))
            ttk.Entry(self, textvariable=v2, width=8, justify="center").grid(row=r, column=3, sticky="w")

        row(1, "涡旋 X(cm):", self.x_var, "Y(cm):", self.y_var)
        row(2, "拓扑荷 l:", self.l_var, "最大振幅:", self.maxamp_var)

        ttk.Checkbutton(self, text="开启反射补偿", variable=self.compensate_var).grid(
            row=3, column=0, columnspan=4, sticky="w", pady=(8, 0))

        btns = ttk.Frame(self)
        btns.grid(row=4, column=0, columnspan=4, pady=(6, 2))
        ttk.Button(btns, text="📡 发送参数", command=self.send_to_boards).pack(side=tk.LEFT, padx=4)

        self.status = ttk.Label(self, text="", wraplength=self.CANVAS + 40,
                                justify="left", foreground="#006400")
        self.status.grid(row=5, column=0, columnspan=4, sticky="w", pady=(4, 0))

    # ---------------- 坐标换算 ----------------
    def _to_px(self, x, y):
        s = self.CANVAS / (2.0 * self.VIEW)
        return self.CANVAS / 2.0 + x * s, self.CANVAS / 2.0 - y * s

    def _to_cm(self, px, py):
        s = self.CANVAS / (2.0 * self.VIEW)
        return (px - self.CANVAS / 2.0) / s, (self.CANVAS / 2.0 - py) / s

    def _on_drag(self, event):
        x, y = self._to_cm(event.x, event.y)
        r_max = VORTEX_RADIUS_CM * 0.98
        r = math.hypot(x, y)
        if r > r_max > 0:                      # 小球不许跑出水槽
            x, y = x * r_max / r, y * r_max / r
        self.x_var.set(round(x, 2))
        self.y_var.set(round(y, 2))            # 触发 trace → refresh

    # ---------------- 解算与重绘 ----------------
    def _read_params(self):
        try:
            p = dict(target_x=float(self.x_var.get()), target_y=float(self.y_var.get()),
                     topo_charge=int(self.l_var.get()), max_amp=float(self.maxamp_var.get()))
        except (tk.TclError, ValueError):
            return None
        if not (0.0 <= p["max_amp"] <= 1.0):
            return None
        return p

    def refresh(self, *_args):
        p = self._read_params()
        if p is None:
            self.sources = self.amps = self.phases = None
            self.status.config(text="⚠️ 参数无效：最大振幅需在 0~1 之间", foreground="#B00000")
            return
        comp = bool(self.compensate_var.get())
        self.sources, self.amps, self.phases = compute_vortex_array(
            wavelength=VORTEX_WAVELENGTH_CM, alpha=VORTEX_ALPHA,
            radius=VORTEX_RADIUS_CM, compensate=comp, **p)
        self._redraw()
        tail = ""
        if comp:
            f = vortex_reflection_factor(p["target_x"], p["target_y"], VORTEX_ALPHA)
            tail = f" | 反射补偿 ×{f:.3f}"
        self.status.config(
            text=(f"l={p['topo_charge']} @({p['target_x']:.1f}, {p['target_y']:.1f})cm | "
                  f"振幅 {self.amps.min():.4f}~{self.amps.max():.4f}{tail}"),
            foreground="#006400")

    def _redraw(self):
        c = self.canvas
        c.delete("all")
        cx, cy = self._to_px(0, 0)
        c.create_line(0, cy, self.CANVAS, cy, fill="#2c3a48")
        c.create_line(cx, 0, cx, self.CANVAS, fill="#2c3a48")

        n = len(self.sources)
        ang = np.linspace(0.0, 2.0 * np.pi, n + 1)
        pts = []
        for a in ang:
            px, py = self._to_px(VORTEX_RADIUS_CM * math.cos(a), VORTEX_RADIUS_CM * math.sin(a))
            pts.extend([px, py])
        c.create_line(*pts, fill="#8fa6bb", width=2)

        for i, (sx, sy) in enumerate(self.sources):
            px, py = self._to_px(sx, sy)
            c.create_oval(px - 4, py - 4, px + 4, py + 4,
                          fill=_phase_to_color(self.phases[i]), outline="#000")

        tx, ty = self._to_px(float(self.x_var.get()), float(self.y_var.get()))
        c.create_oval(tx - 7, ty - 7, tx + 7, ty + 7, fill="#00e05a", outline="#000", width=2)
        c.create_text(6, 8, anchor="w", fill="#8fa6bb", font=("", 8),
                      text="拖动绿球移动涡旋中心 / 颜色=相位")

    # ---------------- 写入通道 ----------------
    def apply_to_channels(self, quiet=False):
        """把 24 路幅度/相位填进通道矩阵。阵元 i → 第 (i//8+1) 块板的 CH(i%8+1)。"""
        if self.amps is None:
            if not quiet:
                messagebox.showwarning("提示", "当前参数无效，先把输入框改对。")
            return False
        if len(self.app.amp_vars) < len(self.amps):
            if not quiet:
                messagebox.showwarning("模式限制", "请先切到「24通道 (所有板子)」再操作涡旋参数。")
            return False

        for i in range(len(self.amps)):
            self.app.amp_vars[i].set(round(float(self.amps[i]), 4))
            self.app.phase_vars[i].set(round(float(self.phases[i]), 4))
        return True

    def send_to_boards(self):
        """写入通道矩阵并立即下发。走的就是「执行 → 写入波形参数」那条路，
        所以 LUT 补正照旧由顶部「全局启用 LUT 补正」勾选框决定。"""
        app = self.app
        if not self.apply_to_channels():
            return

        app.log(f"🌀 涡旋参数已写入 24 通道：l={int(self.l_var.get())}, "
                f"中心=({float(self.x_var.get()):.1f}, {float(self.y_var.get()):.1f})cm, "
                f"最大振幅={float(self.maxamp_var.get()):.4f}, "
                f"反射补偿={'开 ×%.3f' % vortex_reflection_factor(self.x_var.get(), self.y_var.get(), VORTEX_ALPHA) if self.compensate_var.get() else '关'}")
        if not all(v.get() for v in app.enables_vars):
            app.log("   ⚠️ 有通道使能没勾上，那几路不会输出。")

        if not app.controller:
            messagebox.showinfo("提示", "串口未连接：参数已写入通道矩阵，连上串口后再点【发送参数】即可下发。")
            return

        prev_op = app.op_var.get()
        app.op_var.set("1: 写入波形参数")
        try:
            app.execute_operation()
        finally:
            app.op_var.set(prev_op)


# =====================================================================
# 主界面类
# =====================================================================
class LabviewMimicGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("水波阵列控制面板")
        self.root.geometry("1250x920")

        self.controller = None
        self.camera = None
        self.config_file = os.path.join(os.path.dirname(__file__), "hardware_param_cache.json")

        self.calib_vars = [tk.StringVar() for _ in range(3)]
        self.lut_data_list = [None, None, None]
        self._cancel_flag = False

        # 🌟 新增：实时渲染相关
        self.live_renderer = None
        self.live_process_thread = None
        self.fcd_core = None
        self._demo_running = False
        self.vortex_panel = None

        self.setup_ui()
        self.load_hardware_config()
        self.scan_ports()

    def setup_ui(self):
        # ================= 🌟 可滚动主容器 =================
        # 24 通道模式下参数矩阵有 3 块板子，屏幕再高也塞不下，
        # 所以整个界面套一层 Canvas + 竖直滚动条，鼠标滚轮直接滚。
        outer = ttk.Frame(self.root)
        outer.pack(fill=tk.BOTH, expand=True)
        self.main_canvas = tk.Canvas(outer, highlightthickness=0)
        vbar = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=self.main_canvas.yview)
        hbar = ttk.Scrollbar(outer, orient=tk.HORIZONTAL, command=self.main_canvas.xview)
        self.main_canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scroll_frame = ttk.Frame(self.main_canvas)
        self._scroll_window = self.main_canvas.create_window(
            (0, 0), window=self.scroll_frame, anchor="nw")
        self.scroll_frame.bind(
            "<Configure>",
            lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all")))
        # 窗口比内容窄时不挤压内容，交给横向滚动条去够
        self.main_canvas.bind(
            "<Configure>",
            lambda e: self.main_canvas.itemconfig(
                self._scroll_window, width=max(e.width, self.scroll_frame.winfo_reqwidth())))
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.root.bind_all(seq, self._on_mousewheel)

        # ================= 顶部控制区 =================
        top_frame = ttk.LabelFrame(self.scroll_frame, text="系统与操作控制", padding=10)
        top_frame.pack(fill=tk.X, padx=10, pady=5)

        # ---- 第0行：串口、连接、板子、操作 ----
        ttk.Label(top_frame, text="串口号:").grid(row=0, column=0, padx=5, pady=5)
        self.port_var = tk.StringVar()
        port_frame = ttk.Frame(top_frame)
        port_frame.grid(row=0, column=1, padx=5)
        self.port_cb = ttk.Combobox(port_frame, textvariable=self.port_var, width=8, state="readonly")
        self.port_cb.pack(side=tk.LEFT)
        self.refresh_btn = ttk.Button(port_frame, text="🔄", width=3, command=self.scan_ports)
        self.refresh_btn.pack(side=tk.LEFT, padx=(2, 0))

        conn_frame = ttk.Frame(top_frame)
        conn_frame.grid(row=0, column=2, padx=0)
        self.connect_btn = ttk.Button(conn_frame, text="🔌连接串口", command=self.toggle_connection)
        self.connect_btn.pack(side=tk.LEFT, padx=2)
        self.connect_cam_btn = ttk.Button(conn_frame, text="📸连接相机", command=self.toggle_camera)
        self.connect_cam_btn.pack(side=tk.LEFT, padx=2)
        self.tracker_btn = ttk.Button(conn_frame, text="🔄 测角速度", command=self.open_rotation_tracker)
        self.tracker_btn.pack(side=tk.LEFT, padx=2)

        ttk.Separator(top_frame, orient=tk.VERTICAL).grid(row=0, column=3, sticky="ns", padx=10)

        ttk.Label(top_frame, text="目标板子:").grid(row=0, column=4, padx=5)
        self.board_var = tk.StringVar(value="1")
        board_cb = ttk.Combobox(top_frame, textvariable=self.board_var,
                                values=["1", "2", "3", "0 (所有)", "24通道 (1-3板)"], width=14, state="readonly")
        board_cb.grid(row=0, column=5, padx=5)
        board_cb.bind("<<ComboboxSelected>>", lambda e: self.build_param_matrix())

        ttk.Label(top_frame, text="操作类型:").grid(row=0, column=6, padx=5)
        self.op_var = tk.StringVar(value="1: 写入波形参数")
        op_values = ["0: 测试连接", "1: 写入波形参数", "2: 读取波形参数",
                     "3: 写入通道使能", "4: 保存配置", "5: 设备复位"]
        op_cb = ttk.Combobox(top_frame, textvariable=self.op_var, values=op_values, width=16, state="readonly")
        op_cb.grid(row=0, column=7, padx=5)

        self.execute_btn = ttk.Button(top_frame, text="▶ 执行", command=self.execute_operation, state=tk.DISABLED)
        self.execute_btn.grid(row=0, column=8, padx=5)
        self.stop_btn = ttk.Button(top_frame, text="🛑 一键停止", command=self.stop_all_speakers, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=9, padx=10)

        # ---- 第1行左侧：功能按钮区 (pack布局，不会重叠) ----
        btn_row_frame = ttk.Frame(top_frame)
        btn_row_frame.grid(row=1, column=0, columnspan=4, sticky='w', pady=5)
        ttk.Button(btn_row_frame, text="〰️ 任意波形发生器",
                   command=self.open_arb_waveform).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_row_frame, text="🔍 预览LUT修正参数",
                   command=self.preview_lut_correction).pack(side=tk.LEFT, padx=5)
        # 🌟 新增：实时渲染按钮
        self.live_render_btn = ttk.Button(btn_row_frame, text="🎬 启动实时渲染",
                                           command=self.toggle_live_render)
        self.live_render_btn.pack(side=tk.LEFT, padx=5)

        # ---- 第1行右侧：延迟/时长 ----
        ttk.Label(top_frame, text="延迟触发(秒):").grid(row=1, column=4, padx=5, pady=2, sticky='e')
        self.delay_var = tk.DoubleVar(value=0.0)
        ttk.Entry(top_frame, textvariable=self.delay_var, width=8).grid(row=1, column=5, padx=5, sticky='w')
        ttk.Label(top_frame, text="工作时长(秒):").grid(row=1, column=6, padx=5, pady=2, sticky='e')
        self.duration_var = tk.DoubleVar(value=0.0)
        ttk.Entry(top_frame, textvariable=self.duration_var, width=8).grid(row=1, column=7, padx=5, sticky='w')
        ttk.Label(top_frame, text="(0为持续)", font=("", 8), foreground="dimgray").grid(row=1, column=8, sticky='w')

        ttk.Separator(top_frame, orient=tk.HORIZONTAL).grid(row=2, column=0, columnspan=10, sticky="ew", pady=5)

        # ---- 第3~5行：LUT 文件加载槽 ----
        for b in range(3):
            ttk.Label(top_frame, text=f"板{b+1} LUT定标:").grid(row=3+b, column=0, padx=5, pady=2, sticky='e')
            ttk.Entry(top_frame, textvariable=self.calib_vars[b], width=55, state="readonly").grid(
                row=3+b, column=1, columnspan=5, padx=5, sticky='w')
            ttk.Button(top_frame, text=f"📂 加载 板{b+1} 曲线",
                       command=lambda idx=b: self.load_calibration_file(idx)).grid(
                row=3+b, column=6, columnspan=2, padx=5, sticky='w')

        self.use_calib_var = tk.BooleanVar(value=True)
        self.use_calib_chk = ttk.Checkbutton(top_frame, text="全局启用 LUT 补正",
                                              variable=self.use_calib_var, command=self.save_hardware_config)
        self.use_calib_chk.grid(row=4, column=8, columnspan=2, padx=10, sticky='w')

        ttk.Separator(top_frame, orient=tk.HORIZONTAL).grid(row=6, column=0, columnspan=10, sticky="ew", pady=5)

        # ---- 第7行：全局自动定标与曝光控制 ----
        ttk.Label(top_frame, text="自动定标系统:").grid(row=7, column=0, padx=5, pady=5, sticky='e')
        ttk.Label(top_frame, text="定标喇叭数:").grid(row=7, column=1, sticky='e')
        self.cal_spk_num_var = tk.IntVar(value=8)
        ttk.Entry(top_frame, textvariable=self.cal_spk_num_var, width=6,
                  justify="center").grid(row=7, column=2, sticky='w', padx=(2, 15))
        ttk.Label(top_frame, text="曝光(ms, 0=自动):").grid(row=7, column=3, sticky='e', pady=5)
        self.cam_exp_var = tk.DoubleVar(value=10.0)
        ttk.Entry(top_frame, textvariable=self.cam_exp_var, width=6).grid(row=7, column=4, sticky='w')
        ttk.Label(top_frame, text="相机FPS:").grid(row=7, column=5, sticky='e')
        self.cam_fps_var = tk.DoubleVar(value=30.0)
        ttk.Entry(top_frame, textvariable=self.cam_fps_var, width=6).grid(row=7, column=6, sticky='w')
        self.carousel_btn = ttk.Button(top_frame, text="🔊 启动自动定标流程",
                                        command=self.start_carousel, state=tk.DISABLED)
        self.carousel_btn.grid(row=7, column=8, columnspan=2, padx=5, sticky='w', ipadx=10)

        # ---- 第8行：波形全局参数 ----
        ttk.Label(top_frame, text="波形全局参数:").grid(row=8, column=0, padx=5, pady=5, sticky='e')
        ttk.Label(top_frame, text="全局周期(ms):").grid(row=8, column=1, sticky='e')
        self.global_period_var = tk.IntVar(value=150)
        ttk.Entry(top_frame, textvariable=self.global_period_var, width=8).grid(row=8, column=2, sticky='w')
        ttk.Label(top_frame, text="全局相位(0-1):").grid(row=8, column=3, sticky='e')
        self.global_phase_var = tk.DoubleVar(value=0.0)
        ttk.Entry(top_frame, textvariable=self.global_phase_var, width=8).grid(row=8, column=4, sticky='w')

        # ---- 第9行：图像保存 ----
        ttk.Label(top_frame, text="图像保存目录:").grid(row=9, column=0, padx=5, pady=5, sticky='e')
        self.save_dir_var = tk.StringVar()
        ttk.Entry(top_frame, textvariable=self.save_dir_var, width=55, state="readonly").grid(
            row=9, column=1, columnspan=5, padx=5, sticky='w')
        self.save_dir_btn = ttk.Button(top_frame, text="📂 浏览目录", command=self.browse_save_dir)
        self.save_dir_btn.grid(row=9, column=6, padx=5, sticky='w')
        self.snap_btn = ttk.Button(top_frame, text="📸 采集单张图片 (.tiff)", command=self.capture_single_frame)
        self.snap_btn.grid(row=9, column=7, columnspan=3, padx=5, sticky='w')

        # ---- 第10行：序列采集 ----
        ttk.Label(top_frame, text="序列采集帧数:").grid(row=10, column=0, padx=5, pady=5, sticky='e')
        self.seq_frames_var = tk.IntVar(value=100)
        ttk.Entry(top_frame, textvariable=self.seq_frames_var, width=8).grid(row=10, column=1, sticky='w')
        self.seq_snap_btn = ttk.Button(top_frame, text="🎥 高速采集图像序列", command=self.start_capture_sequence)
        self.seq_snap_btn.grid(row=10, column=2, columnspan=3, padx=5, sticky='w')

        # 🌟 新增 ---- 第11行：FCD 参考图路径 (实时渲染用) ----
        ttk.Label(top_frame, text="FCD参考图:").grid(row=11, column=0, padx=5, pady=5, sticky='e')
        self.ref_img_var = tk.StringVar()
        ttk.Entry(top_frame, textvariable=self.ref_img_var, width=55, state="readonly").grid(
            row=11, column=1, columnspan=5, padx=5, sticky='w')
        ttk.Button(top_frame, text="📂 选择参考图", command=self.browse_ref_image).grid(
            row=11, column=6, padx=5, sticky='w')
        ttk.Label(top_frame, text="水深(mm):").grid(row=11, column=7, sticky='e')
        self.fcd_depth_var = tk.DoubleVar(value=30.0)
        ttk.Entry(top_frame, textvariable=self.fcd_depth_var, width=6).grid(row=11, column=8, sticky='w')


        # 🌟 新增 ---- 第12行：FCD 光路参数 (光栅距 / 等效H 随水深实时换算) ----
        ttk.Label(top_frame, text="FCD光路参数:").grid(row=12, column=0, padx=5, pady=5, sticky='e')
        optics_frame = ttk.Frame(top_frame)
        optics_frame.grid(row=12, column=1, columnspan=9, padx=5, sticky='w')

        ttk.Label(optics_frame, text="底板厚(mm):").pack(side=tk.LEFT)
        self.fcd_plate_var = tk.DoubleVar(value=10.0)
        ttk.Entry(optics_frame, textvariable=self.fcd_plate_var, width=6).pack(side=tk.LEFT, padx=(2, 10))

        ttk.Label(optics_frame, text="底板折射率:").pack(side=tk.LEFT)
        self.fcd_n_plate_var = tk.DoubleVar(value=1.49)
        ttk.Entry(optics_frame, textvariable=self.fcd_n_plate_var, width=6).pack(side=tk.LEFT, padx=(2, 10))

        ttk.Label(optics_frame, text="底板下气隙(mm):").pack(side=tk.LEFT)
        self.fcd_gap_var = tk.DoubleVar(value=0.0)
        ttk.Entry(optics_frame, textvariable=self.fcd_gap_var, width=6).pack(side=tk.LEFT, padx=(2, 10))

        ttk.Label(optics_frame, text="水折射率:").pack(side=tk.LEFT)
        self.fcd_n_water_var = tk.DoubleVar(value=1.333)
        ttk.Entry(optics_frame, textvariable=self.fcd_n_water_var, width=6).pack(side=tk.LEFT, padx=(2, 12))

        self.fcd_optics_label = ttk.Label(optics_frame, text="", foreground="#006400")
        self.fcd_optics_label.pack(side=tk.LEFT)

        for _var in (self.fcd_depth_var, self.fcd_plate_var, self.fcd_n_plate_var,
                     self.fcd_gap_var, self.fcd_n_water_var):
            _var.trace_add("write", self._on_fcd_optics_changed)
        self._on_fcd_optics_changed()

        # 🌟 新增 ---- 第13行：实时渲染区域 (自己在参考图上框) ----
        ttk.Label(top_frame, text="实时渲染区域:").grid(row=13, column=0, padx=5, pady=5, sticky='e')
        crop_frame = ttk.Frame(top_frame)
        crop_frame.grid(row=13, column=1, columnspan=9, padx=5, sticky='w')

        ttk.Button(crop_frame, text="🖱 在参考图上框选",
                   command=self.select_live_crop).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(crop_frame, text="Crop(X1,X2,Y1,Y2):").pack(side=tk.LEFT)
        self.live_crop_vars = [tk.IntVar(value=0) for _ in range(4)]
        for _cv in self.live_crop_vars:
            ttk.Entry(crop_frame, textvariable=_cv, width=6).pack(side=tk.LEFT, padx=1)

        ttk.Button(crop_frame, text="清除", width=6,
                   command=self.clear_live_crop).pack(side=tk.LEFT, padx=(6, 12))

        ttk.Label(crop_frame, text="未框选时用中心ROI(px):").pack(side=tk.LEFT)
        self.live_roi_var = tk.IntVar(value=512)
        ttk.Entry(crop_frame, textvariable=self.live_roi_var, width=6).pack(side=tk.LEFT, padx=2)

        self.live_crop_label = ttk.Label(crop_frame, text="", foreground="#006400")
        self.live_crop_label.pack(side=tk.LEFT, padx=10)
        for _cv in self.live_crop_vars:
            _cv.trace_add("write", self._on_live_crop_changed)
        self._on_live_crop_changed()

        # ================= 动态参数矩阵区容器 =================
        self.param_container = ttk.Frame(self.scroll_frame)
        self.param_container.pack(fill=tk.X, padx=10, pady=5)
        self.build_param_matrix()

        # ================= 日志显示区 =================
        log_frame = ttk.LabelFrame(self.scroll_frame, text="操作日志与串口返回", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=8, bg="#f4f4f4")
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log("界面初始化完成。请先连接串口与相机。")

    # =================================================================
    # 打开独立面板
    # =================================================================
    def _on_mousewheel(self, event):
        """滚轮滚主画布；鼠标停在日志框那类自带滚动的控件上时让给它自己滚。"""
        try:
            w = self.root.winfo_containing(event.x_root, event.y_root)
        except Exception:
            w = None
        if w is None:
            return
        try:
            if w.winfo_toplevel() is not self.root:
                return            # 鼠标在弹窗上，别去滚后面的主界面
        except Exception:
            return
        while w is not None:
            if isinstance(w, (tk.Text, tk.Listbox)):
                return
            w = getattr(w, "master", None)
        if event.num == 4:
            step = -3
        elif event.num == 5:
            step = 3
        else:
            step = -3 if event.delta > 0 else 3
        try:
            self.main_canvas.yview_scroll(step, "units")
        except tk.TclError:
            pass

    def open_arb_waveform(self):
        ArbitraryWaveformDialog(self.root, self)

    def open_rotation_tracker(self):
        if RotationTrackerUI is None:
            messagebox.showerror("错误", "找不到 rotation_tracker.py 模块！")
            return
        if not self.camera or not self.camera.is_opened:
            messagebox.showerror("错误", "请先连接工业相机！测角速度需要相机数据流。")
            return
        self.tracker_ui = RotationTrackerUI(camera=self.camera)
        self.log("🔄 已打开角速度测量面板。")

    def preview_lut_correction(self):
        mode = self.board_var.get()
        if "24通道" in mode or "所有" in mode:
            messagebox.showinfo("提示", "请先在右上角选择具体的单块板子 (1, 2, 3) 进行预览。")
            return
        board_idx = int(mode)
        raw_params = []
        for i in range(8):
            amp = self.amp_vars[i].get() if self.enables_vars[i].get() else 0.0
            phase = self.phase_vars[i].get()
            raw_params.append({'amp': amp, 'phase': phase, 'period': 150})
        calibrated = self._apply_lut_calibration(raw_params, board_idx)

        preview_win = tk.Toplevel(self.root)
        preview_win.title(f"板 {board_idx} - LUT 修正参数对比预览")
        preview_win.geometry("600x300")
        preview_win.attributes("-topmost", True)

        tree = ttk.Treeview(preview_win,
                            columns=("CH", "OrigAmp", "LUTAmp", "OrigPhase", "LUTPhase"),
                            show="headings")
        tree.heading("CH", text="通道")
        tree.heading("OrigAmp", text="主界面输入幅度")
        tree.heading("LUTAmp", text="LUT修正后驱动电压")
        tree.heading("OrigPhase", text="主界面输入相位")
        tree.heading("LUTPhase", text="LUT修正后相位")
        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        for i in range(8):
            if raw_params[i]['amp'] > 0:
                tree.insert("", tk.END, values=(
                    f"CH {i+1}",
                    f"{raw_params[i]['amp']:.3f}",
                    f"{calibrated[i]['amp']:.3f}",
                    f"{raw_params[i]['phase']:.3f}",
                    f"{calibrated[i]['phase']:.3f}"))

    # 🌟 新增：FCD参考图选择
    def browse_ref_image(self):
        file_path = filedialog.askopenfilename(
            title="选择FCD参考图像 (未变形的光栅图)",
            filetypes=[("Image Files", "*.tiff *.tif *.png *.bmp *.jpg")])
        if file_path:
            self.ref_img_var.set(file_path)
            self.save_hardware_config()
            self.log(f"📂 FCD参考图已设置: {file_path}")

    # =================================================================
    # 🌟 FCD 光路换算：光栅距与等效 H 都随水深走，不是写死的常数
    #
    # 自由面斜率 ∇h 造成的背景图案视位移为 δr = H_eff · ∇h，其中
    #     H_eff = (1 - n_air/n_water) · [ h_水 + t_底板·(n_water/n_底板)
    #                                            + d_气隙·(n_water/n_air) ]
    # 方括号内是把底板、气隙都折算成“等效水层”之后的光栅距，
    # 括号外的 (1 - n_air/n_water) 是自由面折射带来的偏折系数。
    # 真实光栅距（几何距离）则是 h_水 + t_底板 + d_气隙。
    #
    # 用默认值 (底板 10mm / n=1.49 / 无气隙 / n_water=1.333) 代入，就还原成
    # fcd_backend 内部写死的 H = (水深 + 0.894×10) × 0.25 —— 后端把 n水/n板=0.8946
    # 和 1-n空/n水=0.24981 取整成了 0.894 / 0.25，两者相差 <0.1%，
    # 也就是说实时渲染与 fcd_gui 离线分析用的是同一把尺子。
    # =================================================================
    N_AIR = 1.0

    def get_fcd_optics(self):
        """按当前界面参数换算，返回 (水深, 真实光栅距, 等效H)，单位 mm；参数非法时返回 (None, None, None)。"""
        try:
            depth = float(self.fcd_depth_var.get())
            plate = float(self.fcd_plate_var.get())
            n_plate = float(self.fcd_n_plate_var.get())
            gap = float(self.fcd_gap_var.get())
            n_water = float(self.fcd_n_water_var.get())
        except (tk.TclError, ValueError):
            return None, None, None

        if depth < 0 or plate < 0 or gap < 0 or n_plate <= 0 or n_water <= self.N_AIR:
            return None, None, None

        geometric = depth + plate + gap
        equivalent = depth + plate * (n_water / n_plate) + gap * (n_water / self.N_AIR)
        H_eff = equivalent * (1.0 - self.N_AIR / n_water)
        if H_eff <= 0:
            return None, None, None
        return depth, geometric, H_eff

    def _on_fcd_optics_changed(self, *_args):
        """水深/光路参数一改动就刷新显示；若渲染正在跑，H 同步热更新。"""
        label = getattr(self, "fcd_optics_label", None)
        if label is None:
            return

        depth, geometric, H_eff = self.get_fcd_optics()
        if H_eff is None:
            label.config(text="⚠️ 光路参数无效 (水深/厚度需≥0，折射率需 n水>1)",
                         foreground="#B00000")
            return

        label.config(text=f"→ 真实光栅距 {geometric:.2f} mm，等效 H {H_eff:.3f} mm",
                     foreground="#006400")

        # 渲染进行中改水深时，直接把新的 H 推给正在算的核心，无需重启渲染
        if getattr(self, "fcd_core", None) is not None:
            self.fcd_core.H = H_eff

    # =================================================================
    # 🌟 实时渲染区域：在参考图上自己框
    # 复用 FCD 后端的 find_pixels() 交互选框（fcd_gui.py 的「获取像素坐标」
    # 走的是同一个函数），所以这里框出来的坐标和离线分析里的 Crop 完全通用，
    # 可以直接互相抄。框选结果写进 4 个 Crop 输入框，也可以手动改。
    # =================================================================
    def get_live_crop(self):
        """返回 (x1, x2, y1, y2)；没框选或输入非法时返回 (0, 0, 0, 0) 表示未指定。"""
        try:
            x1, x2, y1, y2 = (int(v.get()) for v in self.live_crop_vars)
        except (tk.TclError, ValueError):
            return (0, 0, 0, 0)
        if x2 <= x1 or y2 <= y1 or min(x1, y1) < 0:
            return (0, 0, 0, 0)
        return (x1, x2, y1, y2)

    def _on_live_crop_changed(self, *_args):
        label = getattr(self, "live_crop_label", None)
        if label is None:
            return
        x1, x2, y1, y2 = self.get_live_crop()
        if x2 > x1:
            label.config(text=f"→ 已框选 {x2 - x1}×{y2 - y1} px", foreground="#006400")
        else:
            label.config(text="→ 未框选，按中心ROI裁切", foreground="#808080")

    def clear_live_crop(self):
        for cv in self.live_crop_vars:
            cv.set(0)
        self.save_hardware_config()
        self.log("🔲 已清除框选区域，实时渲染将回到中心 ROI 裁切。")

    def select_live_crop(self):
        """在参考图上拉框选出实时渲染区域（单击定起点 → 移动 → 再次单击确认）。"""
        ref_path = self.ref_img_var.get()
        if not ref_path or not os.path.exists(ref_path):
            messagebox.showerror("错误", "请先选择有效的 FCD 参考图像，再框选渲染区域！")
            return

        core_cls, core_module, errors = load_fcd_core()
        if core_cls is None:
            messagebox.showerror("错误", "找不到可用的 FCD 解调核心，无法框选：\n" + "\n".join(errors))
            return

        try:
            out_dir = self.save_dir_var.get()
            if not out_dir or not os.path.isdir(out_dir):
                out_dir = os.path.dirname(__file__)
            desired = {"ref_path": ref_path, "out_dir": out_dir, "crop_pixels": (0, 0, 0, 0)}
            params = inspect.signature(core_cls.__init__).parameters
            temp_core = core_cls(**{k: v for k, v in desired.items() if k in params})
            temp_core.crop = (0, 0, 0, 0)      # 必须在整幅原图上框

            self.log("🖱 框选窗口已打开：单击定起点 → 移动拉出红色虚线方框 → 再次单击确认 (Esc/回车放弃)")
            self.root.update_idletasks()
            pts, _ = temp_core.find_pixels()
        except Exception as e:
            self.log(f"❌ 框选失败: {e}\n{traceback.format_exc()}")
            messagebox.showerror("错误", f"框选窗口打开失败：{e}")
            return

        if len(pts) != 2:
            self.log("ℹ️ 未确认任何区域，保持原设置。")
            return

        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        for cv, val in zip(self.live_crop_vars, (x1, x2, y1, y2)):
            cv.set(int(val))
        self.save_hardware_config()

        self.log(f"✅ 渲染区域已框选: X {x1}-{x2}, Y {y1}-{y2} ({x2 - x1}×{y2 - y1} px)")
        if (x2 - x1) > 1024:
            self.log("⚠️ 框选区域偏大，CPU 解调会明显掉帧，建议控制在 384~768 px。")
        if self.live_process_thread and self.live_process_thread.is_running:
            self.log("ℹ️ 新区域要重启实时渲染才生效（先停止再启动）。")

    # =================================================================
    # 🌟 新增：实时FCD渲染
    # =================================================================
    def toggle_live_render(self):
        if not LIVE_RENDER_AVAILABLE:
            messagebox.showerror("错误", "找不到 live_renderer.py！请确保它与此脚本在同一目录下。")
            return

        # 如果正在运行，则停止
        if self.live_renderer and self.live_renderer.is_running:
            self.stop_live_render()
            return

        # 检查相机
        if not self.camera or not self.camera.is_opened:
            result = messagebox.askyesno("提示",
                "相机未连接！\n\n"
                "• 点击「是」→ 使用模拟测试数据启动渲染预览\n"
                "• 点击「否」→ 取消\n\n"
                "模拟模式下可以测试渲染窗口的各种显示模式。")
            if not result:
                return
            self.start_live_render_demo()
            return

        # 检查参考图
        ref_path = self.ref_img_var.get()
        if not ref_path or not os.path.exists(ref_path):
            messagebox.showerror("错误",
                "请先在主界面选择一个有效的FCD参考图像！\n"
                "(未变形状态下拍摄的光栅图)")
            return

        # 检查FCD核心模块（实现在 fcd_backend_gpu.py / fcd_backend_syl.py）
        core_cls, core_module, errors = load_fcd_core()
        if core_cls is None:
            self.log("⚠️ FCD 解调核心导入失败：\n    " + "\n    ".join(errors))
            messagebox.showwarning("提示",
                "找不到可用的 FCD 解调核心！\n"
                "请确认 fcd_backend_syl.py (CPU版) 或 fcd_backend_gpu.py (GPU版)\n"
                "与本脚本在同一目录下。\n\n"
                "导入失败详情：\n" + "\n".join(errors) + "\n\n"
                "将使用模拟数据模式启动渲染预览。")
            self.start_live_render_demo()
            return

        self._fcd_core_cls = core_cls
        self._fcd_core_module = core_module
        self.start_live_render_full()

    def start_live_render_full(self):
        """完整模式：相机 + FCD解调 + 实时渲染"""
        try:
            core_cls = getattr(self, "_fcd_core_cls", None)
            core_module = getattr(self, "_fcd_core_module", None)
            if core_cls is None:
                core_cls, core_module, errors = load_fcd_core()
                if core_cls is None:
                    raise ImportError("无可用的 FCD 解调核心：" + "; ".join(errors))
                self._fcd_core_cls, self._fcd_core_module = core_cls, core_module

            ref_path = self.ref_img_var.get()
            depth, geometric, H_eff = self.get_fcd_optics()
            if H_eff is None:
                raise ValueError("FCD 光路参数无效，请检查水深 / 底板厚 / 气隙 / 折射率的输入。")

            out_dir = self.save_dir_var.get()
            if not out_dir or not os.path.isdir(out_dir):
                out_dir = os.path.dirname(__file__)

            # 两个后端的 __init__ 签名不完全一致 (例如只有 CPU 版有 fps)，
            # 所以先按签名过滤掉后端不认识的参数再构造。
            # water_depth 照常传给后端保持属性自洽，但 self.H 一律以界面换算的
            # H_eff 为准——这样底板厚度/折射率/气隙被改过时也不会和界面显示脱节。
            crop = self.get_live_crop()          # 框选出来的区域，(0,0,0,0) 表示没框
            desired = {
                "ref_path": ref_path,
                "out_dir": out_dir,
                "crop_pixels": crop,
                "water_depth": depth,
            }
            params = inspect.signature(core_cls.__init__).parameters
            kwargs = {k: v for k, v in desired.items() if k in params}
            self.fcd_core = core_cls(**kwargs)
            self.fcd_core.H = H_eff

            try:
                roi = int(self.live_roi_var.get())
            except (tk.TclError, ValueError):
                roi = 512
            roi = max(0, roi)

            if crop[1] > crop[0]:
                region = f"框选区 X{crop[0]}-{crop[1]} Y{crop[2]}-{crop[3]} ({crop[1]-crop[0]}×{crop[3]-crop[2]})"
            else:
                region = f"中心ROI {roi}px" if roi else "整幅"
            self.log(f"✅ FCD核心已初始化 [{core_module}] (水深={depth:g}mm, "
                     f"真实光栅距={geometric:g}mm, 等效H={H_eff:.3f}mm, "
                     f"解调区域={region}, 参考图={os.path.basename(ref_path)})")
            if crop[1] <= crop[0] and roi == 0:
                self.log("⚠️ 没框选区域且 ROI=0，将对整幅图解调，5MP 画面下每帧要数秒，延迟会非常大。")
            elif max(crop[1] - crop[0], roi) > 1024:
                self.log("⚠️ 解调区域超过 1024px 时 CPU 会明显掉帧，建议 384~768。")

            # 锁相解调要知道喇叭的驱动周期：以通道矩阵里实际下发的那个为准
            try:
                period_ms = float(self.period_vars[0].get())
            except (IndexError, tk.TclError, ValueError):
                period_ms = float(self.global_period_var.get())
            if period_ms <= 0:
                period_ms = 150.0
            self.log(f"🔒 实时相位按锁相解调，驱动周期={period_ms:g}ms "
                     f"({1000.0/period_ms:.2f}Hz)；若改了通道周期需重启实时渲染。")

            self.live_renderer = LiveRenderer(fps=15, window_name="FCD Live Monitor")
            self.live_process_thread = LiveProcessThread(
                camera=self.camera, fcd_core=self.fcd_core,
                renderer=self.live_renderer, fps=15, roi_size=roi,
                drive_period_ms=period_ms)

            self.live_renderer.start()
            self.live_process_thread.start()

            self.live_render_btn.config(text="🛑 停止实时渲染")
            self.log("🎬 实时FCD渲染已启动！\n"
                     "   快捷键: m=切换模式 a=自动范围 +/-=调范围 空格=暂停 q=退出")
        except Exception as e:
            self.log(f"❌ 启动实时渲染失败: {e}\n{traceback.format_exc()}")
            self.stop_live_render()

    def start_live_render_demo(self):
        """演示模式：无相机，使用模拟数据"""
        try:
            self.live_renderer = LiveRenderer(fps=15, window_name="FCD Live Monitor (Demo)")
            self.live_renderer.start()

            self._demo_running = True
            threading.Thread(target=self._demo_data_loop, daemon=True).start()

            self.live_render_btn.config(text="🛑 停止实时渲染")
            self.log("🎬 实时渲染已启动 (模拟数据模式)！\n"
                     "   快捷键: m=切换模式 a=自动范围 +/-=调范围 空格=暂停 q=退出")
        except Exception as e:
            self.log(f"❌ 启动演示渲染失败: {e}")
            self.stop_live_render()

    def _demo_data_loop(self):
        """生成动态模拟数据供渲染器显示"""
        size = 200
        X, Y = np.meshgrid(
            np.linspace(-np.pi, np.pi, size),
            np.linspace(-np.pi, np.pi, size))
        t = 0
        while self._demo_running and self.live_renderer and self.live_renderer.is_running:
            t += 0.1
            h = np.sin(X + t) * np.cos(Y + t * 0.7) * 0.5
            u = np.cos(X + t * 0.5) * np.sin(Y) * 0.3
            v = -np.sin(X) * np.cos(Y + t * 0.3) * 0.3
            phase = np.angle(h + 1j * u)
            amp = np.abs(h + 1j * u)
            self.live_renderer.update_data(h, u, v, phase, amp)
            time.sleep(0.05)

    def stop_live_render(self):
        """停止实时渲染"""
        self._demo_running = False
        if self.live_process_thread:
            self.live_process_thread.stop()
            self.live_process_thread = None
        if self.live_renderer:
            self.live_renderer.stop()
            self.live_renderer = None
        self.live_render_btn.config(text="🎬 启动实时渲染")
        self.log("🛑 实时渲染已停止。")

    # =================================================================
    # 参数矩阵 (与原版一模一样)
    # =================================================================
    def build_param_matrix(self):
        for widget in self.param_container.winfo_children():
            widget.destroy()
        self.vortex_panel = None
        mode = self.board_var.get()
        num_channels = 24 if "24通道" in mode else 8
        num_boards = 3 if num_channels == 24 else 1

        self.param_frame = ttk.LabelFrame(self.param_container,
                                           text=f"波形参数与使能输入 ({num_channels} 通道)", padding=10)
        # 24 通道时左边放参数矩阵，右边空白处留给涡旋操控面板
        if num_channels == 24:
            self.param_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        else:
            self.param_frame.pack(fill=tk.X, expand=True)

        self.enables_vars = []
        self.amp_vars = []
        self.phase_vars = []
        self.period_vars = []

        headers = ["通道使能", "幅度 (0-1)\n(振幅最大比例)", "相位 (0-1)", "周期 (ms)"]

        for b in range(num_boards):
            row_offset = b * 6
            if num_boards > 1:
                ttk.Label(self.param_frame,
                          text=f"======== 硬件板 {b+1} (CH {b*8+1} ~ CH {b*8+8}) ========",
                          font=("", 10, "bold"), foreground="#0055AA").grid(
                    row=row_offset, column=0, columnspan=9, pady=(10 if b > 0 else 0), sticky="w")
                row_offset += 1

            for i, h in enumerate(headers):
                ttk.Label(self.param_frame, text=h, font=("", 9, "bold")).grid(
                    row=row_offset+i+1, column=0, padx=10, pady=2, sticky="e")

            for ch in range(8):
                global_ch = b * 8 + ch
                ttk.Label(self.param_frame, text=f"CH {global_ch+1}",
                          font=("", 10, "bold")).grid(row=row_offset, column=ch+1, pady=2)

                en_var = tk.BooleanVar(value=True)
                ttk.Checkbutton(self.param_frame, variable=en_var,
                                command=self.on_enable_toggle).grid(row=row_offset+1, column=ch+1)
                self.enables_vars.append(en_var)

                amp_var = tk.DoubleVar(value=1.0)
                ttk.Entry(self.param_frame, textvariable=amp_var, width=8,
                          justify="center").grid(row=row_offset+2, column=ch+1, padx=5, pady=2)
                self.amp_vars.append(amp_var)

                phase_var = tk.DoubleVar(value=0.0)
                ttk.Entry(self.param_frame, textvariable=phase_var, width=8,
                          justify="center").grid(row=row_offset+3, column=ch+1, padx=5, pady=2)
                self.phase_vars.append(phase_var)

                period_var = tk.IntVar(value=150)
                ttk.Entry(self.param_frame, textvariable=period_var, width=8,
                          justify="center").grid(row=row_offset+4, column=ch+1, padx=5, pady=2)
                self.period_vars.append(period_var)

        # 🌟 24 通道模式：右侧挂贝塞尔涡旋操控面板
        if num_channels == 24:
            self.vortex_panel = VortexArrayPanel(self.param_container, self)
            self.vortex_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(8, 0), anchor="n")

        # 窗口尺寸只是个建议值，放不下由滚动条兜底，绝不超出屏幕
        want_w, want_h = (1420, 1140) if num_channels == 24 else (1250, 920)
        w = min(want_w, self.root.winfo_screenwidth() - 40)
        h = min(want_h, self.root.winfo_screenheight() - 80)
        self.root.geometry(f"{w}x{h}")

    # =================================================================
    # 日志与串口
    # =================================================================
    def log(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def scan_ports(self):
        ports = serial.tools.list_ports.comports()
        port_list = [port.device for port in ports]
        if port_list:
            self.port_cb['values'] = port_list
            current = self.port_var.get()
            if current in port_list: self.port_cb.set(current)
            else: self.port_cb.set(port_list[0])
            self.log(f"🔄 刷新串口列表: 找到 {len(port_list)} 个设备 ({', '.join(port_list)})")
        else:
            self.port_cb['values'] = ["无可用串口"]
            self.port_cb.set("无可用串口")
            self.log("🔄 刷新串口列表: 未检测到设备，请检查 USB 连接。")

    def browse_save_dir(self):
        dir_path = filedialog.askdirectory(title="选择图像与定标数据保存主目录")
        if dir_path:
            self.save_dir_var.set(dir_path)
            self.save_hardware_config()
            self.log(f"📁 图像保存主目录已更新为: {dir_path}")

    # =================================================================
    # 相机：单张采集 (与原版一模一样)
    # =================================================================
    def capture_single_frame(self):
        if not self.camera or not self.camera.is_opened:
            messagebox.showerror("错误", "请先连接工业相机！")
            return
        target_dir = self.save_dir_var.get()
        if not target_dir or not os.path.exists(target_dir):
            messagebox.showerror("错误", "请先选择一个有效的图像保存路径！")
            return
        try:
            import cv2
            import platform
            import mvsdk

            exp_val = self.cam_exp_var.get()
            self.camera.set_exposure(exp_val)
            time.sleep(0.15)

            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self.camera.hCamera, 500)
            mvsdk.CameraImageProcess(self.camera.hCamera, pRawData, self.camera.pFrameBuffer, FrameHead)
            mvsdk.CameraReleaseImageBuffer(self.camera.hCamera, pRawData)

            if platform.system() == "Windows":
                mvsdk.CameraFlipFrameBuffer(self.camera.pFrameBuffer, FrameHead, 1)

            channels = 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3
            w, h = FrameHead.iWidth, FrameHead.iHeight
            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(self.camera.pFrameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((h, w, channels))

            if channels == 3: frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            elif channels == 1: frame = frame.reshape((h, w))

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(target_dir, f"Single_Capture_{timestamp}.tiff")
            encode_params = [int(cv2.IMWRITE_TIFF_COMPRESSION), 1]
            is_success, im_buf = cv2.imencode(".tiff", frame, encode_params)

            if is_success:
                im_buf.tofile(filename)
                self.log(f"📸 [单帧快照] 采集成功！无损 TIFF 已保存至:\n{filename}")
            else:
                self.log("❌ 编码单张图片失败")
        except Exception as e:
            self.log(f"❌ 采集单张图片异常: {e}")

    # =================================================================
    # 相机：序列采集 (与原版一模一样)
    # =================================================================
    def start_capture_sequence(self):
        if not self.camera or not self.camera.is_opened:
            messagebox.showerror("错误", "请先连接工业相机！")
            return
        target_dir = self.save_dir_var.get()
        if not target_dir or not os.path.exists(target_dir):
            messagebox.showerror("错误", "请先选择一个有效的图像保存主目录！")
            return
        frames_to_capture = self.seq_frames_var.get()
        if frames_to_capture <= 0: return
        self.seq_snap_btn.config(state=tk.DISABLED, text="采集进行中...")
        self._cancel_flag = False
        threading.Thread(target=self._capture_sequence_thread,
                         args=(target_dir, frames_to_capture), daemon=True).start()

    def _capture_sequence_thread(self, target_dir, num_frames):
        import cv2
        import platform
        import mvsdk

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        seq_dir = os.path.join(target_dir, f"Sequence_{timestamp}")
        os.makedirs(seq_dir, exist_ok=True)
        self.root.after(0, self.log, f"🎥 开始高速采集 {num_frames} 帧序列，存至:\n{seq_dir}")

        try:
            exp_val = self.cam_exp_var.get()
            self.camera.set_exposure(exp_val)
            time.sleep(0.15)
            encode_params = [int(cv2.IMWRITE_TIFF_COMPRESSION), 1]
            success_count = 0

            for i in range(num_frames):
                if self._cancel_flag:
                    self.root.after(0, self.log, "⚠️ 序列采集已被手动中止！")
                    break

                pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self.camera.hCamera, 1000)
                mvsdk.CameraImageProcess(self.camera.hCamera, pRawData, self.camera.pFrameBuffer, FrameHead)
                mvsdk.CameraReleaseImageBuffer(self.camera.hCamera, pRawData)

                if platform.system() == "Windows":
                    mvsdk.CameraFlipFrameBuffer(self.camera.pFrameBuffer, FrameHead, 1)

                channels = 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3
                w, h = FrameHead.iWidth, FrameHead.iHeight
                frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(self.camera.pFrameBuffer)
                frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((h, w, channels))

                if channels == 3: frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                elif channels == 1: frame = frame.reshape((h, w))

                filename = os.path.join(seq_dir, f"frame_{i:05d}.tiff")
                is_success, im_buf = cv2.imencode(".tiff", frame, encode_params)
                if is_success:
                    im_buf.tofile(filename)
                    success_count += 1

                if (i + 1) % 20 == 0:
                    self.root.after(0, self.log, f"已采集 {i+1}/{num_frames} 帧...")

            self.root.after(0, self.log, f"✅ 序列采集完成！成功保存 {success_count}/{num_frames} 帧。")
        except Exception as e:
            self.root.after(0, self.log, f"❌ 序列采集异常中断: {e}")
        finally:
            self.root.after(0, lambda: self.seq_snap_btn.config(state=tk.NORMAL, text="🎥 高速采集图像序列"))

    # =================================================================
    # 相机/串口 连接断开 (与原版一模一样)
    # =================================================================
    def toggle_camera(self):
        if MindVisionCamera is None:
            self.log("❌ 错误：找不到 camera_controller.py 或者 mvsdk.py！")
            return
        if self.camera is None or not self.camera.is_opened:
            try:
                self.log("正在尝试连接迈德威视工业相机...")
                self.camera = MindVisionCamera()
                self.camera.open_camera()
                self.connect_cam_btn.config(text="📸断开相机")
                self.log("✅ 相机连接成功并已启动数据流！")
                if hasattr(self, 'fcd_gui') and self.fcd_gui:
                    self.fcd_gui.set_camera_reference(self.camera)
            except Exception as e:
                self.log(f"❌ 相机连接失败: {e}")
                self.camera = None
        else:
            self.camera.close_camera()
            self.connect_cam_btn.config(text="📸连接相机")
            self.log("🔌 相机已安全断开。")
            if hasattr(self, 'fcd_gui') and self.fcd_gui:
                self.fcd_gui.set_camera_reference(None)

    def toggle_connection(self):
        if self.controller is None:
            port = self.port_var.get().strip()
            if port == "无可用串口" or not port:
                self.log("❌ 请先选择一个有效的串口！")
                return
            self.log(f"正在尝试连接 {port}...")
            self.controller = SpeakerArrayController(port=port)
            if self.controller.ser and self.controller.ser.is_open:
                self.connect_btn.config(text="🔌断开串口")
                self.execute_btn.config(state=tk.NORMAL)
                self.stop_btn.config(state=tk.NORMAL)
                self.carousel_btn.config(state=tk.NORMAL)
                self.log(f"✅ 成功连接到 {port}")
                self.save_hardware_config()
                self.controller.calibration_data = None
            else:
                self.controller = None
                self.log("❌ 连接失败。")
        else:
            self.controller.close()
            self.controller = None
            self.connect_btn.config(text="🔌连接串口")
            self.execute_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.DISABLED)
            self.carousel_btn.config(state=tk.DISABLED)
            self.log("🔌 串口已断开。")

    # =================================================================
    # LUT 文件导入 (与原版一模一样)
    # =================================================================
    def load_calibration_file(self, idx):
        file_path = filedialog.askopenfilename(
            title=f"选择 板{idx+1} 的 LUT 定标 JSON 文件", filetypes=[("JSON Files", "*.json")])
        if file_path:
            self.calib_vars[idx].set(file_path)
            self.save_hardware_config()
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.lut_data_list[idx] = json.load(f)
                self.log(f"✅ 板{idx+1} 的 LUT 独立定标曲线已载入内存！")
                if self.controller: self.controller.calibration_data = None
            except Exception as e:
                self.log(f"❌ 板{idx+1} LUT 解析失败: {e}")

    # =================================================================
    # 配置保存/加载 (在原版基础上增加 ref_img_path 和 FCD 光路参数)
    # =================================================================
    def save_hardware_config(self):
        config_data = {
            "calib_path_1": self.calib_vars[0].get(),
            "calib_path_2": self.calib_vars[1].get(),
            "calib_path_3": self.calib_vars[2].get(),
            "com_port": self.port_var.get(),
            "delay": self.delay_var.get(),
            "duration": self.duration_var.get(),
            "use_calibration": self.use_calib_var.get(),
            "cam_fps": self.cam_fps_var.get(),
            "cam_exp": self.cam_exp_var.get(),
            "global_period": self.global_period_var.get(),
            "global_phase": self.global_phase_var.get(),
            "save_dir": self.save_dir_var.get(),
            "target_board": self.board_var.get(),
            "ref_img_path": self.ref_img_var.get(),                 # 🌟 新增
            "fcd_water_depth": self.fcd_depth_var.get(),            # 🌟 新增：水深
            "fcd_plate_thickness": self.fcd_plate_var.get(),        # 🌟 新增：底板厚
            "fcd_plate_n": self.fcd_n_plate_var.get(),              # 🌟 新增：底板折射率
            "fcd_air_gap": self.fcd_gap_var.get(),                  # 🌟 新增：底板下气隙
            "fcd_water_n": self.fcd_n_water_var.get(),              # 🌟 新增：水折射率
            "live_roi": self.live_roi_var.get(),                     # 🌟 新增：中心ROI兜底尺寸
            "live_crop": [v.get() for v in self.live_crop_vars],     # 🌟 新增：框选的渲染区域
            "cal_spk_num": self.cal_spk_num_var.get(),               # 🌟 新增：定标喇叭数
        }
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=4)
        except Exception: pass

    def load_hardware_config(self):
        if not os.path.exists(self.config_file): return
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                cfg = json.load(f)

            for i in range(3):
                path = cfg.get(f"calib_path_{i+1}", "")
                if path and os.path.exists(path):
                    self.calib_vars[i].set(path)
                    with open(path, 'r', encoding='utf-8') as json_f:
                        self.lut_data_list[i] = json.load(json_f)
                    self.log(f"📂 自动检索到 板{i+1} LUT定标文件:\n{path}")

            if "com_port" in cfg: self.port_var.set(cfg.get("com_port", "COM6"))
            if "delay" in cfg: self.delay_var.set(cfg.get("delay", 0.0))
            if "duration" in cfg: self.duration_var.set(cfg.get("duration", 0.0))
            if "use_calibration" in cfg: self.use_calib_var.set(cfg.get("use_calibration", True))
            if "cam_fps" in cfg: self.cam_fps_var.set(cfg.get("cam_fps", 30.0))
            if "cam_exp" in cfg: self.cam_exp_var.set(cfg.get("cam_exp", 10.0))
            if "global_period" in cfg: self.global_period_var.set(cfg.get("global_period", 150))
            if "global_phase" in cfg: self.global_phase_var.set(cfg.get("global_phase", 0.0))

            s_dir = cfg.get("save_dir", "")
            if s_dir and os.path.exists(s_dir): self.save_dir_var.set(s_dir)
            else: self.save_dir_var.set(os.path.dirname(__file__))

            # 🌟 新增：恢复FCD参考图与光路参数
            ref_path = cfg.get("ref_img_path", "")
            if ref_path and os.path.exists(ref_path):
                self.ref_img_var.set(ref_path)
                self.log(f"📂 自动加载FCD参考图: {ref_path}")

            for key, var, default in (
                ("fcd_water_depth", self.fcd_depth_var, 30.0),
                ("fcd_plate_thickness", self.fcd_plate_var, 10.0),
                ("fcd_plate_n", self.fcd_n_plate_var, 1.49),
                ("fcd_air_gap", self.fcd_gap_var, 0.0),
                ("fcd_water_n", self.fcd_n_water_var, 1.333),
            ):
                if key in cfg:
                    var.set(cfg.get(key, default))
            if "cal_spk_num" in cfg:
                self.cal_spk_num_var.set(cfg.get("cal_spk_num", 8))
            if "live_roi" in cfg:
                self.live_roi_var.set(cfg.get("live_roi", 512))
            saved_crop = cfg.get("live_crop")
            if isinstance(saved_crop, (list, tuple)) and len(saved_crop) == 4:
                for cv, val in zip(self.live_crop_vars, saved_crop):
                    cv.set(int(val))
                if saved_crop[1] > saved_crop[0]:
                    self.log(f"📂 已恢复框选的渲染区域: X {saved_crop[0]}-{saved_crop[1]}, "
                             f"Y {saved_crop[2]}-{saved_crop[3]}")
            if "fcd_H" in cfg and "fcd_water_depth" not in cfg:
                self.log("ℹ️ 旧缓存里的固定 fcd_H 已弃用，现改为按水深实时换算，请确认水深与光路参数。")

            if "target_board" in cfg:
                self.board_var.set(cfg.get("target_board", "1"))
                self.build_param_matrix()
        except Exception as e:
            self.log(f"⚠️ 读取硬件历史缓存异常: {e}")

    # =================================================================
    # 通道使能/一键停止 (与原版一模一样)
    # =================================================================
    def on_enable_toggle(self):
        if self.controller and self.controller.ser and self.controller.ser.is_open:
            mode = self.board_var.get()
            enables = [var.get() for var in self.enables_vars]
            if "24通道" in mode:
                r1 = self.controller.write_channel_enables(1, enables[0:8])
                r2 = self.controller.write_channel_enables(2, enables[8:16])
                r3 = self.controller.write_channel_enables(3, enables[16:24])
                self.log(f"🔄 24通道开关更新 -> 板1:[{r1}] 板2:[{r2}] 板3:[{r3}]")
            else:
                board_id = int(mode.split()[0]) if "所有" not in mode else 0
                resp = self.controller.write_channel_enables(board_id, enables)
                self.log(f"🔄 通道开关已实时更新 -> 返回: {resp}")

    def stop_all_speakers(self):
        self._cancel_flag = True
        if not self.controller: return
        mode = self.board_var.get()
        self.log(f"\n--- 🛑 执行快捷操作: 一键停止 ({mode}) ---")
        try:
            if "24通道" in mode:
                r1 = self.controller.stop_all(1)
                r2 = self.controller.stop_all(2)
                r3 = self.controller.stop_all(3)
                self.log(f"停止指令已下发。板1:[{r1}] 板2:[{r2}] 板3:[{r3}]")
            else:
                board_id = int(mode.split()[0]) if "所有" not in mode else 0
                resp = self.controller.stop_all(board_id)
                self.log(f"停止指令已下发。返回结果: {resp}")
        except Exception as e:
            self.log(f"❌ 执行出错: {str(e)}")

    # =================================================================
    # LUT 映射引擎 (与原版一模一样，含锚点逻辑)
    # =================================================================
    def _apply_lut_calibration(self, params, board_idx):
        if not self.use_calib_var.get(): return params

        lut_data = self.lut_data_list[board_idx - 1]
        if not lut_data: return params

        all_maxes = [ld.get("Global_Max_Amp_mm", 1.0) for ld in self.lut_data_list if ld is not None]
        absolute_global_max = min(all_maxes) if all_maxes else 1.0
        calibrated = []

        for i, p in enumerate(params):
            if p['amp'] == 0:
                calibrated.append(p)
                continue

            ch_key = f"CH{i+1}"
            if ch_key not in lut_data.get("Speakers", {}):
                calibrated.append(p)
                continue

            ch_data = lut_data["Speakers"][ch_key]
            v_in_arr = ch_data["v_in"]
            amp_out_arr = ch_data["amp_out"]
            phase_out_arr = ch_data["phase_out"]

            target_amp_mm = p['amp'] * absolute_global_max
            req_v = np.interp(target_amp_mm, amp_out_arr, v_in_arr)
            expected_phase_delay = np.interp(req_v, v_in_arr, phase_out_arr)

            anchor_lut = self.lut_data_list[0]
            if anchor_lut and "CH1" in anchor_lut.get("Speakers", {}):
                ref_data = anchor_lut["Speakers"]["CH1"]
            else:
                ref_data = lut_data["Speakers"]["CH1"]

            ref_v = np.interp(target_amp_mm, ref_data["amp_out"], ref_data["v_in"])
            ref_delay = np.interp(ref_v, ref_data["v_in"], ref_data["phase_out"])

            final_phase = p['phase'] % 1.0
            calibrated.append({"amp": req_v, "phase": final_phase, "period": p['period']})

        return calibrated

    # =================================================================
    # 自动定标：轮播采集 (与原版一模一样，含锚点通道)
    # =================================================================
    def start_carousel(self):
        """启动全局自动定标流程 - 支持单板/跨板两种模式"""
        if not self.controller:
            self.log("❌ 请先连接串口！")
            return

        mode = self.board_var.get()

        # 判断工作模式
        if "24通道" in mode:
            # 24通道跨板模式：根据通道号自动分配板子
            cross_board = True
            base_board = 1  # 占位，实际会按通道计算
        elif "所有" in mode:
            messagebox.showwarning("模式限制",
                "自动定标不支持'所有(广播)'模式！\n"
                "请在上方选择具体的板子 [1] [2] [3] 或 [24通道] 模式。")
            return
        else:
            # 🌟 关键修复：使用用户在界面选择的具体板子编号！
            cross_board = False
            try:
                base_board = int(mode.split()[0])
            except Exception:
                base_board = 1
            if base_board not in [1, 2, 3]:
                messagebox.showwarning("板号错误", f"无法识别板号: {mode}")
                return

        self._cancel_flag = False
        threading.Thread(
            target=self._lut_carousel_thread,
            args=(cross_board, base_board),
            daemon=True
        ).start()

    def _lut_carousel_thread(self, cross_board=False, base_board=1):
        """
        全局定标线程 - 修复版

        关键修复点:
        1. cross_board=False 时，所有通道都发送到 base_board（用户选择的板子）
        2. cross_board=True 时，根据通道号自动计算 board_id = (global_ch // 8) + 1
        3. 不再硬编码 board_id=1
        """
        # ---------- 相机检查 ----------
        if not self.camera or not self.camera.is_opened:
            self.root.after(0, self.log, "❌ 错误：尚未连接工业相机！请先点击上方【连接相机】按钮。")
            self._cancel_flag = True
            return

        exp_val = self.cam_exp_var.get()
        self.camera.set_exposure(exp_val)
        time.sleep(0.15)

        # ---------- 参数验证 ----------
        num_speakers = self.cal_spk_num_var.get()
        if num_speakers < 1 or num_speakers > 24:
            self.root.after(0, self.log, "❌ 错误：定标喇叭数量必须设置在 1 ~ 24 之间！")
            return

        # 如果不是跨板模式，且喇叭数 > 8，截断为 8（单板最多8个通道）
        if not cross_board and num_speakers > 8:
            self.root.after(0, self.log, f"⚠️ 单板模式下最多8通道，已将定标数量从 {num_speakers} 截断为 8")
            num_speakers = 8

        # ---------- 日志输出模式信息 ----------
        self.root.after(0, self.log, "\n" + "=" * 50)
        if cross_board:
            self.root.after(0, self.log, f"🎬 开始【24通道跨板】全局自动声光同步定标！目标喇叭数: {num_speakers}")
        else:
            self.root.after(0, self.log, f"🎬 开始为【硬件板 {base_board}】执行自动声光同步定标！目标喇叭数: {num_speakers}")
        self.root.after(0, self.log, "👉 黄金时序: 0.0s起振 -> 0.33s拍摄 -> 1.75s停震 -> 5.0s冷却落盘")
        self.root.after(0, self.log, "=" * 50)

        # ---------- 暂时禁用 LUT 修正（定标采集原始数据） ----------
        backup_calib = self.controller.calibration_data
        self.controller.calibration_data = None

        # ---------- 定标参数 ----------
        levels = [0.1, 0.4, 0.7, 1.0]
        target_fps = self.cam_fps_var.get()
        global_period = self.global_period_var.get()
        global_phase = self.global_phase_var.get()

        # ---------- 保存目录 ----------
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        root_save_dir = self.save_dir_var.get()
        if not root_save_dir or not os.path.exists(root_save_dir):
            root_save_dir = os.path.dirname(__file__)

        if cross_board:
            base_dir = os.path.join(root_save_dir, f"Calibration_RAW_Total_{timestamp}")
        else:
            base_dir = os.path.join(root_save_dir, f"Calibration_RAW_Board{base_board}_{timestamp}")
        os.makedirs(base_dir, exist_ok=True)

        # ---------- 主循环 ----------
        try:
            for global_ch in range(num_speakers):
                if self._cancel_flag:
                    break

                # 🌟🌟🌟 核心修复：正确计算 board_id 🌟🌟🌟
                if cross_board:
                    # 跨板模式：板号由通道位置决定
                    board_id = (global_ch // 8) + 1
                    local_ch = global_ch % 8
                    ch_tag = global_ch + 1          # 跨板时用全局编号 CH1~CH24
                else:
                    # 单板模式：始终使用用户选择的板子！
                    board_id = base_board
                    local_ch = global_ch % 8
                    ch_tag = local_ch + 1           # 单板时用板内编号 CH1~CH8

                for lvl in levels:
                    if self._cancel_flag:
                        break

                    # 命名必须保持 CHx_AmpY.Y，fcd_gui 的 32 阶梯定标就按这个格式找文件夹
                    folder_name = f"CH{ch_tag}_Amp{lvl:.1f}"
                    save_dir = os.path.join(base_dir, folder_name)
                    os.makedirs(save_dir, exist_ok=True)

                    self.root.after(0, self.log,
                        f"🔊 [{folder_name}] 硬件板{board_id} CH{local_ch + 1} 振幅={lvl:.1f} 激发中...")

                    # 构建步骤参数：只激活目标通道，其余全部静音
                    step_params = []
                    for i in range(8):
                        if i == local_ch:
                            step_params.append({
                                "amp": lvl,
                                "phase": global_phase,
                                "period": global_period
                            })
                        else:
                            step_params.append({
                                "amp": 0.0,
                                "phase": 0.0,
                                "period": global_period
                            })

                    base_time = time.perf_counter()

                    # 先使能目标通道（只开当前定标通道，其余关闭）
                    enables = [False] * 8
                    enables[local_ch] = True
                    self.controller.write_channel_enables(board_id, enables)

                    # 下发控制指令到正确的板子
                    self.controller.write_waveform_params(board_id, step_params)

                    # 等待 0.33 秒让波形建立稳定
                    while (time.perf_counter() - base_time) < 0.33:
                        if self._cancel_flag:
                            break
                        time.sleep(0.01)

                    # 开始相机录制
                    if not self._cancel_flag:
                        self.camera.start_recording(target_fps=target_fps, duration_sec=1.5)

                    # 等待到 1.75 秒后停止输出
                    while (time.perf_counter() - base_time) < 1.75:
                        if self._cancel_flag:
                            break
                        time.sleep(0.01)

                    # 停止该板输出
                    self.controller.stop_all(board_id)

                    # 等待到 1.83 秒确保停稳
                    while (time.perf_counter() - base_time) < 1.83:
                        if self._cancel_flag:
                            break
                        time.sleep(0.01)

                    if self._cancel_flag:
                        break

                    self.root.after(0, self.log, "⏳ 进入 5s 恢复冷却...")

                    # 保存相机数据
                    save_start = time.perf_counter()
                    try:
                        self.camera.wait_and_save(save_dir, prefix=folder_name)
                    except Exception as save_err:
                        self.root.after(0, self.log, f"⚠️ 图像保存异常: {save_err}")

                    # 确保总冷却时间达到 5 秒
                    elapsed = time.perf_counter() - save_start
                    remain = 5.0 - elapsed
                    if remain > 0:
                        time.sleep(remain)

            # ---------- 完成 ----------
            if not self._cancel_flag:
                self.root.after(0, self.log,
                    f"✅ 全通道定标轮播完毕！\n📁 数据保存在:\n{base_dir}")

            # 确保所有板停止并关闭使能
            if cross_board:
                for b in [1, 2, 3]:
                    self.controller.stop_all(b)
                    self.controller.write_channel_enables(b, [False] * 8)
            else:
                self.controller.stop_all(base_board)
                self.controller.write_channel_enables(base_board, [False] * 8)

        except Exception as e:
            self.root.after(0, self.log, f"⛔ 同步轮播异常: {e}")
            self.root.after(0, self.log, traceback.format_exc())
        finally:
            # 恢复 LUT 修正数据
            self.controller.calibration_data = backup_calib

    # =================================================================
    # 定时执行线程 (与原版一模一样)
    # =================================================================
    def _timed_execution_thread(self, mode, params, delay, duration):
        if delay > 0:
            t = 0
            while t < delay:
                if self._cancel_flag: return
                time.sleep(0.1)
                t += 0.1
        if self._cancel_flag: return

        try:
            if "24通道" in mode:
                cal_1 = self._apply_lut_calibration(params[0:8], board_idx=1)
                cal_2 = self._apply_lut_calibration(params[8:16], board_idx=2)
                cal_3 = self._apply_lut_calibration(params[16:24], board_idx=3)
                self.controller.write_waveform_params(1, cal_1)
                self.controller.write_waveform_params(2, cal_2)
                self.controller.write_waveform_params(3, cal_3)
                self.root.after(0, self.log, "⏳ 定时参数已下发 (24通道，独立查表映射后)。")
            else:
                b_id = int(mode.split()[0]) if "所有" not in mode else 0
                lut_idx = b_id if b_id in [1, 2, 3] else 1
                cal = self._apply_lut_calibration(params[0:8], board_idx=lut_idx)
                resp = self.controller.write_waveform_params(b_id, cal)
                self.root.after(0, self.log, f"⏳ 定时参数已下发 (查表映射后)。返回: {resp}")
        except Exception as e:
            self.root.after(0, self.log, f"⛔ 下发异常: {e}")
            return

        if self._cancel_flag: return

        if duration > 0:
            t = 0
            while t < duration:
                if self._cancel_flag: return
                time.sleep(0.1)
                t += 0.1
            if not self._cancel_flag:
                try:
                    if "24通道" in mode:
                        self.controller.stop_all(1); self.controller.stop_all(2); self.controller.stop_all(3)
                    else:
                        b_id = int(mode.split()[0]) if "所有" not in mode else 0
                        self.controller.stop_all(b_id)
                    self.root.after(0, self.log, "⏳ 已按计划自动停止。")
                except Exception: pass

    # =================================================================
    # 主执行入口 (与原版一模一样)
    # =================================================================
    def execute_operation(self):
        if not self.controller: return
        self.save_hardware_config()
        mode = self.board_var.get()
        op_id = int(self.op_var.get().split(":")[0])
        num_channels = len(self.enables_vars)
        self.log(f"\n--- 执行操作: {self.op_var.get()} (目标: {mode}) ---")

        try:
            if op_id == 0:
                if "24通道" in mode:
                    r1 = self.controller.test_connection(1)
                    r2 = self.controller.test_connection(2)
                    r3 = self.controller.test_connection(3)
                    self.log(f"返回(24通道):\n板1: {r1}\n板2: {r2}\n板3: {r3}")
                else:
                    b_id = int(mode.split()[0]) if "所有" not in mode else 0
                    self.log(f"返回: {self.controller.test_connection(b_id)}")

            elif op_id == 1:
                params = []
                enables = [var.get() for var in self.enables_vars]
                for i in range(num_channels):
                    final_amp = self.amp_vars[i].get() if enables[i] else 0.0
                    params.append({"amp": final_amp, "phase": self.phase_vars[i].get(),
                                   "period": self.period_vars[i].get()})

                delay, duration = self.delay_var.get(), self.duration_var.get()
                if delay > 0 or duration > 0:
                    self._cancel_flag = False
                    threading.Thread(target=self._timed_execution_thread,
                                     args=(mode, params, delay, duration), daemon=True).start()
                else:
                    if "24通道" in mode:
                        cal_1 = self._apply_lut_calibration(params[0:8], board_idx=1)
                        cal_2 = self._apply_lut_calibration(params[8:16], board_idx=2)
                        cal_3 = self._apply_lut_calibration(params[16:24], board_idx=3)
                        r1 = self.controller.write_waveform_params(1, cal_1)
                        r2 = self.controller.write_waveform_params(2, cal_2)
                        r3 = self.controller.write_waveform_params(3, cal_3)
                        self.log(f"24路独立参数已并发下发。返回-> 板1:[{r1}] 板2:[{r2}] 板3:[{r3}]")
                    else:
                        b_id = int(mode.split()[0]) if "所有" not in mode else 0
                        lut_idx = b_id if b_id in [1, 2, 3] else 1
                        cal = self._apply_lut_calibration(params[0:8], board_idx=lut_idx)
                        resp = self.controller.write_waveform_params(b_id, cal)
                        self.log(f"参数已下发 (独立查表映射后)。返回: {resp}")

            elif op_id == 2:
                if "24通道" in mode:
                    r1 = self.controller.read_waveform_params(1)
                    r2 = self.controller.read_waveform_params(2)
                    r3 = self.controller.read_waveform_params(3)
                    self.log(f"读取波形(24通道):\n板1: {r1}\n板2: {r2}\n板3: {r3}")
                else:
                    b_id = int(mode.split()[0]) if "所有" not in mode else 0
                    resp = self.controller.read_waveform_params(b_id)
                    self.log(f"读取结果: {resp}")

            elif op_id == 3:
                enables = [var.get() for var in self.enables_vars]
                if "24通道" in mode:
                    r1 = self.controller.write_channel_enables(1, enables[0:8])
                    r2 = self.controller.write_channel_enables(2, enables[8:16])
                    r3 = self.controller.write_channel_enables(3, enables[16:24])
                    self.log(f"使能已下发。返回-> 板1:[{r1}] 板2:[{r2}] 板3:[{r3}]")
                else:
                    b_id = int(mode.split()[0]) if "所有" not in mode else 0
                    resp = self.controller.write_channel_enables(b_id, enables)
                    self.log(f"使能状态已下发。返回结果: {resp}")

            elif op_id == 4:
                if "24通道" in mode:
                    r1 = self.controller.save_configuration(1)
                    r2 = self.controller.save_configuration(2)
                    r3 = self.controller.save_configuration(3)
                    self.log(f"保存配置返回-> 板1:[{r1}] 板2:[{r2}] 板3:[{r3}]")
                else:
                    b_id = int(mode.split()[0]) if "所有" not in mode else 0
                    resp = self.controller.save_configuration(b_id)
                    self.log(f"保存配置返回: {resp}")

            elif op_id == 5:
                if "24通道" in mode:
                    r1 = self.controller.reset_device(1)
                    r2 = self.controller.reset_device(2)
                    r3 = self.controller.reset_device(3)
                    self.log(f"复位返回-> 板1:[{r1}] 板2:[{r2}] 板3:[{r3}]")
                else:
                    b_id = int(mode.split()[0]) if "所有" not in mode else 0
                    resp = self.controller.reset_device(b_id)
                    self.log(f"设备复位返回: {resp}")

        except ValueError as ve:
            self.log(f"⛔ 安全拦截: {str(ve)}")
            messagebox.showerror("硬件安全警告", str(ve))
        except Exception as e:
            self.log(f"❌ 执行出错: {str(e)}")
            self.log(traceback.format_exc())


# =====================================================================
# 程序入口
# =====================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = LabviewMimicGUI(root)

    def on_closing():
        # 🌟 新增：停止实时渲染
        if hasattr(app, 'live_renderer') and app.live_renderer:
            app.stop_live_render()
        if app.controller: app.controller.close()
        if app.camera: app.camera.close_camera()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()