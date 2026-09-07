# -*- coding: utf-8 -*-
"""
软件包络播放器 (envelope_player.py)  v2
========================================
固件的任意波形模式尚不可用,本工具用验证过的标准 ## 指令按时间序列
改写幅度/相位/周期,以"正弦载波 + PC 侧包络"近似非正弦激励。
可以做:高斯波包、渐升渐降、拍频、幅度调制、慢扫频、通道间时序编排。
不能做:改变载波单个周期内的波形形状(那必须等固件的任意模式)。

⚠️ 固件特性 (2026-08 env_rate_test 实测): 每次 ## 写入都会重启波形相位!
   由此得出两条铁律,本程序已内置:
   1. 更新间隔必须大于载波周期(否则波形走不完一个周期,输出被压小/压没)
      —— 默认 4Hz(250ms 间隔),适配 150ms 周期;周期更长时自动报警。
   2. 参数没变就不重写 —— 幅度量化到 0.5% 步进,和上一帧相同的板帧直接
      跳过,包络平坦段载波完全连续无干扰。
   副作用:包络变化期间,该板 8 个通道的载波相位会在每次更新时一起重置;
   需要相位严格连续的测量请把包络设计成分段恒定。

用法:
    python envelope_player.py --plot    # 只预览包络曲线,不碰硬件
    python envelope_player.py           # 连接硬件实际播放

配置在下方 PROGRAM 区:
    amp   : 0~1,常数或含 t(秒) 的公式字符串(可用 np)。1 = 硬件安全上限
            (与标准面板满幅一致,底层自动乘 0.2 物理安全锁)。
    phase : 0~1,常数或公式。
    period: 载波周期 ms,常数或公式(运行中改周期同样会重启相位)。
未列出的通道幅度恒为 0。也可以 `from envelope_player import play` 在
test.py 那样的时间轴脚本里调用,与相机录制编排在一起。
"""

import sys
import time
import math
import serial
import numpy as np

# ==================== 硬件配置 ====================
COM_PORT  = "COM4"
BAUD      = 115200
UPDATE_HZ = 4.0      # 更新率;间隔(=1/UPDATE_HZ)必须大于最长载波周期
DURATION  = 12.0     # 播放总时长(秒)
AMP_QUANT = 200      # 幅度量化档数(200 = 0.5% 步进,用于跳过未变化的帧)

# ==================== 包络程序 ====================
# PROGRAM[板号][通道号(1-8)] = {"amp": ..., "phase": ..., "period": ...}
PROGRAM = {
    1: {
        1: {  # CH1: 高斯波包,t=6s 处最强
            "amp": "np.exp(-((t - 6.0) / 2.0) ** 2)",
            "phase": 0.0,
            "period": 150,
        },
        2: {  # CH2: 0.25Hz 拍频包络(波浪式起伏)
            "amp": "0.5 * (1 + np.cos(2 * np.pi * 0.25 * t)) / 2",
            "phase": 0.0,
            "period": 150,
        },
    },
}
# ==================================================


def _compile_program(program):
    """把公式字符串预编译,常数原样保留"""
    compiled = {}
    for board, chans in program.items():
        compiled[board] = {}
        for ch, spec in chans.items():
            entry = {}
            for key, default in (("amp", 0.0), ("phase", 0.0), ("period", 150)):
                val = spec.get(key, default)
                if isinstance(val, str):
                    entry[key] = compile(val, f"<{key} B{board}CH{ch}>", "eval")
                else:
                    entry[key] = val
            compiled[board][ch] = entry
    return compiled


def _eval_at(entry, t):
    env = {"np": np, "math": math, "t": t}
    out = {}
    for key in ("amp", "phase", "period"):
        v = entry[key]
        out[key] = float(eval(v, {"__builtins__": {}}, env)) if not isinstance(v, (int, float)) else float(v)
    return out


def _build_frame(board, chans, t):
    """一块板的 ## 参数帧;幅度先量化(便于跳帧)再过 0.2 物理安全锁"""
    payload = ""
    for i in range(8):
        spec = chans.get(i + 1)
        if spec is None:
            payload += f"{0:04X}{0:04X}{150:08X}"
            continue
        v = _eval_at(spec, t)
        a = max(0.0, min(1.0, v["amp"]))
        a = round(a * AMP_QUANT) / AMP_QUANT * 0.2
        p = max(0.0, min(1.0, v["phase"]))
        per = max(1, int(round(v["period"])))
        payload += f"{int(a * 65535):04X}{int(p * 65535):04X}{per:08X}"
    return f"{board}##{payload}"


def _send(ser, text):
    ser.write((text + "\r\n").encode("ascii"))
    ser.flush()


def play(program=PROGRAM, duration=DURATION, update_hz=UPDATE_HZ,
         com_port=COM_PORT, log=print):
    """按时间轴播放包络程序。可从外部脚本 import 调用,与相机时序编排。"""
    compiled = _compile_program(program)
    n_ticks = int(duration * update_hz)
    interval_ms = 1000.0 / update_hz

    # 铁律 1 检查: 更新间隔必须大于最长载波周期
    max_period = 0.0
    for chans in compiled.values():
        for entry in chans.values():
            max_period = max(max_period, _eval_at(entry, 0.0)["period"])
    if interval_ms <= max_period:
        log(f"⚠️⚠️ 更新间隔 {interval_ms:.0f}ms ≤ 载波周期 {max_period:.0f}ms,"
            f"波形会被反复打断而压制!请把 UPDATE_HZ 降到 {1000.0 / max_period:.1f} 以下。")

    ser = serial.Serial(com_port, BAUD, timeout=1.0, dsrdtr=False, rtscts=False)
    ser.setDTR(False)
    ser.setRTS(False)
    log("⏳ 等待硬件初始化 (1.5 秒)...")
    time.sleep(1.5)
    ser.reset_input_buffer()

    late = sent = skipped = 0
    try:
        # 起手:幅度置零 + 按程序使能对应通道
        last_frame = {}
        for board, chans in compiled.items():
            _send(ser, _build_frame(board, {}, 0.0))
            mask = 0
            for ch in chans:
                mask |= 1 << (ch - 1)
            _send(ser, f"{board}#E{mask:02X}")
            time.sleep(0.1)
        if ser.in_waiting:
            ser.read(ser.in_waiting)

        log(f"▶ 开始播放:{duration}s @ {update_hz}Hz (间隔 {interval_ms:.0f}ms)")
        start = time.perf_counter()
        for k in range(n_ticks + 1):
            t = k / update_hz
            for board, chans in compiled.items():
                fr = _build_frame(board, chans, t)
                if fr == last_frame.get(board):
                    skipped += 1          # 铁律 2: 内容没变不重写,不打扰载波
                else:
                    _send(ser, fr)
                    last_frame[board] = fr
                    sent += 1
            if ser.in_waiting:
                ser.read(ser.in_waiting)  # 丢弃回执,保持节拍
            target = start + (k + 1) / update_hz
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                late += 1
        log(f"⏹ 播放完成。实发 {sent} 帧,跳过 {skipped} 帧(未变化),迟到 {late} 帧"
            + (" (建议降低 UPDATE_HZ)" if late > n_ticks * 0.05 else ""))
    finally:
        try:
            for board in compiled:
                _send(ser, _build_frame(board, {}, 0.0))
                _send(ser, f"{board}#E00")
            time.sleep(0.1)
        except Exception:
            pass
        ser.close()
        log("🔌 已归零并释放串口。")


def plot_preview(program=PROGRAM, duration=DURATION):
    import matplotlib.pyplot as plt
    compiled = _compile_program(program)
    ts = np.linspace(0, duration, 600)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for board, chans in compiled.items():
        for ch, entry in chans.items():
            vals = [_eval_at(entry, t) for t in ts]
            ax1.plot(ts, [v["amp"] for v in vals], label=f"B{board}-CH{ch}")
            ax2.plot(ts, [v["period"] for v in vals], label=f"B{board}-CH{ch}")
    ax1.set_ylabel("amp (0-1)")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True)
    ax1.legend(fontsize=8)
    ax1.set_title("Envelope Preview")
    ax2.set_ylabel("period (ms)")
    ax2.set_xlabel("t (s)")
    ax2.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plot_preview()
    else:
        play()
