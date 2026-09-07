# -*- coding: utf-8 -*-
"""
## 参数重写频率耐受测试 (env_rate_test.py)
===========================================
envelope_player 以 20Hz 连续重写 ## 参数时喇叭不动。本测试找出固件能
承受的最大重写频率,并区分两种病因:
  病因A: 每次 ## 写入都重启波形相位 -> 重写太密波形永远走不完 -> 静音
  病因B: 固件处理不过来 -> 接收缓冲区溢出 -> 指令全废

流程(全程 CH1 恒定满幅、周期 150ms,参数每次一模一样,只变发送频率):
  阶段0: 只写一次 + 使能,观察 4 秒          <- 对照组,必须动
  阶段1~5: 分别以 1s / 0.5s / 0.2s / 0.1s / 0.05s 间隔连续重写 5 秒
           每个阶段后停写 2 秒(间歇)
观察记录(最关键!): 每个阶段"重写期间"动不动、"间歇期间"动不动。
  - 重写期间不动但间歇恢复 -> 病因A(重写打断波形),记下最快的还能动的间隔
  - 到某阶段后连间歇也不动 -> 病因B(固件被打挂),记下从哪个阶段开始
脚本同时统计每阶段固件回执数(发N帧应回N个'1##'),回执掉队=固件跟不上。
    python env_rate_test.py
"""

import time
import serial

COM_PORT = "COM8"
BAUD     = 115200
BOARD    = 1
STD_AMP  = 0x3333
PERIOD   = 150


def frame():
    payload = f"{STD_AMP:04X}{0:04X}{PERIOD:08X}" + "".join(
        f"{0:04X}{0:04X}{PERIOD:08X}" for _ in range(7))
    return f"{BOARD}##{payload}"


def send(ser, text):
    ser.write((text + "\r\n").encode("ascii"))
    ser.flush()


def drain(ser):
    buf = b""
    while ser.in_waiting:
        buf += ser.read(ser.in_waiting)
        time.sleep(0.005)
    return buf


def read_reply(ser, window=1.0):
    deadline = time.time() + window
    buf = b""
    while time.time() < deadline:
        if ser.in_waiting:
            buf += ser.read(ser.in_waiting)
            deadline = time.time() + 0.15
        time.sleep(0.01)
    return buf.decode("ascii", errors="replace").strip()


def main():
    ser = serial.Serial(COM_PORT, BAUD, timeout=1.0, dsrdtr=False, rtscts=False)
    ser.setDTR(False)
    ser.setRTS(False)
    time.sleep(1.5)
    ser.reset_input_buffer()

    try:
        send(ser, f"{BOARD}#H");   print("握手:", read_reply(ser))
        send(ser, f"{BOARD}#RST"); print("复位:", read_reply(ser, 1.5))
        time.sleep(0.5)

        # ---------- 阶段0: 对照组 ----------
        send(ser, frame());          print("写参数:", read_reply(ser))
        send(ser, f"{BOARD}#E01");   print("使能:", read_reply(ser))
        print("\n👂 [阶段0 对照] 只写了一次 —— CH1 必须在动!(4 秒)")
        time.sleep(4)

        # ---------- 阶段1~5: 梯度重写 ----------
        intervals = [1.0, 0.5, 0.2, 0.1, 0.05]
        for stage, itv in enumerate(intervals, 1):
            n = max(3, int(5.0 / itv))
            print(f"\n👂 [阶段{stage}] 每 {itv*1000:.0f}ms 重写一次,共 {n} 帧 (5 秒) —— 重写期间动吗?")
            drain(ser)
            acked = b""
            start = time.perf_counter()
            for k in range(n):
                send(ser, frame())
                acked += drain(ser)
                target = start + (k + 1) * itv
                dt = target - time.perf_counter()
                if dt > 0:
                    time.sleep(dt)
            time.sleep(0.3)
            acked += drain(ser)
            n_ack = acked.count(b"##")
            garbled = sum(1 for b in acked if b > 127)
            print(f"   ℹ️ 发 {n} 帧,收到 {n_ack} 个 '##' 回执" +
                  (f",{garbled} 个乱码字节" if garbled else ""))
            print(f"👂 [阶段{stage} 间歇] 停止发送 2 秒 —— 现在动吗?")
            time.sleep(2)

        print("\n👂 [收尾观察] 全部测完,再看 3 秒当前状态")
        time.sleep(3)

    finally:
        print("\n🧹 安全停止……")
        try:
            drain(ser)
            zero = f"{BOARD}##" + "".join(f"{0:04X}{0:04X}{PERIOD:08X}" for _ in range(8))
            send(ser, zero);           print("全零:", read_reply(ser, 0.5))
            send(ser, f"{BOARD}#E00"); print("关使能:", read_reply(ser, 0.5))
            send(ser, f"{BOARD}#RST"); print("复位:", read_reply(ser, 1.0))
        except Exception:
            pass
        ser.close()
        print("🔌 串口已释放。")


if __name__ == "__main__":
    main()
