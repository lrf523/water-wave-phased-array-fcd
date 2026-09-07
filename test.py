# ni_trace_exact_test_ch4.py
import time
import serial

COM_PORT = "COM6"

M_BOARD = 0
PARAM_BOARD = 1
PERIOD = 150

# 来自 LabVIEW 的连接字符串
WAVE_HEADER = "A40001020303F3F3F"

# 先用全 2048 静止波形，安全判断
FLAT_LINES = [
    "A00800800800800800800800800800800800800800800800800800800",
    "A10800800800800800800800800800800800800800800800800800800",
    "A20800800800800800800800800800800800800800800800800800800",
    "A30800800800800800800800800800800800800800800800800800800",
]

# 你截图里的 tan(x) 数据，确认静止版跑通后再换这个
TAN_LINES = [
    "A00800BDB9BCAA7BA3CBADF6F68FFFFFFFFFFFFFFFFFFFFF001",
    "A100010010010010010010011732C43E94EF5E16C57A287D95A",
    "A20A40B34C3ED68EC1FFFFFFFFFFFFFFFFFFFFFFFFF001001001",
    "A30010010010010CF23936F48157B66474381E8FA9DCAC9BC8",
]

# 默认先测静止平线，避免炸麦
DATA_LINES = FLAT_LINES
# 如果静止版能让喇叭停，再改成：
# DATA_LINES = TAN_LINES


def read_all(ser, window=1.0):
    buf = b""
    deadline = time.time() + window
    while time.time() < deadline:
        n = ser.in_waiting
        if n:
            buf += ser.read(n)
            deadline = time.time() + 0.2
        time.sleep(0.02)
    return buf.decode("ascii", errors="replace").strip()


def send(ser, cmd, tag, wait=0.15, read_window=1.0):
    print(f"\n-> [{tag}] {cmd if len(cmd) < 120 else cmd[:120] + '...'}")
    ser.write((cmd + "\r\n").encode("ascii"))
    ser.flush()
    time.sleep(wait)
    reply = read_all(ser, read_window)
    print(f"<- [{tag}] {reply if reply else '<无回包>'}")
    return reply


def make_ch4_param():
    ch_off = f"00000000{PERIOD:08X}"
    ch4_on = f"33330000{PERIOD:08X}"
    return f"{PARAM_BOARD}##" + ch_off * 3 + ch4_on + ch_off * 4


def make_all_off_param():
    ch_off = f"00000000{PERIOD:08X}"
    return f"{PARAM_BOARD}##" + ch_off * 8


ser = serial.Serial(
    COM_PORT,
    115200,
    timeout=1.0,
    dsrdtr=False,
    rtscts=False
)

ser.setDTR(False)
ser.setRTS(False)

print("=" * 70)
print("严格按 NI I/O Trace 复刻 LabVIEW")
print("重点：发送波形头是 0#A40...，不是 0#M2A40...")
print("=" * 70)

time.sleep(3)
ser.reset_input_buffer()

try:
    # 清状态
    send(ser, f"{PARAM_BOARD}#E00", "关使能")
    send(ser, make_all_off_param(), "全零参数")
    time.sleep(0.5)
    ser.reset_input_buffer()

    # 握手
    send(ser, f"{PARAM_BOARD}#H", "握手")

    # 1. 设任意模式：按 NI Trace，是 0#M1，不带 A40
    send(ser, f"{M_BOARD}#M1", "0#M1 设任意模式")

    # 2. 发送波形头：按 NI Trace，是 0#A40001020303F3F3F
    send(ser, f"{M_BOARD}#{WAVE_HEADER}", "0#A40 波形头")

    # 3. 发送 4 行数据
    for i, line_tail in enumerate(DATA_LINES):
        send(ser, f"{M_BOARD}#{line_tail}", f"数据行 {i+1}/4", wait=0.05, read_window=1.0)
        time.sleep(0.01)

    # 4. 写 CH4 参数
    send(ser, make_ch4_param(), "1## CH4 参数")

    # 5. 使能 CH4
    send(ser, f"{PARAM_BOARD}#E08", "1#E08 使能CH4")

    print("\n观察：")
    print("如果 DATA_LINES = FLAT_LINES，CH4 应该接近静止。")
    print("如果仍然标准正弦，说明还有时序/终止符/目标板差异。")
    input("按回车继续 >>> ")

    print("\n现在切回正弦对照：")
    send(ser, f"{M_BOARD}#M0", "0#M0 正弦模式")

    print("观察：现在应该恢复标准正弦。")
    input("按回车停止 >>> ")

finally:
    print("\n安全停止")
    try:
        send(ser, f"{PARAM_BOARD}#E00", "关使能")
        send(ser, make_all_off_param(), "全零参数")
    except Exception as e:
        print("停止异常：", e)

    ser.close()
    print("串口已关闭")