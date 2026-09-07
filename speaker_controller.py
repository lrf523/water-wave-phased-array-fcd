import serial
import time
import json  # 🌟 新增导入

class SpeakerArrayController:
    def __init__(self, port, baudrate=115200, timeout=1.0):
        self.calibration_data = None  # 用于存储定标参数字典
        try:
            # 🌟 修改 1：显式关闭硬件流控 (dsrdtr 和 rtscts)，防止触发单片机异常复位
            self.ser = serial.Serial(port, baudrate, timeout=timeout, dsrdtr=False, rtscts=False)
            
            # 🌟 修改 2：强制拉高/拉低电平，防止 Arduino/STM32 意外重启
            self.ser.setDTR(False)
            self.ser.setRTS(False)
            
            print(f"✅ 成功连接到端口 {port}，波特率 {baudrate}")
            
            print("⏳ 等待硬件底层及转发板初始化 (1.5秒)...")
            time.sleep(1.5) 
            
            # 🌟 修改 3：包裹缓存清空指令。如果芯片不支持，就安静地跳过，绝不报错崩溃
            try:
                self.ser.reset_input_buffer()
                self.ser.reset_output_buffer()
            except Exception as buffer_err:
                pass # 忽略不支持清空缓存的 USB 驱动报错

        except serial.SerialException as e:
            print(f"❌ 串口连接失败: {e}")
            self.ser = None

    def load_calibration(self, file_path):
        """🌟 新增方法：加载 JSON 定标文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.calibration_data = json.load(f)
            return True, f"定标补偿文件已加载"
        except Exception as e:
            self.calibration_data = None
            return False, f"定标文件解析失败: {e}"

    def _send_command(self, command_str, wait_time=0.1):
        # （保持原样不变）
        if not self.ser or not self.ser.is_open:
            return None
        try:
            self.ser.reset_input_buffer()
            full_command = (command_str + "\r\n").encode('ascii')
            self.ser.write(full_command)
            self.ser.flush()
            time.sleep(wait_time) 
            if self.ser.in_waiting > 0:
                return self._parse_response(self.ser.read(self.ser.in_waiting))
            return None
        except serial.SerialException as e:
            print(f"串口通信异常: {e}")
            return None

    def test_connection(self, board_id=0):
        return self._send_command(f"{board_id}#H")

    def write_waveform_params(self, board_id, params_list):
        if len(params_list) != 8:
            raise ValueError("必须提供完整的 8 个通道参数")
            
        payload_str = ""
        for i, p in enumerate(params_list):
            # 🛑 硬件安全限制 1：周期禁止为 0 或负数 (防烧毁)
            if p['period'] <= 0:
                raise ValueError(f"通道 {i+1} 的周期被设为了 {p['period']}，这可能烧毁硬件！周期必须大于 0。")

            # 🌟 定标前处理阶段：提取原始输入
            raw_amp = p['amp']
            raw_phase = p['phase']

            # 🛑 硬件安全限制 2：底层钳位 (防定标后超幅)
            # 无论定标系数多大，输入值都会被死死卡在 0.0 到 1.0 之间
            safe_input_amp = max(0.0, min(1.0, raw_amp)) 
            # 终极物理安全锁：乘以 0.2 的降幅系数，保证发送给硬件的值永不超 0.2
            actual_hardware_amp = safe_input_amp * 0.2  
            
            amp_hex = f"{int(actual_hardware_amp * 65535):04X}"
            
            safe_phase = max(0.0, min(1.0, raw_phase))
            phase_hex = f"{int(safe_phase * 65535):04X}"
            
            period_hex = f"{int(p['period']):08X}"
            
            payload_str += amp_hex + phase_hex + period_hex
            
        command_str = f"{board_id}##{payload_str}"
        return self._send_command(command_str, wait_time=0.2)

    def read_waveform_params(self, board_id):
        return self._send_command(f"{board_id}#?")

    def write_channel_enables(self, board_id, enables_list):
        if len(enables_list) != 8:
            raise ValueError("必须提供 8 个通道的使能状态")
            
        bitmask = 0
        for i, state in enumerate(enables_list):
            if state:
                bitmask |= (1 << i)
                
        command_str = f"{board_id}#E{bitmask:02X}"
        return self._send_command(command_str)

    def save_configuration(self, board_id):
        return self._send_command(f"{board_id}#W", wait_time=0.4)

    def reset_device(self, board_id):
        return self._send_command(f"{board_id}#RST", wait_time=0.5)

    def stop_all(self, board_id):
        """一键停止：强制发送所有通道振幅为 0 的波形，同时周期给定安全的 150"""
        params = [{"amp": 0.0, "phase": 0.0, "period": 150} for _ in range(8)]
        return self.write_waveform_params(board_id, params)

    # ================= 响应解析 =================

    def _parse_response(self, raw_data):
        if not raw_data:
            return {"status": False, "msg": "No response (返回为空)"}
            
        try:
            text = raw_data.decode('ascii', errors='ignore').strip()
            
            if '#' in text:
                core_text = text.split('#', 1)[1]
            else:
                core_text = text
                
            char_code = core_text[0] if len(core_text) > 0 else ''
            
            if char_code in ['H', '$']:
                return {"status": True, "msg": "通信握手成功", "raw": text}
            elif char_code == '#':
                return {"status": True, "msg": "参数写入成功 (待机静音)", "data": core_text[1:], "raw": text} 
            elif char_code == '?':
                return {"status": True, "msg": "参数写入成功 (喇叭输出中!)", "data": core_text[1:], "raw": text} 
            elif char_code == 'E':
                return {"status": True, "msg": "使能操作成功", "data": core_text[1:], "raw": text}
            else:
                return {"status": False, "msg": "未知响应", "raw": text}
                
        except Exception as e:
            return {"status": False, "msg": f"解析失败: {e}", "raw": raw_data}

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("串口已关闭")