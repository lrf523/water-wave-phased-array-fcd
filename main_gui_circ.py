import os
import json
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import traceback
import time       
import threading  
import serial.tools.list_ports  
import numpy as np

try:
    from speaker_controller import SpeakerArrayController
except ImportError:
    messagebox.showerror("导入错误", "找不到 speaker_controller.py，请确保它与此脚本在同一目录下！")
    exit()

try:
    from camera_controller import MindVisionCamera
except ImportError:
    MindVisionCamera = None

class LabviewMimicGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("水波阵列控制面板")
        self.root.geometry("1250x850") 
        
        self.controller = None
        self.camera = None 
        self.config_file = os.path.join(os.path.dirname(__file__), "hardware_param_cache.json")
        
        # 🌟 核心修改 1：为 3 块板子分配独立的 LUT 内存和路径变量
        self.calib_vars = [tk.StringVar() for _ in range(3)]
        self.lut_data_list = [None, None, None] 
        self._cancel_flag = False  
        
        # 🌟 新增：独立图片序列的拍摄时长控制变量 (默认 2.0 秒)
        self.camera_duration = tk.DoubleVar(value=2.0)

        self.setup_ui()
        self.load_hardware_config() 
        self.scan_ports() 

    def setup_ui(self):
        # ================= 顶部控制区 =================
        top_frame = ttk.LabelFrame(self.root, text="系统与操作控制", padding=10)
        top_frame.pack(fill=tk.X, padx=10, pady=5)

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

        ttk.Separator(top_frame, orient=tk.VERTICAL).grid(row=0, column=3, sticky="ns", padx=10)

        ttk.Label(top_frame, text="目标板子:").grid(row=0, column=4, padx=5)
        self.board_var = tk.StringVar(value="1")
        board_cb = ttk.Combobox(top_frame, textvariable=self.board_var, 
                                values=["1", "2", "3", "0 (所有)", "24通道 (1-3板)"], width=14, state="readonly")
        board_cb.grid(row=0, column=5, padx=5)
        board_cb.bind("<<ComboboxSelected>>", lambda e: self.build_param_matrix())

        ttk.Label(top_frame, text="操作类型:").grid(row=0, column=6, padx=5)
        self.op_var = tk.StringVar(value="1: 写入波形参数")
        op_values = ["0: 测试连接", "1: 写入波形参数", "2: 读取波形参数", "3: 写入通道使能", "4: 保存配置", "5: 设备复位"]
        op_cb = ttk.Combobox(top_frame, textvariable=self.op_var, values=op_values, width=16, state="readonly")
        op_cb.grid(row=0, column=7, padx=5)

        self.execute_btn = ttk.Button(top_frame, text="▶ 执行", command=self.execute_operation, state=tk.DISABLED)
        self.execute_btn.grid(row=0, column=8, padx=5)

        self.stop_btn = ttk.Button(top_frame, text="🛑 一键停止", command=self.stop_all_speakers, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=9, padx=10)

        # 定时工作时长与延迟设定区
        ttk.Label(top_frame, text="延迟触发(秒):").grid(row=1, column=4, padx=5, pady=2, sticky='e')
        self.delay_var = tk.DoubleVar(value=0.0)
        ttk.Entry(top_frame, textvariable=self.delay_var, width=8).grid(row=1, column=5, padx=5, sticky='w')

        ttk.Label(top_frame, text="工作时长(秒):").grid(row=1, column=6, padx=5, pady=2, sticky='e')
        self.duration_var = tk.DoubleVar(value=0.0)
        ttk.Entry(top_frame, textvariable=self.duration_var, width=8).grid(row=1, column=7, padx=5, sticky='w')
        ttk.Label(top_frame, text="(0为持续)", font=("", 8), foreground="dimgray").grid(row=1, column=8, sticky='w')

        ttk.Separator(top_frame, orient=tk.HORIZONTAL).grid(row=2, column=0, columnspan=10, sticky="ew", pady=5)
        
        # 🌟 核心修改 2：展开为 3 个独立的 LUT 文件加载槽
        for b in range(3):
            ttk.Label(top_frame, text=f"板{b+1} 定标:").grid(row=3+b, column=0, padx=5, pady=2, sticky='e')
            ttk.Entry(top_frame, textvariable=self.calib_vars[b], width=55, state="readonly").grid(row=3+b, column=1, columnspan=5, padx=5, sticky='w')
            ttk.Button(top_frame, text=f"📂 加载 板{b+1} 定标数据", command=lambda idx=b: self.load_calibration_file(idx)).grid(row=3+b, column=6, columnspan=2, padx=5, sticky='w')

        self.use_calib_var = tk.BooleanVar(value=True)
        self.use_calib_chk = ttk.Checkbutton(top_frame, text="全局启用定标补正", variable=self.use_calib_var, command=self.save_hardware_config)
        self.use_calib_chk.grid(row=4, column=8, columnspan=2, padx=10, sticky='w')

        ttk.Separator(top_frame, orient=tk.HORIZONTAL).grid(row=6, column=0, columnspan=10, sticky="ew", pady=5)

        # 同步轮播、喇叭数量与曝光控制
        ttk.Label(top_frame, text="自动定标系统").grid(row=7, column=0, padx=5, pady=5, sticky='e')
        
        # 🌟 增加定标喇叭数量输入框 (默认 8 通道)
        self.cal_spk_num_var = tk.IntVar(value=8)
        ttk.Label(top_frame, text="定标喇叭数:").grid(row=7, column=1, sticky='e')
        ttk.Entry(top_frame, textvariable=self.cal_spk_num_var, width=6, justify="center").grid(row=7, column=2, sticky='w', padx=(2, 15))
        
        self.carousel_btn = ttk.Button(top_frame, text="启动自动定标流程", command=self.start_carousel, state=tk.DISABLED)
        self.carousel_btn.grid(row=7, column=8, columnspan=2, padx=5, sticky='w', ipadx=10)

        ttk.Label(top_frame, text="曝光(ms, 0=自动):").grid(row=7, column=3, sticky='e', pady=5)
        self.cam_exp_var = tk.DoubleVar(value=10.0)
        ttk.Entry(top_frame, textvariable=self.cam_exp_var, width=6).grid(row=7, column=4, sticky='w')

        ttk.Label(top_frame, text="相机FPS:").grid(row=7, column=5, sticky='e')
        self.cam_fps_var = tk.DoubleVar(value=30.0)
        ttk.Entry(top_frame, textvariable=self.cam_fps_var, width=6).grid(row=7, column=6, sticky='w')

        # 波形全局参数
        ttk.Label(top_frame, text="波形全局参数:").grid(row=8, column=0, padx=5, pady=5, sticky='e')
        
        ttk.Label(top_frame, text="全局周期(ms):").grid(row=8, column=1, sticky='e')
        self.global_period_var = tk.IntVar(value=150)
        ttk.Entry(top_frame, textvariable=self.global_period_var, width=8).grid(row=8, column=2, sticky='w')

        ttk.Label(top_frame, text="全局相位(0-1):").grid(row=8, column=3, sticky='e')
        self.global_phase_var = tk.DoubleVar(value=0.0)
        ttk.Entry(top_frame, textvariable=self.global_phase_var, width=8).grid(row=8, column=4, sticky='w')

        # 路径保存与单张抓取
        ttk.Label(top_frame, text="图像保存目录:").grid(row=9, column=0, padx=5, pady=5, sticky='e')
        self.save_dir_var = tk.StringVar()
        ttk.Entry(top_frame, textvariable=self.save_dir_var, width=55, state="readonly").grid(row=9, column=1, columnspan=5, padx=5, sticky='w')
        
        self.save_dir_btn = ttk.Button(top_frame, text="📂 浏览目录", command=self.browse_save_dir)
        self.save_dir_btn.grid(row=9, column=4, padx=5, sticky='w')
        
        # 缩小单张采集的占位，留出空间
        self.snap_btn = ttk.Button(top_frame, text="📸 采集单张", command=self.capture_single_frame)
        self.snap_btn.grid(row=9, column=6, padx=5, sticky='w')

        # 🌟 新增：拍摄时长设置 与 序列采集按钮
        ttk.Label(top_frame, text="序列时长(s):").grid(row=9, column=7, sticky='e')
        self.cam_dur_var = tk.DoubleVar(value=2.0)
        ttk.Entry(top_frame, textvariable=self.cam_dur_var, width=5).grid(row=9, column=8, sticky='w')
        
        self.seq_btn = ttk.Button(top_frame, text="📸 采集序列", command=self.capture_image_sequence)
        self.seq_btn.grid(row=9, column=9, padx=5, sticky='w')

        # ================= 动态参数矩阵区容器 =================
        self.param_container = ttk.Frame(self.root)
        self.param_container.pack(fill=tk.X, padx=10, pady=5)
        
        self.build_param_matrix()

        # ================= 日志显示区 =================
        log_frame = ttk.LabelFrame(self.root, text="操作日志与串口返回", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=8, bg="#f4f4f4")
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log("界面初始化完成。请先连接串口与相机。")

    def build_param_matrix(self):
        for widget in self.param_container.winfo_children():
            widget.destroy()

        mode = self.board_var.get()
        num_channels = 24 if "24通道" in mode else 8
        num_boards = 3 if num_channels == 24 else 1

        self.param_frame = ttk.LabelFrame(self.param_container, text=f"波形参数与使能输入 ({num_channels} 通道)", padding=10)
        self.param_frame.pack(fill=tk.X, expand=True)

        self.enables_vars = []
        self.amp_vars = []
        self.phase_vars = []
        self.period_vars = []

        headers = ["通道使能", "幅度 (0-1)\n(振幅最大比例)", "相位 (0-1)", "周期 (ms)"]

        for b in range(num_boards):
            row_offset = b * 6
            if num_boards > 1:
                ttk.Label(self.param_frame, text=f"======== 硬件板 {b+1} (CH {b*8+1} ~ CH {b*8+8}) ========", 
                          font=("", 10, "bold"), foreground="#0055AA").grid(row=row_offset, column=0, columnspan=9, pady=(10 if b>0 else 0), sticky="w")
                row_offset += 1
            
            for i, h in enumerate(headers):
                ttk.Label(self.param_frame, text=h, font=("", 9, "bold")).grid(row=row_offset+i+1, column=0, padx=10, pady=2, sticky="e")

            for ch in range(8):
                global_ch = b * 8 + ch
                ttk.Label(self.param_frame, text=f"CH {global_ch+1}", font=("", 10, "bold")).grid(row=row_offset, column=ch+1, pady=2)
                
                en_var = tk.BooleanVar(value=True)
                ttk.Checkbutton(self.param_frame, variable=en_var, command=self.on_enable_toggle).grid(row=row_offset+1, column=ch+1)
                self.enables_vars.append(en_var)

                amp_var = tk.DoubleVar(value=1.0)
                ttk.Entry(self.param_frame, textvariable=amp_var, width=8, justify="center").grid(row=row_offset+2, column=ch+1, padx=5, pady=2)
                self.amp_vars.append(amp_var)

                phase_var = tk.DoubleVar(value=0.0)
                ttk.Entry(self.param_frame, textvariable=phase_var, width=8, justify="center").grid(row=row_offset+3, column=ch+1, padx=5, pady=2)
                self.phase_vars.append(phase_var)

                period_var = tk.IntVar(value=150)
                ttk.Entry(self.param_frame, textvariable=period_var, width=8, justify="center").grid(row=row_offset+4, column=ch+1, padx=5, pady=2)
                self.period_vars.append(period_var)
                
        if num_channels == 24: self.root.geometry("1250x1100")
        else: self.root.geometry("1250x850")

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
                self.log(f"❌ 编码单张图片失败")
                
        except Exception as e:
            self.log(f"❌ 采集单张图片异常: {e}")

    def capture_image_sequence(self):
        """🌟 新增：图片序列采集功能"""
        if not self.camera or not self.camera.is_opened:
            messagebox.showerror("错误", "请先连接并打开工业相机！")
            return
        
        target_dir = self.save_dir_var.get()
        if not target_dir or not os.path.exists(target_dir):
            messagebox.showerror("错误", "请先选择一个有效的图像保存主目录！")
            return
            
        try:
            # 1. 提取界面上的复用参数与新增时长参数
            target_fps = float(self.cam_fps_var.get())
            duration_sec = float(self.cam_dur_var.get())
            
            if duration_sec <= 0:
                messagebox.showerror("参数错误", "拍摄时长必须大于 0 秒！")
                return
                
            # 2. 复用并下发曝光时间
            exp_val = self.cam_exp_var.get()
            self.camera.set_exposure(exp_val)
            time.sleep(0.15) # 等待相机底层曝光寄存器生效
            
            # 3. 按照要求，在目标路径下新建带采集时间戳的独立文件夹
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            seq_dir = os.path.join(target_dir, f"Sequence_{timestamp}")
            os.makedirs(seq_dir, exist_ok=True)
            
            expected_frames = int(target_fps * duration_sec)
            self.log(f"⏳ 开始采集图像序列: 时长 {duration_sec}s, 帧率 {target_fps}FPS (预计约 {expected_frames} 帧)")
            self.root.update()
            
            # 4. 调用 camera_controller.py 原有的录制接口
            self.camera.start_recording(target_fps=target_fps, duration_sec=duration_sec)
            
            # 5. 阻塞主线程等待，保持界面响应
            while self.camera.is_recording:
                self.root.update()
                time.sleep(0.05)
                
            # 6. 调用原有的落盘接口，将图片写入刚新建的文件夹
            self.log(f"💾 抓取结束，正在将数据落盘至: Sequence_{timestamp} ...")
            self.root.update()
            self.camera.wait_and_save(save_dir=seq_dir, prefix="frame")
            
            self.log(f"✅ 图片序列采集并落盘成功！文件夹名: Sequence_{timestamp}")
            messagebox.showinfo("采集成功", f"成功保存图像序列！\n共计保存: {len(self.camera.frame_cache)} 帧\n保存位置:\n{seq_dir}")
            
        except Exception as e:
            self.log(f"❌ 采集图片序列异常: {e}")
            messagebox.showerror("采集失败", f"采集过程出错:\n{str(e)}")

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
            except Exception as e:
                self.log(f"❌ 相机连接失败: {e}")
                self.camera = None
        else:
            self.camera.close_camera()
            self.connect_cam_btn.config(text="📸连接相机")
            self.log("🔌 相机已安全断开。")

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
                self.log(f"❌ 连接失败。")
        else:
            self.controller.close()
            self.controller = None
            self.connect_btn.config(text="🔌连接串口")
            self.execute_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.DISABLED)
            self.carousel_btn.config(state=tk.DISABLED)
            self.log("🔌 串口已断开。")

    # 🌟 核心修改 3：独立的 LUT 文件导入逻辑
    def load_calibration_file(self, idx):
        file_path = filedialog.askopenfilename(title=f"选择 板{idx+1} 的 LUT 定标 JSON 文件", filetypes=[("JSON Files", "*.json")])
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
            "cam_dur": getattr(self, 'cam_dur_var', tk.DoubleVar(value=2.0)).get(), # 🌟 新增保存序列时长
            "global_period": self.global_period_var.get(),
            "global_phase": self.global_phase_var.get(),
            "save_dir": self.save_dir_var.get(),
            "target_board": self.board_var.get() 
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
            
            # 恢复三个 LUT 文件路径
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
            if "cam_dur" in cfg and hasattr(self, 'cam_dur_var'): self.cam_dur_var.set(cfg.get("cam_dur", 2.0)) # 🌟 恢复序列时长
            if "global_period" in cfg: self.global_period_var.set(cfg.get("global_period", 150))
            if "global_phase" in cfg: self.global_phase_var.set(cfg.get("global_phase", 0.0))
            
            s_dir = cfg.get("save_dir", "")
            if s_dir and os.path.exists(s_dir): self.save_dir_var.set(s_dir)
            else: self.save_dir_var.set(os.path.dirname(__file__))
            
            if "target_board" in cfg:
                self.board_var.set(cfg.get("target_board", "1"))
                self.build_param_matrix()
        except Exception as e:
            self.log(f"⚠️ 读取硬件历史缓存异常: {e}")

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

    # 🌟 核心修复：精确对口且【全局跨板锚定】的 LUT 映射引擎
    def _apply_lut_calibration(self, params, board_idx):
        if not self.use_calib_var.get(): return params 
        
        lut_data = self.lut_data_list[board_idx - 1]
        if not lut_data: return params 

        # 跨板全局振幅绝对对齐
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

            # 1. 振幅非线性逆插值映射 (统一参考绝对全局上限)
            target_amp_mm = p['amp'] * absolute_global_max
            req_v = np.interp(target_amp_mm, amp_out_arr, v_in_arr) 

            # 🌟 核心修改：彻底废除相位补偿逻辑！直接让最终下发相位等于用户输入的原始目标相位！
            final_phase = p['phase'] % 1.0

            calibrated.append({"amp": req_v, "phase": final_phase, "period": p['period']})
            
        return calibrated

    def start_carousel(self):
        if not self.controller: return
        self._cancel_flag = False 
        # 解除原单板限制，支持最高 24 通道的全自动跨板轮播
        threading.Thread(target=self._lut_carousel_thread, daemon=True).start()

    def _lut_carousel_thread(self):
        if not self.camera or not self.camera.is_opened:
            self.root.after(0, self.log, "❌ 错误：尚未连接工业相机！请先点击上方【连接相机】按钮。")
            self._cancel_flag = True
            return

        exp_val = self.cam_exp_var.get()
        self.camera.set_exposure(exp_val)
        time.sleep(0.15) 

        num_speakers = self.cal_spk_num_var.get()
        if num_speakers < 1 or num_speakers > 24:
            self.root.after(0, self.log, "❌ 错误：定标喇叭数量必须设置在 1 ~ 24 之间！")
            return

        self.root.after(0, self.log, "\n" + "="*50)
        self.root.after(0, self.log, f"🎬 开始执行【纯振幅】全局自动化声光同步定标！目标总喇叭数: {num_speakers} 个")
        self.root.after(0, self.log, "👉 核心卡点时序: 0.0s起振 -> 0.33s拍摄 -> 1.75s停震 -> 5.0s冷却落盘")
        self.root.after(0, self.log, "="*50)
        
        backup_calib = self.controller.calibration_data
        self.controller.calibration_data = None
        
        levels = [0.1, 0.4, 0.7, 1.0] 
        target_fps = self.cam_fps_var.get()
        global_period = self.global_period_var.get()
        global_phase = self.global_phase_var.get()
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        root_save_dir = self.save_dir_var.get()
        if not root_save_dir or not os.path.exists(root_save_dir):
            root_save_dir = os.path.dirname(__file__)
            
        base_dir = os.path.join(root_save_dir, f"Calibration_RAW_Total_{timestamp}")
        os.makedirs(base_dir, exist_ok=True)
        
        try:
            for global_ch in range(num_speakers):
                board_id = (global_ch // 8) + 1  
                local_ch = global_ch % 8         
                
                for lvl in levels:
                    if self._cancel_flag: break
                    
                    folder_name = f"CH{global_ch+1}_Amp{lvl:.1f}"
                    save_dir = os.path.join(base_dir, folder_name)
                    
                    self.root.after(0, self.log, f"🔊 [{folder_name}] 硬件板{board_id} CH{local_ch+1} 纯净打靶激发中...")

                    # 🌟 核心修改：彻底取消参考锚点，测哪路哪路发波，其余全部闭嘴，消灭所有空间声场干涉！
                    step_params = []
                    for i in range(8):
                        if i == local_ch:
                            step_params.append({"amp": lvl, "phase": global_phase, "period": global_period})
                        else:
                            step_params.append({"amp": 0.0, "phase": 0.0, "period": global_period})

                    base_time = time.perf_counter()
                    
                    # 下发控制指令
                    self.controller.write_waveform_params(board_id, step_params)
                    
                    while (time.perf_counter() - base_time) < 0.33:
                        if self._cancel_flag: break
                        time.sleep(0.01)
                    
                    if not self._cancel_flag:
                        self.camera.start_recording(target_fps=target_fps, duration_sec=1.5)
                    
                    while (time.perf_counter() - base_time) < 1.83:
                        if self._cancel_flag: break
                        time.sleep(0.01)
                        
                    self.controller.stop_all(board_id)

                    while (time.perf_counter() - base_time) < 1.83:
                        if self._cancel_flag: break
                        time.sleep(0.01)
                    
                    if self._cancel_flag: break
                    self.root.after(0, self.log, f"⏳ 进入 5s 恢复冷却...")

                    save_start = time.perf_counter()
                    self.camera.wait_and_save(save_dir, prefix=folder_name)
                    
                    elapsed = time.perf_counter() - save_start
                    remain = 5.0 - elapsed
                    if remain > 0:
                        time.sleep(remain)

            if not self._cancel_flag:
                self.root.after(0, self.log, f"✅ 全通道纯振幅定标轮播完毕！\n📁 数据保存在:\n{base_dir}")
            
            for b in [1, 2, 3]: self.controller.stop_all(b)

        except Exception as e:
            self.root.after(0, self.log, f"⛔ 同步轮播异常: {e}")
        finally:
            self.controller.calibration_data = backup_calib

    def _timed_execution_thread(self, mode, params, delay, duration):
        if delay > 0:
            t = 0
            while t < delay:
                if self._cancel_flag: return
                time.sleep(0.1)
                t += 0.1
        if self._cancel_flag: return
        
        try:
            # 🌟 核心修改 5：延时执行时也将参数严格切片，并分别进行独立 LUT 映射
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
                    params.append({"amp": final_amp, "phase": self.phase_vars[i].get(), "period": self.period_vars[i].get()})
                
                delay, duration = self.delay_var.get(), self.duration_var.get()
                if delay > 0 or duration > 0:
                    self._cancel_flag = False 
                    threading.Thread(target=self._timed_execution_thread, args=(mode, params, delay, duration), daemon=True).start()
                else:
                    # 🌟 核心修改 6：立即执行时参数切片与独立映射并发下发
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
                    r1 = self.controller.save_configuration(1); r2 = self.controller.save_configuration(2); r3 = self.controller.save_configuration(3)
                    self.log(f"保存配置返回-> 板1:[{r1}] 板2:[{r2}] 板3:[{r3}]")
                else:
                    b_id = int(mode.split()[0]) if "所有" not in mode else 0
                    resp = self.controller.save_configuration(b_id)
                    self.log(f"保存配置返回: {resp}")
                    
            elif op_id == 5:
                if "24通道" in mode:
                    r1 = self.controller.reset_device(1); r2 = self.controller.reset_device(2); r3 = self.controller.reset_device(3)
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

if __name__ == "__main__":
    root = tk.Tk()
    app = LabviewMimicGUI(root)
    def on_closing():
        if app.controller: app.controller.close()
        if app.camera: app.camera.close_camera() 
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()