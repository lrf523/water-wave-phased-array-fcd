import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import traceback
import json
import os
from fcd_backend_syl import FCDCore

class FCDApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FCD水波波场分析系统")
        self.root.geometry("820x800") # 稍微把窗口高度加一点点，从 720 改为 800
        
        self.config_file = os.path.join(os.path.dirname(__file__), "fcd_path_cache.json")
        
        # ===== 修改后：喇叭定标系统参数变量 =====
        self.cal_dir = tk.StringVar(value="")      # 独立定标图像路径
        self.cal_fps = tk.DoubleVar(value=30.0)    # 独立定标专用FPS
        self.cal_period = tk.DoubleVar(value=150.0) # 全局周期(ms)
        
        self.setup_ui()
        self.load_path_config()

    def setup_ui(self):
        # 1. 路径配置区
        path_frame = ttk.LabelFrame(self.root, text="文件与路径配置", padding=10)
        path_frame.pack(fill=tk.X, padx=10, pady=4)
        
        self.ref_var = tk.StringVar()
        self._add_path_row(path_frame, "参考图像 (静态):", self.ref_var, 0, is_dir=False)
        self.def_var = tk.StringVar()
        self._add_path_row(path_frame, "形变图像 (单帧):", self.def_var, 1, is_dir=False)
        self.seq_var = tk.StringVar()
        self._add_path_row(path_frame, "图片序列目录:", self.seq_var, 2, is_dir=True)
        self.out_var = tk.StringVar()
        self._add_path_row(path_frame, "输出/日志目录:", self.out_var, 3, is_dir=True)

        cfg_btn_frame = ttk.Frame(path_frame)
        cfg_btn_frame.grid(row=4, column=0, columnspan=3, pady=4, sticky="ew")
        ttk.Button(cfg_btn_frame, text="💾 保存当前全部配置(含高级参数)", command=self.save_path_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(cfg_btn_frame, text="📂 手动重载配置", command=self.load_path_config).pack(side=tk.LEFT, padx=5)

        # 2. 基础参数配置区
        self.param_frame = ttk.LabelFrame(self.root, text="基础物理与裁剪参数", padding=10)
        self.param_frame.pack(fill=tk.X, padx=10, pady=4)
        
        ttk.Label(self.param_frame, text="Crop (X1,X2,Y1,Y2):").grid(row=0, column=0, padx=5)
        self.crop_x1 = ttk.Entry(self.param_frame, width=6); self.crop_x1.grid(row=0, column=1)
        self.crop_x2 = ttk.Entry(self.param_frame, width=6); self.crop_x2.grid(row=0, column=2)
        self.crop_y1 = ttk.Entry(self.param_frame, width=6); self.crop_y1.grid(row=0, column=3)
        self.crop_y2 = ttk.Entry(self.param_frame, width=6); self.crop_y2.grid(row=0, column=4)
        
        ttk.Label(self.param_frame, text="水面深度 hw (mm):").grid(row=0, column=5, padx=15)
        self.water_depth_var = tk.StringVar(value="30.0")
        ttk.Entry(self.param_frame, textvariable=self.water_depth_var, width=8).grid(row=0, column=6)

        ttk.Label(self.param_frame, text="序列帧率(FPS):").grid(row=0, column=7, padx=10)
        self.seq_fps_var = tk.StringVar(value="30.0")
        ttk.Entry(self.param_frame, textvariable=self.seq_fps_var, width=6).grid(row=0, column=8)

        # 3. 高级菜单
        self.toggle_btn = ttk.Button(self.root, text="▶ 展开高级解调微调与序列解析面板", command=self.toggle_advanced_menu)
        self.toggle_btn.pack(fill=tk.X, padx=10, pady=4)

        self.adv_frame = ttk.LabelFrame(self.root, text="FCD 频域及图片序列高级微调面板", padding=10)
        
        # 规整化 Grid 布局与注释恢复
        ttk.Label(self.adv_frame, text="1. 低频抑制盲区半径(px):").grid(row=0, column=0, sticky="e", pady=2)
        self.adv_low_pass = ttk.Entry(self.adv_frame, width=8); self.adv_low_pass.grid(row=0, column=1, sticky="w")
        ttk.Label(self.adv_frame, text="消除水面宏观温漂导致的大面积红蓝倾斜", font=("", 8), foreground="dimgray").grid(row=0, column=2, sticky="w", padx=5)

        ttk.Label(self.adv_frame, text="2. 载波带通收紧因子(0-1):").grid(row=1, column=0, sticky="e", pady=2)
        self.adv_krad = ttk.Entry(self.adv_frame, width=8); self.adv_krad.grid(row=1, column=1, sticky="w")
        ttk.Label(self.adv_frame, text="消除图像上密集点状网络伪影。越小越平滑，过小会模糊", font=("", 8), foreground="dimgray").grid(row=1, column=2, sticky="w", padx=5)

        ttk.Label(self.adv_frame, text="3. 边界截除宽度(px):").grid(row=2, column=0, sticky="e", pady=2)
        self.adv_edge = ttk.Entry(self.adv_frame, width=8); self.adv_edge.grid(row=2, column=1, sticky="w")
        ttk.Label(self.adv_frame, text="消除FFT边界引起的延拓发散条纹", font=("", 8), foreground="dimgray").grid(row=2, column=2, sticky="w", padx=5)

        ttk.Label(self.adv_frame, text="4. 颜色显示极值截断(%):").grid(row=3, column=0, sticky="e", pady=2)
        p_frame = ttk.Frame(self.adv_frame); p_frame.grid(row=3, column=1, sticky="w")
        self.adv_p_low = ttk.Entry(p_frame, width=3); self.adv_p_low.pack(side=tk.LEFT)
        ttk.Label(p_frame, text="-").pack(side=tk.LEFT)
        self.adv_p_high = ttk.Entry(p_frame, width=3); self.adv_p_high.pack(side=tk.LEFT)
        ttk.Label(self.adv_frame, text="忽略极值噪点，对准真实水波提升对比度", font=("", 8), foreground="dimgray").grid(row=3, column=2, sticky="w", padx=5)

        ttk.Label(self.adv_frame, text="5. 动量箭头(步距/缩放):").grid(row=4, column=0, sticky="e", pady=2)
        q_frame = ttk.Frame(self.adv_frame); q_frame.grid(row=4, column=1, sticky="w")
        self.adv_qstep = ttk.Entry(q_frame, width=3); self.adv_qstep.pack(side=tk.LEFT)
        ttk.Label(q_frame, text="/").pack(side=tk.LEFT)
        self.adv_qscale = ttk.Entry(q_frame, width=3); self.adv_qscale.pack(side=tk.LEFT)
        ttk.Label(self.adv_frame, text="仅在勾选下方动量流时生效", font=("", 8), foreground="dimgray").grid(row=4, column=2, sticky="w", padx=5)

        # 🌟 规整化的 9 个序列输出项开关矩阵
        ttk.Label(self.adv_frame, text="6. 序列批量图窗导出项开关:").grid(row=5, column=0, sticky="ne", pady=8)
        seq_frm = ttk.Frame(self.adv_frame)
        seq_frm.grid(row=5, column=1, columnspan=2, sticky="w", pady=5)
        
        self.chk_hf = tk.BooleanVar(value=True); self.chk_disp = tk.BooleanVar(value=True); self.chk_sz = tk.BooleanVar(value=True)
        self.chk_amp = tk.BooleanVar(value=True); self.chk_ndisp = tk.BooleanVar(value=True); self.chk_s3d = tk.BooleanVar(value=True)
        self.chk_ph = tk.BooleanVar(value=True); self.chk_pa = tk.BooleanVar(value=True); self.chk_mom = tk.BooleanVar(value=True)

        ttk.Checkbutton(seq_frm, text="水位场(hfield)", variable=self.chk_hf).grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Checkbutton(seq_frm, text="三维位移场(disp)", variable=self.chk_disp).grid(row=0, column=1, sticky="w", padx=5, pady=2)
        ttk.Checkbutton(seq_frm, text="Z向自旋(sz)", variable=self.chk_sz).grid(row=0, column=2, sticky="w", padx=5, pady=2)
        ttk.Checkbutton(seq_frm, text="振幅(amplitude)", variable=self.chk_amp).grid(row=1, column=0, sticky="w", padx=5, pady=2)
        ttk.Checkbutton(seq_frm, text="归一化位移(norm)", variable=self.chk_ndisp).grid(row=1, column=1, sticky="w", padx=5, pady=2)
        ttk.Checkbutton(seq_frm, text="三维自旋(s3d)", variable=self.chk_s3d).grid(row=1, column=2, sticky="w", padx=5, pady=2)
        ttk.Checkbutton(seq_frm, text="相位(phase)", variable=self.chk_ph).grid(row=2, column=0, sticky="w", padx=5, pady=2)
        ttk.Checkbutton(seq_frm, text="彩色相幅(phaseamp)", variable=self.chk_pa).grid(row=2, column=1, sticky="w", padx=5, pady=2)
        ttk.Checkbutton(seq_frm, text="动量密度流(momentum)", variable=self.chk_mom).grid(row=2, column=2, sticky="w", padx=5, pady=2)

        # 4. 操作与日志区
        action_frame = ttk.LabelFrame(self.root, text="执行操作", padding=10)
        action_frame.pack(fill=tk.X, padx=10, pady=4)
        
        ttk.Button(action_frame, text="1. 获取像素", command=lambda: self.run_task(self._task_findpixels)).grid(row=0, column=0, padx=5, pady=2)
        ttk.Button(action_frame, text="2. 单帧分析", command=lambda: self.run_task(self._task_single)).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(action_frame, text="3. 序列分析", command=lambda: self.run_task(self._task_sequence)).grid(row=0, column=3, padx=5, pady=2)
        ttk.Button(action_frame, text="4. 交互测距", command=lambda: self.run_task(self._task_measure)).grid(row=0, column=4, padx=5, pady=2)
        ttk.Button(action_frame, text="5. Q值计算", command=lambda: self.run_task(self._task_qvalue)).grid(row=0, column=5, padx=5, pady=2)
        

        # 建立定标控制区 UI
        cal_frame = ttk.LabelFrame(self.root, text="32阶梯系统级定标", padding=10)
        cal_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(cal_frame, text="定标序列总目录:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        ttk.Entry(cal_frame, textvariable=self.cal_dir, width=40, state="readonly").grid(row=0, column=1, columnspan=3, sticky="w")
        ttk.Button(cal_frame, text="📂 浏览目录", command=lambda: self.cal_dir.set(filedialog.askdirectory(title="选择32组定标数据的总文件夹"))).grid(row=0, column=4, padx=5)
        
        ttk.Label(cal_frame, text="定标拍摄 FPS:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        ttk.Entry(cal_frame, textvariable=self.cal_fps, width=10).grid(row=1, column=1, sticky="w")
        
        ttk.Label(cal_frame, text="全局周期 (ms):").grid(row=1, column=2, padx=5, pady=5, sticky="e")
        ttk.Entry(cal_frame, textvariable=self.cal_period, width=10).grid(row=1, column=3, sticky="w")
        
        ttk.Button(cal_frame, text="开始全自动定标计算", command=lambda: self.run_task(self._task_calibrate)).grid(row=1, column=4, padx=5, ipadx=10)

        log_frame = ttk.LabelFrame(self.root, text="系统日志", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)
        self.log_txt = scrolledtext.ScrolledText(log_frame, state=tk.DISABLED, bg="#f4f4f4")
        self.log_txt.pack(fill=tk.BOTH, expand=True)
        self.log("系统就绪。高级调参机制已建立。")

    def toggle_advanced_menu(self):
        """🌟 菜单点击动态展开或收起"""
        if self.adv_frame.winfo_viewable():
            self.adv_frame.pack_forget()
            self.toggle_btn.config(text="▶ 展开微调参数菜单")
        else:
            # 动态插入到基础物理参数组件下方
            self.adv_frame.pack(fill=tk.X, padx=10, pady=4, after=self.param_frame)
            self.toggle_btn.config(text="▼ 收起高级解调微调参数面板")

    def _add_path_row(self, parent, label_text, var, row, is_dir=False):
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky="e", padx=5, pady=3)
        ttk.Entry(parent, textvariable=var, width=58).grid(row=row, column=1, padx=5)
        cmd = lambda: var.set(filedialog.askdirectory()) if is_dir else var.set(filedialog.askopenfilename())
        ttk.Button(parent, text="浏览...", command=cmd).grid(row=row, column=2, padx=5)

    def log(self, message):
        try:
            self.log_txt.config(state=tk.NORMAL)
            self.log_txt.insert(tk.END, message + "\n")
            self.log_txt.see(tk.END)
            self.log_txt.config(state=tk.DISABLED)
        except tk.TclError:
            # 捕获并忽略 GUI 已经被销毁时的日志写入错误
            print(f"后台运行中，界面已关闭: {message}")

    def save_path_config(self):
        # 将界面上存在的所有状态、路径、文本框数值、复选框统统打包
        config_data = {
            # 1. 核心路径
            "ref_path": getattr(self, 'ref_var', getattr(self, 'ref_path', None)).get() if hasattr(self, 'ref_var') or hasattr(self, 'ref_path') else "",
            "def_path": getattr(self, 'def_var', getattr(self, 'def_path', None)).get() if hasattr(self, 'def_var') or hasattr(self, 'def_path') else "",
            "seq_dir": getattr(self, 'seq_var', getattr(self, 'seq_dir', None)).get() if hasattr(self, 'seq_var') or hasattr(self, 'seq_dir') else "",
            "out_dir": getattr(self, 'out_var', getattr(self, 'out_dir', None)).get() if hasattr(self, 'out_var') or hasattr(self, 'out_dir') else "",
            
            # 2. 定标专用参数
            "cal_dir": self.cal_dir.get(),
            "cal_fps": self.cal_fps.get(),
            "cal_period": self.cal_period.get(),
            
            # 3. 基础物理与裁剪参数
            "water_depth": self.water_depth_var.get(),
            "seq_fps": self.seq_fps_var.get(),
            "crop_x1": self.crop_x1.get(),
            "crop_x2": self.crop_x2.get(),
            "crop_y1": self.crop_y1.get(),
            "crop_y2": self.crop_y2.get(),
            
            # 4. 高级微调面板参数
            "adv_low_pass": self.adv_low_pass.get(),
            "adv_krad": self.adv_krad.get(),
            "adv_edge": self.adv_edge.get(),
            "adv_p_low": self.adv_p_low.get(),
            "adv_p_high": self.adv_p_high.get(),
            "adv_qstep": self.adv_qstep.get(),
            "adv_qscale": self.adv_qscale.get(),
            
            # 5. 序列导出项 9个开关复选框
            "chk_hf": self.chk_hf.get(),
            "chk_disp": self.chk_disp.get(),
            "chk_sz": self.chk_sz.get(),
            "chk_amp": self.chk_amp.get(),
            "chk_ndisp": self.chk_ndisp.get(),
            "chk_s3d": self.chk_s3d.get(),
            "chk_ph": self.chk_ph.get(),
            "chk_pa": self.chk_pa.get(),
            "chk_mom": self.chk_mom.get()
        }
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=4)
            self.log("✅ 全部配置已成功保存！下次启动将自动恢复。")
        except Exception as e:
            self.log(f"❌ 保存配置失败: {e}")

    def load_path_config(self):
        try:
            cfg = {}
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                
            # 1. 核心路径恢复
            if "ref_path" in cfg and hasattr(self, 'ref_var'): self.ref_var.set(cfg["ref_path"])
            if "def_path" in cfg and hasattr(self, 'def_var'): self.def_var.set(cfg["def_path"])
            if "seq_dir" in cfg and hasattr(self, 'seq_var'): self.seq_var.set(cfg["seq_dir"])
            if "out_dir" in cfg and hasattr(self, 'out_var'): self.out_var.set(cfg["out_dir"])
            
            # 2. 定标专用参数恢复
            self.cal_dir.set(cfg.get("cal_dir", ""))
            self.cal_fps.set(cfg.get("cal_fps", 30.0))
            self.cal_period.set(cfg.get("cal_period", 150.0))
            
            # 3. 基础参数与裁剪恢复
            self.water_depth_var.set(cfg.get("water_depth", "30.0"))
            self.seq_fps_var.set(cfg.get("seq_fps", "30.0"))
            self._set_entry_val(self.crop_x1, cfg.get("crop_x1", "0"))
            self._set_entry_val(self.crop_x2, cfg.get("crop_x2", "0"))
            self._set_entry_val(self.crop_y1, cfg.get("crop_y1", "0"))
            self._set_entry_val(self.crop_y2, cfg.get("crop_y2", "0"))
            
            # 4. 高级参数面板恢复（即使没有缓存，也会利用 get 的机制填入最合理的默认值，防止输入框留白）
            self._set_entry_val(self.adv_low_pass, cfg.get("adv_low_pass", "65.0"))
            self._set_entry_val(self.adv_krad, cfg.get("adv_krad", "0.28"))
            self._set_entry_val(self.adv_edge, cfg.get("adv_edge", "10"))
            self._set_entry_val(self.adv_p_low, cfg.get("adv_p_low", "2.0"))
            self._set_entry_val(self.adv_p_high, cfg.get("adv_p_high", "98.0"))
            self._set_entry_val(self.adv_qstep, cfg.get("adv_qstep", "6"))
            self._set_entry_val(self.adv_qscale, cfg.get("adv_qscale", "4.0"))
            
            # 5. 序列导出项的开关恢复
            if "chk_hf" in cfg: self.chk_hf.set(cfg["chk_hf"])
            if "chk_disp" in cfg: self.chk_disp.set(cfg["chk_disp"])
            if "chk_sz" in cfg: self.chk_sz.set(cfg["chk_sz"])
            if "chk_amp" in cfg: self.chk_amp.set(cfg["chk_amp"])
            if "chk_ndisp" in cfg: self.chk_ndisp.set(cfg["chk_ndisp"])
            if "chk_s3d" in cfg: self.chk_s3d.set(cfg["chk_s3d"])
            if "chk_ph" in cfg: self.chk_ph.set(cfg["chk_ph"])
            if "chk_pa" in cfg: self.chk_pa.set(cfg["chk_pa"])
            if "chk_mom" in cfg: self.chk_mom.set(cfg["chk_mom"])
            
        except Exception as e:
            print(f"⚠️ 缓存配置读取异常: {e}")

    def _get_core(self):
        def safe_int(v, default=0):
            try: return int(v)
            except ValueError: return default
        def safe_float(v, default=0.0):
            try: return float(v)
            except ValueError: return default

        crop = (safe_int(self.crop_x1.get()), safe_int(self.crop_x2.get()), 
                safe_int(self.crop_y1.get()), safe_int(self.crop_y2.get()))
        
        return FCDCore(
            ref_path=self.ref_var.get(),
            def_path=self.def_var.get(),
            seq_dir=self.seq_var.get(),
            out_dir=self.out_var.get(),
            crop_pixels=crop,
            water_depth=safe_float(self.water_depth_var.get(), 30.0),
            fps=safe_float(self.seq_fps_var.get(), 30.0),
            low_pass_suppress=safe_float(self.adv_low_pass.get(), 65.0),
            krad_factor=safe_float(self.adv_krad.get(), 0.28),
            edge_width=safe_int(self.adv_edge.get(), 10),
            p_low=safe_float(self.adv_p_low.get(), 2.0),
            p_high=safe_float(self.adv_p_high.get(), 98.0),
            
            out_hf=self.chk_hf.get(), out_amp=self.chk_amp.get(), out_ph=self.chk_ph.get(),
            out_pa=self.chk_pa.get(), out_disp=self.chk_disp.get(), out_ndisp=self.chk_ndisp.get(),
            out_sz=self.chk_sz.get(), out_s3d=self.chk_s3d.get(), out_mom=self.chk_mom.get(),
            
            q_step=safe_int(self.adv_qstep.get(), 6),
            q_scale=safe_float(self.adv_qscale.get(), 4.0)
        )

    def _set_entry_val(self, entry_obj, val_str):
        entry_obj.delete(0, tk.END)
        entry_obj.insert(0, str(val_str))

    def run_task(self, task_func):
        try:
            task_func()
        except Exception as e:
            self.log(f"❌ 执行出错: {str(e)}\n{traceback.format_exc()}")

    def _task_findpixels(self):
        temp_core = self._get_core()
        temp_core.crop = (0, 0, 0, 0)
        pts, _ = temp_core.find_pixels()
        
        if len(pts) == 2:
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            
            # 后端已经通过视觉虚线框锁死了绝对正方形，这里直接填入即可
            self._set_entry_val(self.crop_x1, x1)
            self._set_entry_val(self.crop_x2, x2)
            self._set_entry_val(self.crop_y1, y1)
            self._set_entry_val(self.crop_y2, y2)
            
            side = x2 - x1
            self.log(f"获取坐标成功！\n已通过交互式红色虚线框截取正方形区域 ({side} x {side} px)。\n自动填入坐标 -> X: {x1}-{x2}, Y: {y1}-{y2}")

    def _task_single(self):
        self.log(self._get_core().analyze_single_frame())

    def _task_measure(self):
        dist, _ = self._get_core().measure_distance()
        if dist: self.log(f"测距成功: {dist:.3f} cm")

    def _task_qvalue(self):
        Q, _ = self._get_core().calculate_q_value()
        if Q: self.log(f"斯格明子 Q 值积分结果: {Q:.4f}")

    def _task_sequence(self):
        self.log("时序连续帧解析中...")
        self.root.update()
        out_p, _ = self._get_core().process_sequence()
        self.log(f"批处理完成。最大解析振幅已导出。")

    def _task_calibrate(self):
        try:
            fps = float(self.cal_fps.get())
            period = float(self.cal_period.get())
            calib_dir = self.cal_dir.get()
        except ValueError:
            self.log("❌ 参数输入错误，请确保输入的是数字！")
            return
            
        self.log(f"\n准备进行声学定标解析...\n参数 - FPS:{fps}, 周期:{period}ms")
        self.root.update()
        
        # 传入独立的目录和精简后的参数
        result_log = self._get_core().run_calibration(calib_dir, fps, period)
        self.log(result_log)

if __name__ == "__main__":
    root = tk.Tk()
    app = FCDApp(root)
    root.mainloop()