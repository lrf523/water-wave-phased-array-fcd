import os
import cv2
import json
import numpy as np
import cupy as cp  # 🌟 新增：GPU 算力引擎
from datetime import datetime
import math

# 强行切换到 Agg 后端
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.path import Path  

# 100% 确保支持中文字符与负号渲染
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

class FCDCore:
    def __init__(self, ref_path, def_path=None, seq_dir=None, out_dir=None, crop_pixels=(0,0,0,0), 
                 water_depth=30.0, low_pass_suppress=65.0, krad_factor=0.28, edge_width=10, 
                 p_low=2, p_high=98, 
                 out_hf=True, out_amp=True, out_ph=True, out_pa=True, 
                 out_disp=True, out_ndisp=True, out_sz=True, out_s3d=True, out_mom=True,
                 q_step=6, q_scale=4.0):
        self.ref_path = ref_path
        self.def_path = def_path
        self.seq_dir = seq_dir
        self.out_dir = out_dir if out_dir else os.getcwd()
        self.crop = crop_pixels 
        
        self.H = (water_depth + 0.894 * 10.0) * 0.25
        
        self.low_pass_suppress_r = low_pass_suppress
        self.krad_factor = krad_factor
        self.edge = edge_width
        self.p_low = p_low
        self.p_high = p_high
        
        # 🌟 新增：9 个独立的序列图窗输出开关
        self.out_hf = out_hf
        self.out_amp = out_amp
        self.out_ph = out_ph
        self.out_pa = out_pa
        self.out_disp = out_disp
        self.out_ndisp = out_ndisp
        self.out_sz = out_sz
        self.out_s3d = out_s3d
        self.out_mom = out_mom
        
        self.q_step = q_step
        self.q_scale = q_scale
        
        self.log_dir = os.path.join(self.out_dir, "logs")
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def write_log(self, prefix, content):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = os.path.join(self.log_dir, f"Log_{prefix}_{timestamp}.txt")
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return log_path

    def _read_and_crop(self, path):
        img_data = np.fromfile(path, dtype=np.uint8)
        if img_data.size == 0:
            raise FileNotFoundError(f"文件不存在或为空: {path}")
        img = cv2.imdecode(img_data, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法解码图像: {path}")
        x1, x2, y1, y2 = self.crop
        if x2 > x1 and y2 > y1:
            return img[y1:y2, x1:x2].astype(float)
        return img.astype(float)

    # ================= 🌟 GPU 核心数学算子开始 =================
    def _fftinvgrad_gpu(self, u, v):
        rows, cols = u.shape
        kx = cp.fft.fftfreq(cols) * 2 * cp.pi
        ky = cp.fft.fftfreq(rows) * 2 * cp.pi
        KX, KY = cp.meshgrid(kx, ky)
        K2 = KX**2 + KY**2
        K2[0, 0] = 1.0
        Fu = cp.fft.fft2(u)
        Fv = cp.fft.fft2(v)
        Fh = (-1j * KX * Fu - 1j * KY * Fv) / K2
        Fh[0, 0] = 0.0
        return cp.real(cp.fft.ifft2(Fh))

    def _find_orth_carrier_pks_gpu(self, Iref_gpu):
        rows, cols = Iref_gpu.shape
        F = cp.fft.fftshift(cp.fft.fft2(Iref_gpu))
        F_mag = cp.abs(F)
        cy, cx = rows // 2, cols // 2
        
        x = cp.arange(cols); y = cp.arange(rows)
        X, Y = cp.meshgrid(x, y)
        S = min(rows, cols)
        X_iso = (X - cx) * (S / cols)
        Y_iso = (Y - cy) * (S / rows)
        
        low_pass_suppress = 1.0 - cp.exp(-(X_iso**2 + Y_iso**2) / (2.0 * (self.low_pass_suppress_r**2)))
        F_mag_filtered = F_mag * low_pass_suppress
        
        y1, x1 = cp.unravel_index(cp.argmax(F_mag_filtered), F_mag_filtered.shape)
        kr = cp.array([y1 - cy, x1 - cx])
        kr_iso = cp.array([(y1 - cy) * (S / rows), (x1 - cx) * (S / cols)])
        
        F_mag_filtered[(X_iso - kr_iso[1])**2 + (Y_iso - kr_iso[0])**2 <= 55**2] = 0
        y1_sym, x1_sym = 2*cy - y1, 2*cx - x1
        kr_iso_sym = cp.array([(y1_sym - cy) * (S / rows), (x1_sym - cx) * (S / cols)])
        F_mag_filtered[(X_iso - kr_iso_sym[1])**2 + (Y_iso - kr_iso_sym[0])**2 <= 55**2] = 0
        
        y2, x2 = cp.unravel_index(cp.argmax(F_mag_filtered), F_mag_filtered.shape)
        ku = cp.array([y2 - cy, x2 - cx])
        ku_iso = cp.array([(y2 - cy) * (S / rows), (x2 - cx) * (S / cols)])
        
        krad_iso = (cp.sqrt(cp.sum((kr_iso - ku_iso)**2)) / 2.0) * self.krad_factor
        return kr, ku, krad_iso

    def _fcd_demodulate_correct_gpu(self, Iref_gpu, Idef_gpu):
        rows, cols = Iref_gpu.shape
        Fref = cp.fft.fftshift(cp.fft.fft2(Iref_gpu))
        Fdef = cp.fft.fftshift(cp.fft.fft2(Idef_gpu))
        
        kr, ku, krad_iso = self._find_orth_carrier_pks_gpu(Iref_gpu)
        cy, cx = rows // 2, cols // 2
        x = cp.arange(cols); y = cp.arange(rows)
        X, Y = cp.meshgrid(x, y)
        S = min(rows, cols)
        X_iso = (X - cx) * (S / cols)
        Y_iso = (Y - cy) * (S / rows)
        
        def extract_carrier_signal(F_img, peak):
            peak_iso = cp.array([peak[0] * (S / rows), peak[1] * (S / cols)])
            dist2 = (X_iso - peak_iso[1])**2 + (Y_iso - peak_iso[0])**2
            gauss_filter = cp.exp(-dist2 / (2.0 * (krad_iso**2)))
            F_filtered = F_img * gauss_filter
            F_shifted = cp.roll(F_filtered, (int(-peak[0]), int(-peak[1])), axis=(0, 1))
            return cp.fft.ifft2(cp.fft.ifftshift(F_shifted))

        cr_ref = extract_carrier_signal(Fref, kr)
        cr_def = extract_carrier_signal(Fdef, kr)
        cu_ref = extract_carrier_signal(Fref, ku)
        cu_def = extract_carrier_signal(Fdef, ku)
        
        psi_r = cr_def * cp.conj(cr_ref)
        psi_u = cu_def * cp.conj(cu_ref)
        
        def complex_gradient(psi):
            d_dy = cp.zeros(psi.shape, dtype=cp.float32)
            d_dy[:-1, :] = cp.angle(psi[1:, :] * cp.conj(psi[:-1, :]))
            d_dy[-1, :] = d_dy[-2, :]
            d_dx = cp.zeros(psi.shape, dtype=cp.float32)
            d_dx[:, :-1] = cp.angle(psi[:, 1:] * cp.conj(psi[:, :-1]))
            d_dx[:, -1] = d_dx[:, -2]
            return d_dy, d_dx

        dr_dy, dr_dx = complex_gradient(psi_r)
        du_dy, du_dx = complex_gradient(psi_u)
        
        u = dr_dx + du_dx
        v = dr_dy + du_dy
        return u, v

    def process_single_frame(self):
        Iref = self._read_and_crop(self.ref_path)
        Idef = self._read_and_crop(self.def_path)
        
        # 🌟 将数据推送到显存 (VRAM)
        Iref_gpu = cp.array(Iref, dtype=cp.float32)
        Idef_gpu = cp.array(Idef, dtype=cp.float32)
        
        # 在显卡上光速计算
        u_gpu, v_gpu = self._fcd_demodulate_correct_gpu(Iref_gpu, Idef_gpu)
        h_gpu = self._fftinvgrad_gpu(-u_gpu, -v_gpu) / self.H
        
        # 将结果拉回系统内存 (RAM) 供画图使用
        h = cp.asnumpy(h_gpu)
        u = cp.asnumpy(u_gpu)
        v = cp.asnumpy(v_gpu)
        e = self.edge
        
        # 释放显存
        cp.get_default_memory_pool().free_all_blocks()
        return h[e:-e, e:-e], u[e:-e, e:-e], v[e:-e, e:-e], Idef

    def _estimate_mm_per_pixel(self, Iref, G=1.5):
        Iref_gpu = cp.array(Iref, dtype=cp.float32)
        kr_gpu, ku_gpu, _ = self._find_orth_carrier_pks_gpu(Iref_gpu)
        kr = cp.asnumpy(kr_gpu)
        ku = cp.asnumpy(ku_gpu)
        rows, cols = Iref.shape
        kr_phys = np.array([kr[1] / cols, kr[0] / rows]) * 2.0 * np.pi
        ku_phys = np.array([ku[1] / cols, ku[0] / rows]) * 2.0 * np.pi
        k0 = kr_phys + ku_phys
        kmag = np.linalg.norm(k0)
        return kmag * G / (2.0 * np.pi) if kmag > 0 else 0.12
    # ================= 🌟 GPU 核心数学算子结束 =================

    def _set_dynamic_ticks(self, ax, shape, mm_per_pixel):
        cm_per_pixel = mm_per_pixel / 10.0
        h_px, w_px = shape
        width_cm = w_px * cm_per_pixel
        height_cm = h_px * cm_per_pixel
        
        nice_steps = [1, 2, 5, 10, 20, 50, 100]
        target_step = max(width_cm, height_cm) / 6.0
        step_size = next((s for s in nice_steps if s >= target_step), nice_steps[-1])
        
        # 🌟 修复核心1：移除 math.ceil，确保刻度严格在图像物理尺寸内部，绝不越界撑大画框
        x_cm_main = np.arange(0, width_cm + 1e-5, step_size)
        y_cm_main = np.arange(0, height_cm + 1e-5, step_size)
        
        ax.set_xticks(x_cm_main / cm_per_pixel)
        ax.set_yticks(y_cm_main / cm_per_pixel)
        ax.set_xticklabels([f"{x:.0f}" for x in x_cm_main])
        ax.set_yticklabels([f"{y:.0f}" for y in y_cm_main])
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        ax.tick_params(which='major', length=6, labelsize=8)
        ax.tick_params(which='minor', length=3)
        ax.set_xlabel('X (cm)', fontsize=9)
        ax.set_ylabel('Y (cm)', fontsize=9)
        # (移除了强制的 set_xlim 和 set_ylim，让 imshow 自带的边界完美贴合数据)

    def _set_colorbar_ticks(self, cbar, data, mm_per_pixel, vmin, vmax, label):
        phys_min, phys_max = vmin * mm_per_pixel, vmax * mm_per_pixel
        phys_range = max(1e-5, phys_max - phys_min)
        nice_steps = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
        step_size = next((s for s in nice_steps if s >= phys_range / 5.0), nice_steps[-1])
        start_val = math.ceil(phys_min / step_size) * step_size
        end_val = math.floor(phys_max / step_size) * step_size
        phys_ticks = np.arange(start_val, end_val + step_size*0.1, step_size)
        cbar.set_ticks(phys_ticks / mm_per_pixel)
        fmt = "{:.0f}" if step_size >= 1 else ("{:.1f}" if step_size >= 0.1 else "{:.2f}")
        cbar.set_ticklabels([fmt.format(x) for x in phys_ticks])
        cbar.set_label(label, fontsize=10, labelpad=10)

    # 🌟 修改 1：增加 title_text 参数，在右侧图例上方补充说明绘图内容的标题
    def _draw_2d_hsv_wheel(self, ax, title_text):
        res = 150
        x = np.linspace(-1, 1, res)
        y = np.linspace(-1, 1, res)
        X, Y = np.meshgrid(x, y)
        rho = np.sqrt(X**2 + Y**2)
        phi = np.arctan2(Y, X)
        
        hsv = np.zeros((res, res, 3))
        hsv[..., 0] = (phi + np.pi) / (2 * np.pi) # 角度对应色相
        hsv[..., 1] = 1.0                         # 饱和度饱和
        hsv[..., 2] = np.where(rho <= 1.0, rho, 0.0) # 半径大小对应亮度强度，圆外裁剪掉
        
        rgb = plt.matplotlib.colors.hsv_to_rgb(hsv)
        ax.imshow(rgb, extent=[-1, 1, -1, 1])
        ax.axis('off')
        
        # 在右侧图例区域添加明确的标题与物理含义说明
        ax.set_title(f"{title_text}\n\n【2D HSV 图例】\n色相(角度): 位移方向\n明度(半径): 位移强度", 
                     fontsize=10, pad=10, fontweight='bold', loc='center')

    def analyze_single_frame(self):
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt

        h, u, v, _ = self.process_single_frame()
        Iref = self._read_and_crop(self.ref_path)
        mm_per_pixel = self._estimate_mm_per_pixel(Iref)
        base_name = os.path.splitext(os.path.basename(self.def_path))[0]
        shape = h.shape

        h_vmin, h_vmax = np.percentile(h, self.p_low), np.percentile(h, self.p_high)
        
        # 🌟 修复核心2：防噪声底噪放大机制。如果起伏极小(<0.2mm)，拒绝拉伸对比度
        if h_vmax - h_vmin < 0.2:
            h_center = (h_vmax + h_vmin) / 2.0
            h_vmin, h_vmax = h_center - 0.1, h_center + 0.1

        # ================= 图 1: 水位场 =================
        fig1, ax1 = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
        im1 = ax1.imshow(h, cmap='jet', vmin=h_vmin, vmax=h_vmax)
        ax1.set_aspect('equal')
        self._set_dynamic_ticks(ax1, shape, mm_per_pixel)
        cbar1 = fig1.colorbar(im1, ax=ax1, shrink=0.85, aspect=25)
        self._set_colorbar_ticks(cbar1, h, mm_per_pixel, h_vmin, h_vmax, label='水位高度形变 (mm)')
        fig1.savefig(os.path.join(self.out_dir, f"{base_name}height.png"), dpi=300, bbox_inches='tight', pad_inches=0.02)
        plt.close(fig1)

        np.savetxt(os.path.join(self.out_dir, f"{base_name}_height_mm.csv"), h * mm_per_pixel, delimiter=",")

        u_real, v_real = np.real(u), np.real(v)
        phase = np.arctan2(v_real, -u_real)
        phase_norm = (phase + np.pi) / (2 * np.pi)
        h_norm = np.clip((h - h_vmin) / (h_vmax - h_vmin + 1e-10), 0, 1)

        # ================= 图 2: 三维矢量位移场 =================
        fig2, (ax2, ax_w2) = plt.subplots(1, 2, figsize=(7.5, 4.8), gridspec_kw={'width_ratios': [4, 1]}, constrained_layout=True)
        hsv2 = np.zeros((shape[0], shape[1], 3))
        hsv2[..., 0] = phase_norm
        hsv2[..., 1] = 1 - 4 * (h_norm - 0.5)**2
        hsv2[..., 2] = h_norm
        ax2.imshow(plt.matplotlib.colors.hsv_to_rgb(hsv2))
        ax2.set_aspect('equal')
        self._set_dynamic_ticks(ax2, shape, mm_per_pixel)
        self._draw_2d_hsv_wheel(ax_w2, title_text="三维矢量位移场")
        fig2.savefig(os.path.join(self.out_dir, f"{base_name}disp.png"), dpi=300, bbox_inches='tight', pad_inches=0.02)
        plt.close(fig2)

        # ================= 图 3: 归一化位移场 =================
        fig3, (ax3, ax_w3) = plt.subplots(1, 2, figsize=(7.5, 4.8), gridspec_kw={'width_ratios': [4, 1]}, constrained_layout=True)
        disp_norm_val = (h / np.sqrt(u_real**2 + v_real**2 + h**2 + 1e-10) + 1) / 2
        dn_vmin, dn_vmax = np.percentile(disp_norm_val, self.p_low), np.percentile(disp_norm_val, self.p_high)
        
        # 为归一化场同样增加底噪保护机制
        if dn_vmax - dn_vmin < 0.1:
            dn_center = (dn_vmax + dn_vmin) / 2.0
            dn_vmin, dn_vmax = max(0, dn_center - 0.05), min(1, dn_center + 0.05)
            
        disp_norm = np.clip((disp_norm_val - dn_vmin) / (dn_vmax - dn_vmin + 1e-10), 0, 1)
        
        hsv3 = np.zeros((shape[0], shape[1], 3))
        hsv3[..., 0] = phase_norm
        hsv3[..., 1] = 1 - 4 * (disp_norm - 0.5)**2
        hsv3[..., 2] = disp_norm
        ax3.imshow(plt.matplotlib.colors.hsv_to_rgb(hsv3))
        ax3.set_aspect('equal')
        self._set_dynamic_ticks(ax3, shape, mm_per_pixel)
        self._draw_2d_hsv_wheel(ax_w3, title_text="归一化位移场")
        fig3.savefig(os.path.join(self.out_dir, f"{base_name}dispNorm.png"), dpi=300, bbox_inches='tight', pad_inches=0.02)
        plt.close(fig3)

        log_content = f"===== 处理日志 =====\n处理时间: {datetime.now().strftime('%Y%m%d_%H%M%S')}\n"
        log_content += f"自动计算 H 因子 = {self.H:.4f}\n"
        log_content += f"高级配置项: low_pass_suppress={self.low_pass_suppress_r}, krad_factor={self.krad_factor}, edge_width={self.edge}\n"
        log_file = self.write_log(f"1Frame_{base_name}", log_content)
        return f"单帧分析完成！\n结果保存路径:\n{self.out_dir}"

    def find_pixels(self):
        matplotlib.use('TkAgg', force=True)
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        img = self._read_and_crop(self.ref_path)
        img_height, img_width = img.shape[:2]

        fig, ax = plt.subplots(num='点击图片获取坐标')
        ax.imshow(img, cmap='gray')
        ax.set_title('【单击】定起点 → 【移动】拉出红色虚线正方形 → 【再次单击】确认截取', fontsize=11)
        plt.axis('on')

        # 状态字典加入 'bg' 用于存储背景图像缓存（Blitting 提速关键）
        state = {'start': None, 'rect': None, 'bg': None}
        points = []

        def get_square_coords(start_x, start_y, curr_x, curr_y):
            """计算以起点为基准的最大正方形"""
            dx = curr_x - start_x
            dy = curr_y - start_y
            
            # 🌟 修复 1：改为 max，使正方形的边始终“贴”在鼠标移动较远的那一维上
            side = max(abs(dx), abs(dy))
            sign_x = 1 if dx > 0 else -1
            sign_y = 1 if dy > 0 else -1
            return side, sign_x, sign_y

        def onmove(event):
            # 如果没定起点，或背景未缓存，或鼠标移出坐标轴，则不处理
            if not state['start'] or not state['bg'] or event.inaxes != ax:
                return
            
            side, sign_x, sign_y = get_square_coords(state['start'][0], state['start'][1], event.xdata, event.ydata)
            state['rect'].set_width(side * sign_x)
            state['rect'].set_height(side * sign_y)
            
            # 🌟 修复 2：Blit 局部高速重绘技术。恢复背景 -> 仅画红框 -> 刷新局部屏幕
            fig.canvas.restore_region(state['bg'])
            ax.draw_artist(state['rect'])
            fig.canvas.blit(ax.bbox)

        def onclick(event):
            if event.inaxes != ax or event.button != 1:
                return

            if not state['start']:
                # 第一次点击：确定起点
                x, y = event.xdata, event.ydata
                state['start'] = (x, y)
                
                # 生成红色虚线框初始对象
                state['rect'] = patches.Rectangle((x, y), 0, 0, linewidth=1.5, edgecolor='red', facecolor='none', linestyle='--')
                ax.add_patch(state['rect'])
                ax.plot(x, y, 'r+', markersize=10, linewidth=1.5)
                
                # 画完初始十字准星后，截取当前纯净的背景（用于后续高速滑动刷新）
                fig.canvas.draw()
                state['bg'] = fig.canvas.copy_from_bbox(ax.bbox)
            else:
                # 第二次点击：固定正方形并结束
                side, sign_x, sign_y = get_square_coords(state['start'][0], state['start'][1], event.xdata, event.ydata)
                
                # 防止原地误触双击导致崩溃
                if side < 5: return

                end_x = state['start'][0] + side * sign_x
                end_y = state['start'][1] + side * sign_y

                # 约束坐标不要超出图像物理边界
                start_x_c = max(0, min(img_width-1, state['start'][0]))
                start_y_c = max(0, min(img_height-1, state['start'][1]))
                end_x_c = max(0, min(img_width-1, end_x))
                end_y_c = max(0, min(img_height-1, end_y))

                points.append((int(round(start_x_c)), int(round(start_y_c))))
                points.append((int(round(end_x_c)), int(round(end_y_c))))
                plt.close(fig)

        def onkey(event):
            if event.key in ['enter', 'escape']:
                plt.close(fig)

        # 绑定事件
        fig.canvas.mpl_connect('motion_notify_event', onmove)
        fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('key_press_event', onkey)
        plt.show()

        matplotlib.use('Agg', force=True)
        return points, self.write_log("findpixel", f"采集点数: {len(points)}")

    def measure_distance(self):
        matplotlib.use('TkAgg', force=True)
        import matplotlib.pyplot as plt
        h, _, _, _ = self.process_single_frame()
        Iref = self._read_and_crop(self.ref_path)
        mm_per_pixel = self._estimate_mm_per_pixel(Iref)
        fig, ax = plt.subplots(num="测距模式: 左键选择两点，回车结束")
        ax.imshow(h, cmap='jet')
        pts = plt.ginput(n=2, timeout=0)
        plt.close(fig)
        matplotlib.use('Agg', force=True)
        if len(pts) == 2:
            dist_px = np.sqrt((pts[1][0]-pts[0][0])**2 + (pts[1][1]-pts[0][1])**2)
            dist_cm = (dist_px * mm_per_pixel) / 10.0
            return dist_cm, self.write_log("measure", f"距离: {dist_cm:.3f} cm")
        return None, None

    def calculate_q_value(self):
        matplotlib.use('TkAgg', force=True)
        import matplotlib.pyplot as plt
        from matplotlib.widgets import PolygonSelector
        h, u, v, _ = self.process_single_frame()
        R_mag = np.sqrt(u**2 + v**2 + h**2 + 1e-10)
        un, vn, hn = u/R_mag, v/R_mag, h/R_mag
        du_dy, du_dx = np.gradient(un)
        dv_dy, dv_dx = np.gradient(vn)
        dh_dy, dh_dx = np.gradient(hn)
        integrand = np.zeros_like(u)
        for i in range(u.shape[0]):
            for j in range(u.shape[1]):
                dR_dx = np.array([du_dx[i,j], dv_dx[i,j], dh_dx[i,j]])
                dR_dy = np.array([du_dy[i,j], dv_dy[i,j], dh_dy[i,j]])
                integrand[i,j] = np.dot(np.array([un[i,j], vn[i,j], hn[i,j]]), np.cross(dR_dx, dR_dy))
        fig, ax = plt.subplots(num="框选区域计算 Q 值")
        ax.imshow(h, cmap='jet')
        poly_pts = []
        def onselect(verts):
            poly_pts.clear()
            poly_pts.extend(verts)
        selector = PolygonSelector(ax, onselect)
        plt.show()
        matplotlib.use('Agg', force=True)
        if len(poly_pts) < 3: return None, None
        x, y = np.meshgrid(np.arange(u.shape[1]), np.arange(u.shape[0]))
        mask = Path(poly_pts).contains_points(np.vstack((x.flatten(), y.flatten())).T).reshape(u.shape)
        Q = np.sum(integrand[mask]) / (4 * np.pi)
        return Q, self.write_log("Qvalue", f"Q值: {Q:.4f}")

    def _hilbert_gpu(self, x_stack):
        """专为显卡编写的高速 Hilbert 解析信号提取函数"""
        N = x_stack.shape[0]
        Xf = cp.fft.fft(x_stack, axis=0)
        h = cp.zeros(N, dtype=Xf.dtype)
        if N % 2 == 0:
            h[0] = h[N // 2] = 1
            h[1:N // 2] = 2
        else:
            h[0] = 1
            h[1:(N + 1) // 2] = 2
        h = h[:, cp.newaxis, cp.newaxis]
        return cp.fft.ifft(Xf * h, axis=0)

    def process_sequence(self):
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt

        if not self.seq_dir or not os.path.exists(self.seq_dir):
            raise ValueError("图片序列目录无效或不存在！")

        files = sorted([f for f in os.listdir(self.seq_dir) if f.endswith(('.bmp', '.tiff', '.png', '.jpg'))])
        if not files: raise ValueError("没有找到有效图像帧")
        
        seq_out_dir = os.path.join(self.out_dir, os.path.basename(self.seq_dir) + "_results")
        os.makedirs(seq_out_dir, exist_ok=True)
        
        subdirs = []
        if self.out_hf: subdirs.append('hfield')
        if self.out_amp: subdirs.append('amplitude')
        if self.out_ph: subdirs.append('phase')
        if self.out_pa: subdirs.append('phaseamp')
        if self.out_disp: subdirs.append('displacement')
        if self.out_ndisp: subdirs.append('norm_disp')
        if self.out_sz: subdirs.append('sz')
        if self.out_s3d: subdirs.append('s3d')
        if self.out_mom: subdirs.append('momentum')
        for d in subdirs: os.makedirs(os.path.join(seq_out_dir, d), exist_ok=True)

        Iref = self._read_and_crop(self.ref_path)
        mm_per_pixel = self._estimate_mm_per_pixel(Iref)
        Iref_gpu = cp.array(Iref, dtype=cp.float32)
        
        frames = len(files) 
        h_list, u_list, v_list = [], [], []
        
        # 1. 逐帧推入显卡计算，立刻拉回内存，极致保护 8GB 显存
        for i in range(frames):
            Idef = self._read_and_crop(os.path.join(self.seq_dir, files[i]))
            Idef_gpu = cp.array(Idef, dtype=cp.float32)
            u_gpu, v_gpu = self._fcd_demodulate_correct_gpu(Iref_gpu, Idef_gpu)
            h_gpu = self._fftinvgrad_gpu(-u_gpu, -v_gpu) / self.H
            
            h_list.append(cp.asnumpy(h_gpu[self.edge:-self.edge, self.edge:-self.edge]))
            u_list.append(cp.asnumpy(u_gpu[self.edge:-self.edge, self.edge:-self.edge]))
            v_list.append(cp.asnumpy(v_gpu[self.edge:-self.edge, self.edge:-self.edge]))
            
        shape = h_list[0].shape
        
        # 2. 将数据打包成 32位精度推入 GPU 进行时间轴 3D 解析运算
        h_stack = cp.array(np.array(h_list), dtype=cp.float32)
        u_stack = cp.array(np.array(u_list), dtype=cp.float32)
        v_stack = cp.array(np.array(v_list), dtype=cp.float32)
        
        h_stack -= cp.mean(h_stack, axis=0)
        u_stack -= cp.mean(u_stack, axis=0)
        v_stack -= cp.mean(v_stack, axis=0)
        
        # 显卡光速 Hilbert 提取
        h_ana = cp.conj(self._hilbert_gpu(h_stack))
        u_ana = cp.conj(self._hilbert_gpu(u_stack))
        v_ana = cp.conj(self._hilbert_gpu(v_stack))
        
        amp_w = cp.asnumpy(cp.abs(h_ana))
        phase_w = cp.asnumpy(cp.mod(cp.angle(h_ana), 2*cp.pi))
        
        # 自旋与动量计算 (仍在显卡上)
        calc_spin = self.out_sz or self.out_s3d
        if calc_spin:
            sx = -cp.imag(cp.conj(v_ana) * h_ana - cp.conj(h_ana) * v_ana)
            sy = cp.imag(cp.conj(h_ana) * u_ana - cp.conj(u_ana) * h_ana)
            sz = -cp.imag(cp.conj(u_ana) * v_ana - cp.conj(v_ana) * u_ana)
            sz_cpu = cp.asnumpy(sz)
            sx_cpu = cp.asnumpy(sx)
            sy_cpu = cp.asnumpy(sy)
            sz_vmin, sz_vmax = np.percentile(sz_cpu, self.p_low), np.percentile(sz_cpu, self.p_high)
            
        if self.out_mom:
            # 动量计算在显存内完成
            du_dy, du_dx = cp.gradient(u_ana, axis=(1, 2))
            dv_dy, dv_dx = cp.gradient(v_ana, axis=(1, 2))
            dw_dy, dw_dx = cp.gradient(h_ana, axis=(1, 2))
            Px_gpu = cp.imag(cp.conj(u_ana)*du_dx + cp.conj(v_ana)*dv_dx + cp.conj(h_ana)*dw_dx)
            Py_gpu = cp.imag(cp.conj(u_ana)*du_dy + cp.conj(v_ana)*dv_dy + cp.conj(h_ana)*dw_dy)
            Px_cpu = cp.asnumpy(Px_gpu)
            Py_cpu = cp.asnumpy(Py_gpu)

        # 运算完毕，把剩下的需要画图的栈拉回 CPU，清空显卡显存
        h_stack_cpu = cp.asnumpy(h_stack)
        u_stack_cpu = cp.asnumpy(u_stack)
        v_stack_cpu = cp.asnumpy(v_stack)
        cp.get_default_memory_pool().free_all_blocks()
            
        h_vmin, h_vmax = np.percentile(h_stack_cpu, self.p_low), np.percentile(h_stack_cpu, self.p_high)
        amp_vmin, amp_vmax = np.percentile(amp_w, self.p_low), np.percentile(amp_w, self.p_high)

        # 辅助渲染函数
        def create_fig(cmap, vmin, vmax, title, label=None, is_2d=False):
            if is_2d:
                fig, (ax, ax_w) = plt.subplots(1, 2, figsize=(7.5, 4.8), gridspec_kw={'width_ratios': [4, 1]}, constrained_layout=True)
                im = ax.imshow(np.zeros((shape[0], shape[1], 3)), aspect='equal')
                self._draw_2d_hsv_wheel(ax_w, title_text=title)
            else:
                fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
                im = ax.imshow(np.zeros(shape), cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
                if label:
                    cbar = fig.colorbar(im, ax=ax, shrink=0.85, aspect=25)
                    self._set_colorbar_ticks(cbar, np.array([vmin, vmax]), mm_per_pixel, vmin, vmax, label=label)
                else:
                    ax.set_title(title, fontsize=12)
            self._set_dynamic_ticks(ax, shape, mm_per_pixel)
            return fig, ax, im

        if self.out_hf: f_h, a_h, im_h = create_fig('jet', h_vmin, h_vmax, "水位形变", "水位形变 (mm)")
        if self.out_amp: f_a, a_a, im_a = create_fig('hot', amp_vmin, amp_vmax, "解析信号振幅", "振幅强度")
        if self.out_ph: f_p, a_p, im_p = create_fig('hsv', 0, 2*np.pi, "解析相位角", "相位 (rad)")
        if self.out_pa: f_pa, a_pa, im_pa = create_fig(None, None, None, "彩色相位振幅复合场\n(色相:相位 明度:振幅)")
        if self.out_disp: f_d, a_d, im_d = create_fig(None, None, None, "三维矢量位移场", is_2d=True)
        if self.out_ndisp: f_dn, a_dn, im_dn = create_fig(None, None, None, "归一化位移场", is_2d=True)
        if self.out_sz: f_sz, a_sz, im_sz = create_fig('viridis', sz_vmin, sz_vmax, "Z向自旋角动量", "自旋密度")
        if self.out_s3d: f_s3, a_s3, im_s3 = create_fig(None, None, None, "三维自旋角动量场", is_2d=True)

        for t in range(frames):
            tag = f"{t:03d}"
            
            if self.out_hf:
                im_h.set_data(h_stack_cpu[t])
                f_h.savefig(os.path.join(seq_out_dir, 'hfield', f'hfield_{tag}.jpg'), dpi=150, bbox_inches='tight', pad_inches=0.02)
            
            if self.out_amp:
                im_a.set_data(amp_w[t])
                f_a.savefig(os.path.join(seq_out_dir, 'amplitude', f'amp_{tag}.jpg'), dpi=150, bbox_inches='tight', pad_inches=0.02)
            
            if self.out_ph:
                im_p.set_data(phase_w[t])
                f_p.savefig(os.path.join(seq_out_dir, 'phase', f'phase_{tag}.jpg'), dpi=150, bbox_inches='tight', pad_inches=0.02)
            
            if self.out_pa:
                pa_hsv = np.zeros((shape[0], shape[1], 3))
                pa_hsv[..., 0] = phase_w[t] / (2*np.pi)
                pa_hsv[..., 1] = 1.0
                pa_hsv[..., 2] = np.clip((amp_w[t] - amp_vmin)/(amp_vmax - amp_vmin + 1e-10), 0, 1)
                pa_rgb = plt.matplotlib.colors.hsv_to_rgb(pa_hsv)
                im_pa.set_data(pa_rgb)
                f_pa.savefig(os.path.join(seq_out_dir, 'phaseamp', f'phaseamp_{tag}.jpg'), dpi=150, bbox_inches='tight', pad_inches=0.02)
            
            if self.out_disp or self.out_ndisp:
                u_r, v_r = u_stack_cpu[t], v_stack_cpu[t]
                ph_norm = (np.arctan2(v_r, -u_r) + np.pi) / (2*np.pi)
                
                if self.out_disp:
                    hn = np.clip((h_stack_cpu[t] - h_vmin)/(h_vmax - h_vmin + 1e-10), 0, 1)
                    hsv_d = np.zeros((shape[0], shape[1], 3))
                    hsv_d[..., 0] = ph_norm; hsv_d[..., 1] = 1 - 4*(hn - 0.5)**2; hsv_d[..., 2] = hn
                    im_d.set_data(plt.matplotlib.colors.hsv_to_rgb(hsv_d))
                    f_d.savefig(os.path.join(seq_out_dir, 'displacement', f'disp_{tag}.jpg'), dpi=150, bbox_inches='tight', pad_inches=0.02)
                
                if self.out_ndisp:
                    dn_val = (h_stack_cpu[t] / np.sqrt(u_r**2 + v_r**2 + h_stack_cpu[t]**2 + 1e-10) + 1) / 2
                    hsv_dn = np.zeros((shape[0], shape[1], 3))
                    hsv_dn[..., 0] = ph_norm; hsv_dn[..., 1] = 1 - 4*(dn_val - 0.5)**2; hsv_dn[..., 2] = dn_val
                    im_dn.set_data(plt.matplotlib.colors.hsv_to_rgb(hsv_dn))
                    f_dn.savefig(os.path.join(seq_out_dir, 'norm_disp', f'norm_disp_{tag}.jpg'), dpi=150, bbox_inches='tight', pad_inches=0.02)
            
            if self.out_sz:
                im_sz.set_data(sz_cpu[t])
                f_sz.savefig(os.path.join(seq_out_dir, 'sz', f'sz_{tag}.jpg'), dpi=150, bbox_inches='tight', pad_inches=0.02)
            
            if self.out_s3d:
                s_ph = (np.arctan2(sy_cpu[t], sx_cpu[t]) + np.pi) / (2*np.pi)
                sn = np.clip((sz_cpu[t] - sz_vmin)/(sz_vmax - sz_vmin + 1e-10), 0, 1)
                hsv_s3 = np.zeros((shape[0], shape[1], 3))
                hsv_s3[..., 0] = s_ph; hsv_s3[..., 1] = 1 - 4*(sn - 0.5)**2; hsv_s3[..., 2] = sn
                im_s3.set_data(plt.matplotlib.colors.hsv_to_rgb(hsv_s3))
                f_s3.savefig(os.path.join(seq_out_dir, 's3d', f's3d_{tag}.jpg'), dpi=150, bbox_inches='tight', pad_inches=0.02)
            
            if self.out_mom:
                fig_m, ax_m = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
                ax_m.imshow(pa_rgb if self.out_pa else np.zeros((shape[0], shape[1], 3)), aspect='equal')
                self._set_dynamic_ticks(ax_m, shape, mm_per_pixel)
                
                slc = slice(None, None, self.q_step)
                X_grid, Y_grid = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
                Px, Py = Px_cpu[t], Py_cpu[t]
                p_mag = np.sqrt(Px**2 + Py**2)
                mask = p_mag[slc, slc] > np.percentile(p_mag, 10)
                
                ax_m.quiver(X_grid[slc, slc][mask], Y_grid[slc, slc][mask], 
                            Px[slc, slc][mask], Py[slc, slc][mask], 
                            color='red', scale=1.0/self.q_scale, scale_units='xy', angles='xy')
                ax_m.set_title("动量流场密度 (Momentum Density)")
                fig_m.savefig(os.path.join(seq_out_dir, 'momentum', f'momentum_{tag}.jpg'), dpi=150, bbox_inches='tight', pad_inches=0.02)
                plt.close(fig_m)

        plt.close('all')
        out_path = os.path.join(seq_out_dir, "Seq_Amp_Max.png")
        plt.imsave(out_path, np.max(amp_w, axis=0), cmap='hot')

        log_c = f"===== 序列分析完成 (GPU 加速版) =====\n输出目录: {seq_out_dir}\n共处理: {frames}帧\n"
        return os.path.join(seq_out_dir, 'hfield'), self.write_log("ImageSeq_GPU", log_c)