import os
import cv2
import numpy as np
from scipy.fft import fft2, ifft2, fftfreq, fftshift, ifftshift
from scipy.signal import hilbert
from datetime import datetime
import math
from scipy.signal import hilbert, butter, filtfilt # 🌟 新增滤波
from scipy.optimize import curve_fit               # 🌟 新增曲线拟合
from scipy.signal.windows import tukey

# 强行切换到 Agg 后端，确保多线程下终端不卡死
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.path import Path 

import json
from scipy.signal import hilbert
import matplotlib.patches as patches

import time
import matplotlib.lines as mlines

# 100% 确保支持中文字符与负号渲染
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False 

class InteractiveMeasurer:
    def __init__(self, ax, fig, cm_per_pixel):
        self.ax = ax
        self.fig = fig
        self.cm_per_pixel = cm_per_pixel

        # 🌟 终极修复核心 1：在初始状态下，死死记住当前图像最纯净的画幅边界坐标限制
        self.orig_xlim = self.ax.get_xlim()
        self.orig_ylim = self.ax.get_ylim()

        # 全局控制
        self.mode = 'p2p' # 'p2p' (点对点) 或 'pl' (平行线)
        self.state = 'idle'
        self.count = 0
        self.measurements = []
        self.bg = None
        self.ext_len = 8000 # 平行线无限延长长度
        
        # 预定义高对比度颜色表，用于固化不同的测量组
        self.colors = ['#FF00FF', '#00FFFF', '#FFFF00', '#00FF00', '#FF9900', '#FF0000']

        # ================= 动态绘图对象初始化 =================
        # 1. P2P 动态对象
        self.dyn_p2p_line, = self.ax.plot([], [], color='white', lw=2, ls='--', visible=False)
        self.dyn_p2p_p1, = self.ax.plot([], [], 'wo', markersize=6, visible=False)
        self.dyn_p2p_p2, = self.ax.plot([], [], 'wo', markersize=6, visible=False)
        self.dyn_p2p_txt = self.ax.text(0, 0, "", color='black', fontsize=10, 
                                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'), visible=False)

        # 2. PL 平行线动态对象
        self.dyn_pl_l1, = self.ax.plot([], [], color='white', lw=2, ls='--', visible=False)
        self.dyn_pl_l2, = self.ax.plot([], [], color='white', lw=2, ls='--', visible=False)
        self.dyn_pl_perp, = self.ax.plot([], [], color='white', lw=2, ls='-', visible=False) # 垂线
        self.dyn_pl_p1, = self.ax.plot([], [], 'ws', markersize=6, visible=False) # 垂线起点
        self.dyn_pl_p2, = self.ax.plot([], [], 'ws', markersize=6, visible=False) # 垂线终点
        self.dyn_pl_txt = self.ax.text(0, 0, "", color='black', fontsize=10,
                                       bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'), visible=False)

        # 挂载事件
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_move = self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # 强制绘制一次以获取纯净背景
        self.fig.canvas.draw()
        self.capture_bg()
        self.update_title()

    def update_title(self):
        base_title = f"【按 M 键切换模式】当前: {'点对点测距' if self.mode == 'p2p' else '平行线测距'} | 已测 {self.count} 组数据\n"
        
        if self.mode == 'p2p':
            if self.state == 'idle':
                t2 = "【左键单击】确定起点"
            elif self.state == 'p1_selected':
                t2 = "【移动】拉出连线，【左键单击】锁定终点并保存"
        else: # PL 模式
            if self.state == 'idle':
                t2 = "【左键单击】确定基准线起点"
            elif self.state == 'ref_start':
                t2 = "【移动】拉出基准线对齐波前，【左键单击】锁定"
            elif self.state == 'parallel':
                t2 = "【移动】生成平行游标，【左键单击】放置"
            elif self.state in ['adjust', 'pan_l1', 'pan_l2']:
                t2 = "【左键拖拽】任一直线可平移微调 | 【C 键】确认保存 | 【右键】撤销重画"

        self.ax.set_title(base_title + t2, fontsize=11, weight='bold', color='yellow' if self.mode=='p2p' else 'cyan',
                          bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=3))
        self.fig.canvas.draw_idle()

    def capture_bg(self):
        """隐蔽所有动态元素，拍摄当前包含已固化测量的背景"""
        artists = [self.dyn_p2p_line, self.dyn_p2p_p1, self.dyn_p2p_p2, self.dyn_p2p_txt,
                   self.dyn_pl_l1, self.dyn_pl_l2, self.dyn_pl_perp, self.dyn_pl_p1, self.dyn_pl_p2, self.dyn_pl_txt]
        vis_states = [a.get_visible() for a in artists]
        for a in artists: a.set_visible(False)
        self.fig.canvas.draw()
        self.bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        for a, v in zip(artists, vis_states): a.set_visible(v)

    def reset_dynamic_state(self):
        """清空未完成的测量状态，隐藏动态游标"""
        self.state = 'idle'
        artists = [self.dyn_p2p_line, self.dyn_p2p_p1, self.dyn_p2p_p2, self.dyn_p2p_txt,
                   self.dyn_pl_l1, self.dyn_pl_l2, self.dyn_pl_perp, self.dyn_pl_p1, self.dyn_pl_p2, self.dyn_pl_txt]
        for a in artists: a.set_visible(False)
        self.capture_bg()
        self.update_title()

    def commit_measurement(self):
        """将当前满意的动态游标固化为静态图层，存入日志"""
        color = self.colors[self.count % len(self.colors)]
        
        if self.mode == 'p2p':
            p1, p2 = self.p2p_p1, self.p2p_p2
            dist_cm = np.hypot(p2[0]-p1[0], p2[1]-p1[1]) * self.cm_per_pixel
            # 绘制静态对象
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=2)
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'o', color=color, markersize=5)
            cx, cy = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
            self.ax.text(cx, cy, f" #{self.count+1}: {dist_cm:.2f}cm ", color='white', 
                         fontsize=10, ha='center', va='center', bbox=dict(facecolor=color, alpha=0.8, edgecolor='none'))
            self.measurements.append({'id': self.count+1, 'type': 'P2P', 'p1': p1, 'p2': p2, 'dist_cm': dist_cm, 'time': time.strftime("%H:%M:%S")})

        elif self.mode == 'pl':
            p_start, p_end, dist_cm = self._calc_pl_geometry()
            dx, dy = self.pl_dir
            norm = np.hypot(dx, dy)
            ux, uy = dx/norm, dy/norm
            # 延长线
            l1_x = [p_start[0] - ux*self.ext_len, p_start[0] + ux*self.ext_len]
            l1_y = [p_start[1] - uy*self.ext_len, p_start[1] + uy*self.ext_len]
            l2_x = [p_end[0] - ux*self.ext_len, p_end[0] + ux*self.ext_len]
            l2_y = [p_end[1] - uy*self.ext_len, p_end[1] + uy*self.ext_len]
            
            # 绘制静态对象
            self.ax.plot(l1_x, l1_y, color=color, lw=1.5, ls='--')
            self.ax.plot(l2_x, l2_y, color=color, lw=1.5, ls='--')
            self.ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], color=color, lw=2.5) # 加粗垂线
            self.ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], 's', color=color, markersize=5)
            
            cx, cy = (p_start[0]+p_end[0])/2, (p_start[1]+p_end[1])/2
            self.ax.text(cx, cy, f" #{self.count+1}: {dist_cm:.2f}cm ", color='white', 
                         fontsize=10, ha='center', va='center', bbox=dict(facecolor=color, alpha=0.8, edgecolor='none'))
            self.measurements.append({'id': self.count+1, 'type': 'Parallel', 'p1': p_start, 'p2': p_end, 'dist_cm': dist_cm, 'time': time.strftime("%H:%M:%S")})

        self.count += 1
        
        # 🌟 终极修复核心 2：在固化创建完图层后，强行把画幅拉回初始状态，遏制 Matplotlib 的乱缩放 Bug
        self.ax.set_xlim(self.orig_xlim)
        self.ax.set_ylim(self.orig_ylim)
        
        self.reset_dynamic_state() # 重置并把刚刚画的静态内容收入 Background 中

    # ================= 核心几何计算 =================
    def _calc_pl_geometry(self):
        dx, dy = self.pl_dir
        A, B = -dy, dx
        norm2 = A**2 + B**2
        if norm2 == 0: return self.pl_base, self.pl_base, 0
        
        C2 = dy * self.pl_l2_pt[0] - dx * self.pl_l2_pt[1]
        x0, y0 = self.pl_base
        x_int = x0 - A * (A * x0 + B * y0 + C2) / norm2
        y_int = y0 - B * (A * x0 + B * y0 + C2) / norm2
        
        dist_px = np.hypot(x_int - x0, y_int - y0)
        return (x0, y0), (x_int, y_int), dist_px * self.cm_per_pixel

    def _get_extended(self, pt, dx, dy):
        norm = np.hypot(dx, dy)
        if norm == 0: return [pt[0], pt[0]], [pt[1], pt[1]]
        ux, uy = dx/norm, dy/norm
        return [pt[0] - ux*self.ext_len, pt[0] + ux*self.ext_len], [pt[1] - uy*self.ext_len, pt[1] + uy*self.ext_len]

    # ================= 交互事件响应 =================
    def on_key(self, event):
        if event.key in ['m', 'M']:
            self.mode = 'pl' if self.mode == 'p2p' else 'p2p'
            self.reset_dynamic_state()
        elif event.key in ['c', 'C'] and self.mode == 'pl' and self.state in ['adjust', 'pan_l1', 'pan_l2']:
            self.commit_measurement()
        elif event.key == 'enter':
            import matplotlib.pyplot as plt
            plt.close(self.fig)

    def on_press(self, event):
        if event.inaxes != self.ax: return

        if event.button == 3: # 右键撤销当前动态操作
            self.reset_dynamic_state()
            return

        if event.button != 1: return # 仅响应左键

        if self.mode == 'p2p':
            if self.state == 'idle':
                self.p2p_p1 = (event.xdata, event.ydata)
                self.dyn_p2p_p1.set_data([event.xdata], [event.ydata])
                self.dyn_p2p_p1.set_visible(True)
                self.dyn_p2p_line.set_visible(True)
                self.dyn_p2p_p2.set_visible(True)
                self.dyn_p2p_txt.set_visible(True)
                self.state = 'p1_selected'
                self.update_title()
            elif self.state == 'p1_selected':
                self.p2p_p2 = (event.xdata, event.ydata)
                self.commit_measurement() # P2P 直接固化

        elif self.mode == 'pl':
            if self.state == 'idle':
                self.pl_p1 = (event.xdata, event.ydata)
                self.dyn_pl_l1.set_visible(True)
                self.state = 'ref_start'
                self.update_title()
            elif self.state == 'ref_start':
                dx = event.xdata - self.pl_p1[0]
                dy = event.ydata - self.pl_p1[1]
                if np.hypot(dx, dy) < 2: return 
                self.pl_dir = (dx, dy)
                self.pl_base = ((self.pl_p1[0]+event.xdata)/2, (self.pl_p1[1]+event.ydata)/2) 
                self.dyn_pl_l2.set_visible(True)
                self.dyn_pl_perp.set_visible(True)
                self.dyn_pl_p1.set_visible(True)
                self.dyn_pl_p2.set_visible(True)
                self.dyn_pl_txt.set_visible(True)
                self.state = 'parallel'
                self.update_title()
            elif self.state == 'parallel':
                self.pl_l2_pt = (event.xdata, event.ydata)
                self.state = 'adjust'
                self.update_title()
            elif self.state == 'adjust':
                dx, dy = self.pl_dir
                A, B = -dy, dx
                norm = np.hypot(A, B)
                if norm == 0: return
                C1 = dy * self.pl_base[0] - dx * self.pl_base[1]
                C2 = dy * self.pl_l2_pt[0] - dx * self.pl_l2_pt[1]
                
                ex, ey = event.xdata, event.ydata
                d1 = abs(A*ex + B*ey + C1) / norm
                d2 = abs(A*ex + B*ey + C2) / norm

                if d1 < d2 and d1 < 40: 
                    self.state = 'pan_l1'
                    self.pan_offset = (self.pl_base[0] - ex, self.pl_base[1] - ey)
                    self.update_title()
                elif d2 <= d1 and d2 < 40: 
                    self.state = 'pan_l2'
                    self.pan_offset = (self.pl_l2_pt[0] - ex, self.pl_l2_pt[1] - ey)
                    self.update_title()

    def on_move(self, event):
        if event.inaxes != self.ax or self.bg is None: return

        if self.mode == 'p2p' and self.state == 'p1_selected':
            p1 = self.p2p_p1
            p2 = (event.xdata, event.ydata)
            dist_cm = np.hypot(p2[0]-p1[0], p2[1]-p1[1]) * self.cm_per_pixel
            
            self.dyn_p2p_line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
            self.dyn_p2p_p2.set_data([p2[0]], [p2[1]])
            self.dyn_p2p_txt.set_position(((p1[0]+p2[0])/2, (p1[1]+p2[1])/2))
            self.dyn_p2p_txt.set_text(f" {dist_cm:.2f} cm ")

        elif self.mode == 'pl':
            if self.state == 'ref_start':
                dx = event.xdata - self.pl_p1[0]
                dy = event.ydata - self.pl_p1[1]
                x1, y1 = self._get_extended(self.pl_p1, dx, dy)
                self.dyn_pl_l1.set_data(x1, y1)
            elif self.state in ['parallel', 'pan_l1', 'pan_l2']:
                if self.state == 'parallel':
                    self.pl_l2_pt = (event.xdata, event.ydata)
                elif self.state == 'pan_l1':
                    self.pl_base = (event.xdata + self.pan_offset[0], event.ydata + self.pan_offset[1])
                elif self.state == 'pan_l2':
                    self.pl_l2_pt = (event.xdata + self.pan_offset[0], event.ydata + self.pan_offset[1])

                p_start, p_end, dist_cm = self._calc_pl_geometry()
                dx, dy = self.pl_dir
                x1, y1 = self._get_extended(p_start, dx, dy)
                x2, y2 = self._get_extended(p_end, dx, dy)
                
                self.dyn_pl_l1.set_data(x1, y1)
                self.dyn_pl_l2.set_data(x2, y2)
                self.dyn_pl_perp.set_data([p_start[0], p_end[0]], [p_start[1], p_end[1]])
                self.dyn_pl_p1.set_data([p_start[0]], [p_start[1]])
                self.dyn_pl_p2.set_data([p_end[0]], [p_end[1]])
                
                self.dyn_pl_txt.set_position(((p_start[0]+p_end[0])/2, (p_start[1]+p_end[1])/2))
                self.dyn_pl_txt.set_text(f" D = {dist_cm:.3f} cm ")
        else:
            return

        # 高速 Blitting 局部无缝刷新引擎
        self.fig.canvas.restore_region(self.bg)
        artists = [self.dyn_p2p_line, self.dyn_p2p_p1, self.dyn_p2p_p2, self.dyn_p2p_txt,
                   self.dyn_pl_l1, self.dyn_pl_l2, self.dyn_pl_perp, self.dyn_pl_p1, self.dyn_pl_p2, self.dyn_pl_txt]
        for a in artists:
            if a.get_visible(): self.ax.draw_artist(a)
        self.fig.canvas.blit(self.ax.bbox)

    def on_release(self, event):
        if self.mode == 'pl' and self.state in ['pan_l1', 'pan_l2']:
            self.state = 'adjust'
            self.update_title()

# =====================================================================
# 🌟 定标专用交互类 1：全局阵列平行基准线提取器 (极速修复版)
# =====================================================================
class MasterLineSelector:
    def __init__(self, ax, fig):
        self.ax = ax
        self.fig = fig
        self.start_pt = None
        self.end_pt = None
        self.state = 'draw'
        self.bg = None # Blitting 背景缓存
        
        # 平移专用的坐标缓存
        self.press_x = None
        self.press_y = None
        self.orig_xdata = None
        self.orig_ydata = None
        
        self.line, = self.ax.plot([], [], color='cyan', lw=2, ls='--')
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.ax.set_title("【定标 1/2: 全局平行基准线】\n请拖拽画一条平行于喇叭阵列的基准长线", color='red', weight='bold')

    def capture_bg(self):
        vis = self.line.get_visible()
        self.line.set_visible(False)
        self.fig.canvas.draw()
        self.bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self.line.set_visible(vis)

    def on_press(self, event):
        if event.inaxes != self.ax: return
        if event.button == 3: # 右键重画
            self.start_pt = None
            self.end_pt = None
            self.state = 'draw'
            self.line.set_data([], [])
            self.bg = None
            self.fig.canvas.draw_idle()
            return
        if event.button == 1:
            if self.state == 'draw':
                if self.start_pt is None:
                    self.start_pt = (event.xdata, event.ydata)
                    self.line.set_data([event.xdata], [event.ydata])
                    self.capture_bg()
                else:
                    self.end_pt = (event.xdata, event.ydata)
                    self.line.set_data([self.start_pt[0], self.end_pt[0]], [self.start_pt[1], self.end_pt[1]])
                    self.state = 'done'
                    self.bg = None
                    self.ax.set_title("基准线已锁定！拖拽可平行移动，按 [Enter] 键进入逐个定位", color='green', weight='bold')
                    self.fig.canvas.draw_idle()
            elif self.state == 'done':
                # 进入平移模式，死死记住初始位置，防止飞出画面
                self.state = 'pan'
                self.press_x = event.xdata
                self.press_y = event.ydata
                self.orig_xdata = list(self.line.get_xdata())
                self.orig_ydata = list(self.line.get_ydata())
                self.capture_bg()

    def on_move(self, event):
        if event.inaxes != self.ax: return
        if self.bg is None: self.capture_bg()

        if self.state == 'draw' and self.start_pt is not None:
            self.line.set_data([self.start_pt[0], event.xdata], [self.start_pt[1], event.ydata])
        elif self.state == 'pan' and self.press_x is not None:
            dx = event.xdata - self.press_x
            dy = event.ydata - self.press_y
            self.line.set_data([self.orig_xdata[0]+dx, self.orig_xdata[1]+dx], 
                               [self.orig_ydata[0]+dy, self.orig_ydata[1]+dy])
        else:
            return
        
        # 极速刷新机制
        self.fig.canvas.restore_region(self.bg)
        self.ax.draw_artist(self.line)
        self.fig.canvas.blit(self.ax.bbox)

    def on_release(self, event):
        if self.state == 'pan':
            self.state = 'done'
            self.start_pt = (self.line.get_xdata()[0], self.line.get_ydata()[0])
            self.end_pt = (self.line.get_xdata()[1], self.line.get_ydata()[1])
            self.bg = None

    def on_key(self, event):
        if event.key == 'enter' and self.state == 'done':
            import matplotlib.pyplot as plt
            plt.close(self.fig)

# =====================================================================
# 🌟 定标专用交互类 2：正交探点选择器 (极速修复版)
# =====================================================================
class OrthogonalPicker:
    def __init__(self, ax, fig, master_p1, master_p2, ch_name):
        self.ax = ax
        self.fig = fig
        self.p1 = np.array(master_p1)
        self.p2 = np.array(master_p2)
        self.target_pt = None
        self.state = 'picking'
        self.bg = None
        
        self.ax.plot([self.p1[0], self.p2[0]], [self.p1[1], self.p2[1]], color='cyan', lw=2, ls='--')
        self.perp_line, = self.ax.plot([], [], color='magenta', lw=2)
        self.marker, = self.ax.plot([], [], 'mo', markersize=8)
        
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        self.ax.set_title(f"【定标 2/2: 提取 {ch_name}】\n移动鼠标拉出垂线，点击左键锁定，按 Enter 确认", color='yellow', backgroundcolor='black', weight='bold')

    def capture_bg(self):
        v1 = self.perp_line.get_visible()
        v2 = self.marker.get_visible()
        self.perp_line.set_visible(False)
        self.marker.set_visible(False)
        self.fig.canvas.draw()
        self.bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self.perp_line.set_visible(v1)
        self.marker.set_visible(v2)

    def _get_proj(self, x, y):
        v = self.p2 - self.p1
        w = np.array([x, y]) - self.p1
        c1 = np.dot(w, v)
        c2 = np.dot(v, v)
        if c2 == 0: return self.p1
        return self.p1 + v * (c1 / c2)

    def on_move(self, event):
        if event.inaxes != self.ax or self.state == 'done': return
        if self.bg is None: self.capture_bg()

        proj = self._get_proj(event.xdata, event.ydata)
        dx, dy = self.p2[0]-self.p1[0], self.p2[1]-self.p1[1]
        norm = np.hypot(dx, dy)
        ux, uy = -dy/norm, dx/norm
        self.perp_line.set_data([proj[0]-ux*2000, proj[0]+ux*2000], [proj[1]-uy*2000, proj[1]+uy*2000])
        self.marker.set_data([proj[0]], [proj[1]])
        
        self.fig.canvas.restore_region(self.bg)
        self.ax.draw_artist(self.perp_line)
        self.ax.draw_artist(self.marker)
        self.fig.canvas.blit(self.ax.bbox)

    def on_press(self, event):
        if event.inaxes != self.ax: return
        if event.button == 1:
            self.target_pt = self._get_proj(event.xdata, event.ydata)
            self.state = 'done'
            self.ax.set_title(f"{self.target_pt.astype(int)} 已锁定！按 Enter 进入下一个", color='green', weight='bold')
            self.fig.canvas.draw_idle()
        elif event.button == 3:
            self.state = 'picking'
            self.target_pt = None

    def on_key(self, event):
        if event.key == 'enter' and self.state == 'done':
            import matplotlib.pyplot as plt
            plt.close(self.fig)

# =====================================================================
# 🌟 定标专用：正弦曲线拟合函数
# =====================================================================
def sine_fit_func(t, A, omega, phi, C):
    return A * np.sin(omega * t + phi) + C

class FCDCore:
    def __init__(self, ref_path, def_path=None, seq_dir=None, out_dir=None, crop_pixels=(0,0,0,0), 
                 water_depth=30.0, fps=30.0,low_pass_suppress=65.0, krad_factor=0.9, edge_width=10, 
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
        self.fps = fps
        
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

    def write_log(self, prefix, content, target_dir=None):
        # 🌟 修改：支持动态传入目标保存文件夹，未传入则默认保存到基础 log 目录
        if target_dir is None:
            target_dir = self.log_dir
        os.makedirs(target_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = os.path.join(target_dir, f"Log_{prefix}_{timestamp}.txt")
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

    def _subpixel_peak(self, mag, y, x):
        """
        新增辅助函数：基于局部对数抛物线拟合的亚像素寻峰
        1:1 复刻 MATLAB findpeaks2.m 中的 subpxpeak 逻辑
        """
        rows, cols = mag.shape
        if y <= 0 or y >= rows - 1 or x <= 0 or x >= cols - 1:
            return float(y), float(x)
        
        eps = 1e-10
        val = np.log(mag[y, x] + eps)
        
        # X 方向亚像素偏移
        val_l = np.log(mag[y, x - 1] + eps)
        val_r = np.log(mag[y, x + 1] + eps)
        denom_x = val_l + val_r - 2 * val
        dx = -0.5 * (val_r - val_l) / denom_x if denom_x != 0 else 0.0
        
        # Y 方向亚像素偏移
        val_u = np.log(mag[y - 1, x] + eps)
        val_d = np.log(mag[y + 1, x] + eps)
        denom_y = val_u + val_d - 2 * val
        dy = -0.5 * (val_d - val_u) / denom_y if denom_y != 0 else 0.0
        
        return y + dy, x + dx

    def _find_orth_carrier_pks(self, Iref):
        rows, cols = Iref.shape
        F = fftshift(fft2(Iref, workers=-1))
        F_mag = np.abs(F)

        cy, cx = rows // 2, cols // 2
        Y, X = np.ogrid[:rows, :cols]

        # 保护极低频
        low_pass_suppress = 1.0 - np.exp(-((X - cx)**2 + (Y - cy)**2) / (2.0 * (self.low_pass_suppress_r**2)))
        F_mag_filtered = F_mag * low_pass_suppress

        # 🌟 修复：找第一载波 kr，并强制进行亚像素插值！
        y1, x1 = np.unravel_index(np.argmax(F_mag_filtered), F_mag_filtered.shape)
        y1_sub, x1_sub = self._subpixel_peak(F_mag_filtered, y1, x1)
        kr = np.array([y1_sub - cy, x1_sub - cx])

        # 正交叉乘抑制逻辑
        K_y = Y - cy
        K_x = X - cx
        K_mag = np.sqrt(K_x**2 + K_y**2)
        K_mag[K_mag == 0] = 1.0  
        kr_mag = np.sqrt(kr[0]**2 + kr[1]**2)
        
        sin_theta = np.abs(K_x * kr[0] - K_y * kr[1]) / (K_mag * kr_mag)
        F_mag_ortho = F_mag_filtered * sin_theta

        # 🌟 修复：在强制正交化后的频谱中抓取第二载波 ku，并强制进行亚像素插值！
        y2, x2 = np.unravel_index(np.argmax(F_mag_ortho), F_mag_ortho.shape)
        y2_sub, x2_sub = self._subpixel_peak(F_mag_ortho, y2, x2)
        ku = np.array([y2_sub - cy, x2_sub - cx])

        krad = (np.sqrt(np.sum((kr - ku)**2)) / 2.0) * self.krad_factor
        return kr, ku, krad

    def _prepare_cache(self, Iref):
        if getattr(self, '_cache_valid', False) and getattr(self, '_cached_shape', None) == Iref.shape:
            return

        rows, cols = Iref.shape
        self._cached_shape = Iref.shape

        self.kr, self.ku, self.krad = self._find_orth_carrier_pks(Iref)
        self.Fref = fft2(Iref - np.mean(Iref), workers=-1)

        X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
        cy, cx = rows // 2, cols // 2

        # ====== 替换 _prepare_cache 中的掩膜部分 ======
        # 🌟 修复 1：彻底废除高斯掩膜，恢复 MATLAB 原版的硬圆盘掩膜 (Hard Disk Mask)
        # 这将把边缘伪影的空域弥散限制在 15 像素以内，不再需要 120px 的巨大裁切！
        dist2_r = (X - (cx + self.kr[1]))**2 + (Y - (cy + self.kr[0]))**2
        self.mask_r = ifftshift((dist2_r < self.krad**2).astype(float))

        dist2_u = (X - (cx + self.ku[1]))**2 + (Y - (cy + self.ku[0]))**2
        self.mask_u = ifftshift((dist2_u < self.krad**2).astype(float))
        # ===================================================

        self.cr_ref_conj = np.conj(ifft2(self.Fref * self.mask_r, workers=-1))
        self.cu_ref_conj = np.conj(ifft2(self.Fref * self.mask_u, workers=-1))

        self._cache_valid = True

    def _fcd_demodulate_correct(self, Iref, Idef):
        self._prepare_cache(Iref)

        Fdef = fft2(Idef - np.mean(Idef), workers=-1)
        cr_def = ifft2(Fdef * self.mask_r, workers=-1)
        cu_def = ifft2(Fdef * self.mask_u, workers=-1)

        psi_r = cr_def * self.cr_ref_conj
        psi_u = cu_def * self.cu_ref_conj

        # 直接获取无损相位，绝对不需要 Unwrap！
        dphi_r = -np.angle(psi_r)
        dphi_u = -np.angle(psi_u)

        rows, cols = Iref.shape
        K_rx = 2.0 * np.pi * self.kr[1] / cols
        K_ry = 2.0 * np.pi * self.kr[0] / rows
        K_ux = 2.0 * np.pi * self.ku[1] / cols
        K_uy = 2.0 * np.pi * self.ku[0] / rows

        det = K_rx * K_uy - K_ry * K_ux
        if abs(det) < 1e-8: det = 1e-8

        # 最标准的克莱姆法则求解矩阵，这本身就是完美的，千万不要乱加负号
        d_x = (K_uy * dphi_r - K_ry * dphi_u) / det
        d_y = (-K_ux * dphi_r + K_rx * dphi_u) / det

        return d_x, d_y

    def _get_bg_cache(self, shape):
        """生成并缓存伪逆矩阵，使得曲面拟合降维为纯乘法运算"""
        if hasattr(self, '_bg_cache') and self._bg_cache['shape'] == shape:
            return self._bg_cache

        rows, cols = shape
        X, Y = np.meshgrid(np.linspace(-1, 1, cols), np.linspace(-1, 1, rows))
        X_f, Y_f = X.flatten(), Y.flatten()

        A = np.column_stack((
            np.ones_like(X_f), X_f, Y_f,
            X_f**2, Y_f**2, X_f*Y_f,
            X_f**3, Y_f**3, (X_f**2)*Y_f, X_f*(Y_f**2)
        ))

        # 🌟 预计算 Moore-Penrose 伪逆 (耗时约0.5秒，但只执行一次)
        A_pinv = np.linalg.pinv(A)

        self._bg_cache = {
            'shape': shape,
            'A': A,
            'A_pinv': A_pinv
        }
        return self._bg_cache

    def _remove_background_surface(self, h, order=3):
        """极速曲面清洗器 (耗时从数百毫秒降低至几毫秒)"""
        c = self._get_bg_cache(h.shape)
        
        # O(N) 一步矩阵乘法直接获取 10 个曲面系数，避免每一帧都跑 lstsq 求解器
        C = c['A_pinv'] @ h.flatten()
        bg = (c['A'] @ C).reshape(h.shape)
        
        return h - bg
    
    def _get_sylv_cache(self, m, n):
        """生成并缓存积分器所需的所有降维、特征值与逆矩阵 (仅在第一帧耗时)"""
        if hasattr(self, '_sylv_cache') and self._sylv_cache['shape'] == (m, n):
            return self._sylv_cache

        def designgrad1D(N):
            D = np.zeros((N, N))
            idx = np.arange(1, N-1)
            D[idx, idx-1] = -0.5
            D[idx, idx+1] = 0.5
            D[0, :3] = [-1.5, 2.0, -0.5]
            D[-1, -3:] = [0.5, -2.0, 1.5]
            return D

        def housh(v):
            v = v.reshape(-1, 1)
            return np.eye(len(v)) - 2.0 * (v @ v.T) / (v.T @ v)

        Dx, Dy = designgrad1D(n), designgrad1D(m)
        vn, vm = np.ones((n, 1)), np.ones((m, 1))
        vn[0, 0] = 1.0 + np.sqrt(n)
        vm[0, 0] = 1.0 + np.sqrt(m)

        Px, Py = housh(vn), housh(vm)
        Dhx, Dhy = Dx @ Px, Dy @ Py

        A_sub = (Dhy.T @ Dhy)[1:, 1:]
        B_sub = (Dhx.T @ Dhx)[1:, 1:]

        # 🌟 极速核心 1：利用对称性一次性完成解析特征值分解，彻底干掉 solve_sylvester
        evals_A, evecs_A = np.linalg.eigh(A_sub)
        evals_B, evecs_B = np.linalg.eigh(B_sub)
        eigen_denom = evals_A[:, None] + evals_B[None, :]

        # 极速核心 2：预计算边缘权重所需矩阵的逆
        A_sub_inv = evecs_A @ np.diag(1.0 / evals_A) @ evecs_A.T
        B_sub_inv = evecs_B @ np.diag(1.0 / evals_B) @ evecs_B.T

        self._sylv_cache = {
            'shape': (m, n),
            'Px': Px, 'Py': Py, 'Dhx': Dhx, 'Dhy': Dhy,
            'evecs_A': evecs_A, 'evecs_B': evecs_B,
            'eigen_denom': eigen_denom,
            'A_sub_inv': A_sub_inv, 'B_sub_inv': B_sub_inv
        }
        return self._sylv_cache

    def _fftinvgrad(self, hx, hy):
        """
        全解析 O(N^2) 代数积分器。输出结果与原版完美一致，但速度提升几十倍！
        (为了兼容其他函数调用，名字保留为 _fftinvgrad)
        """
        m, n = hx.shape
        c = self._get_sylv_cache(m, n)

        # 提取边界参数
        C = c['Dhy'].T @ hy @ c['Px'] + c['Py'].T @ hx @ c['Dhx']
        c01, c10, C_sub = C[0, 1:], C[1:, 0], C[1:, 1:]

        # O(N^2) 极速求权重
        w01 = c['B_sub_inv'] @ c01
        w10 = c['A_sub_inv'] @ c10

        # 🌟 超级加速：在特征空间内直接用标量除法解 Sylvester 方程
        C_tilde = c['evecs_A'].T @ C_sub @ c['evecs_B']
        W_tilde = C_tilde / c['eigen_denom']
        W11 = c['evecs_A'] @ W_tilde @ c['evecs_B'].T

        # 重构并逆映射
        W = np.zeros((m, n))
        W[0, 1:], W[1:, 0], W[1:, 1:] = w01, w10, W11
        
        return c['Py'] @ W @ c['Px'].T

    def process_single_frame(self):
        Iref = self._read_and_crop(self.ref_path)
        Idef = self._read_and_crop(self.def_path)
        
        mpp = self._estimate_mm_per_pixel(Iref)
        u_px, v_px = self._fcd_demodulate_correct(Iref, Idef)
        
        # 🌟 终极修复：先切除 FCD 频域泄露污染的边缘 (截断伪影)
        e = self.edge
        u_crop = u_px[e:-e, e:-e]
        v_crop = v_px[e:-e, e:-e]
        
        h_px_int = self._fftinvgrad(-u_crop, -v_crop)

        # 🌟 应用严格的二次方物理量纲缩放
        h = h_px_int * (mpp**2) / self.H
        h = self._remove_background_surface(h, order=3)
        u = u_crop * mpp
        v = v_crop * mpp
        
        return h, u, v, Idef

    def _estimate_mm_per_pixel(self, Iref, G=1.5):
        kr, ku, _ = self._find_orth_carrier_pks(Iref)
        rows, cols = Iref.shape
        kr_phys = np.array([kr[1] / cols, kr[0] / rows]) * 2.0 * np.pi
        ku_phys = np.array([ku[1] / cols, ku[0] / rows]) * 2.0 * np.pi
        k0 = kr_phys + ku_phys
        kmag = np.linalg.norm(k0)
        return kmag * G / (2.0 * np.pi) if kmag > 0 else 0.12

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
        # 1. 换算为物理真实极值
        phys_min, phys_max = vmin * mm_per_pixel, vmax * mm_per_pixel
        phys_range = max(1e-6, phys_max - phys_min)
        
        # 🌟 扩充小尺度的 nice_steps，完美应对微米级波动
        nice_steps = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
        step_size = next((s for s in nice_steps if s >= phys_range / 5.0), nice_steps[-1])
        
        start_val = math.ceil(phys_min / step_size) * step_size
        end_val = math.floor(phys_max / step_size) * step_size
        
        # 2. 生成中间的规律刻度
        inner_ticks = np.arange(start_val, end_val + step_size * 0.1, step_size)
        
        # 3. 🌟 核心逻辑：强制首尾挂载极值，并剔除离极值太近的中间刻度 (防止数字重叠)
        threshold = step_size * 0.20 # 设定安全距离阈值为步长的 20%
        valid_inner = [t for t in inner_ticks if (t - phys_min) > threshold and (phys_max - t) > threshold]
        
        final_ticks = [phys_min] + valid_inner + [phys_max]
        
        # 设置 Colorbar 的实际刻度位置 (需要转回像素空间比率)
        cbar.set_ticks(np.array(final_ticks) / mm_per_pixel)
        
        # 4. 🌟 根据动态范围，智能分配极高精度的小数位数
        if step_size >= 1:
            fmt = "{:.0f}"
        elif step_size >= 0.1:
            fmt = "{:.1f}"
        elif step_size >= 0.01:
            fmt = "{:.2f}"
        elif step_size >= 0.001:
            fmt = "{:.3f}"
        else:
            fmt = "{:.4f}"
            
        cbar.set_ticklabels([fmt.format(x) for x in final_ticks])
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
        import matplotlib
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
        import os, time

        if not self.ref_path or not os.path.exists(self.ref_path):
            raise ValueError("未选择有效的参考图像！")
        if not self.def_path or not os.path.exists(self.def_path):
            raise ValueError("未选择有效的形变图像！")

        ref_name = os.path.basename(self.ref_path)
        def_name = os.path.basename(self.def_path)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        run_dir = os.path.join(self.out_dir, "SingleFrame_Results", f"Run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)

        Iref = self._read_and_crop(self.ref_path)
        mm_per_pixel = self._estimate_mm_per_pixel(Iref)

        h_t, u_t, v_t, _ = self.process_single_frame()

        shape = h_t.shape

        def create_fig(cmap, vmin, vmax, title, label=None, is_2d=False, legend_kwargs=None):
            if is_2d:
                fig, (ax, ax_w) = plt.subplots(1, 2, figsize=(8.5, 5), gridspec_kw={'width_ratios': [4, 1.5]}, constrained_layout=True)
                im = ax.imshow(np.zeros(shape), aspect='equal')
                ax.set_title(title, fontsize=12)
                
                max_val = legend_kwargs.get('max_val', 1.0)
                unit = legend_kwargs.get('unit', '')
                lx = legend_kwargs.get('label_x', 'X')
                ly = legend_kwargs.get('label_y', 'Y')
                is_n = legend_kwargs.get('is_norm', False)
                
                lim = max_val if not is_n else 1.0
                x = np.linspace(-lim, lim, 200)
                y = np.linspace(lim, -lim, 200) 
                Xg, Yg = np.meshgrid(x, y)
                mag = np.sqrt(Xg**2 + Yg**2)
                ang = np.mod(np.arctan2(Yg, Xg), 2*np.pi) / (2*np.pi)
                
                hsv = np.zeros((200, 200, 3))
                hsv[..., 0] = ang
                if is_n:
                    hsv[..., 1] = np.where(mag <= lim, 1.0, 0.0)
                    hsv[..., 2] = np.where(mag <= lim, 1.0, 1.0)
                else:
                    hsv[..., 1] = 1.0
                    hsv[..., 2] = np.clip(mag / (max_val + 1e-10), 0, 1)
                    
                rgb = plt.matplotlib.colors.hsv_to_rgb(hsv)
                #rgb[mag > lim] = 1.0 
                
                ax_w.imshow(rgb, extent=[-lim, lim, -lim, lim])
                ax_w.set_xlabel(f"{lx} ({unit})" if unit else lx, fontsize=10)
                ax_w.set_ylabel(f"{ly} ({unit})" if unit else ly, fontsize=10)
                ax_w.set_title("正交分量映射图例", fontsize=10, weight='bold')
                ax_w.tick_params(labelsize=8)
                ax_w.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
            else:
                fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
                im = ax.imshow(np.zeros(shape), cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
                ax.set_title(title, fontsize=12)
                if label:
                    cbar = fig.colorbar(im, ax=ax, shrink=0.85, aspect=25)
                    self._set_colorbar_ticks(cbar, np.array([vmin, vmax]), mm_per_pixel, vmin, vmax, label=label)
            
            self._set_dynamic_ticks(ax, shape, mm_per_pixel)
            return fig, ax, im

        # 准备输出
        saved_files = []

        h_vmin, h_vmax = np.percentile(h_t, self.p_low), np.percentile(h_t, self.p_high)
        
        # 🌟 强制使水位场的 Colorbar 关于 0 绝对对称
        h_abs_max = max(abs(h_vmin), abs(h_vmax))
        h_vmin, h_vmax = -h_abs_max, h_abs_max
        f_h, a_h, im_h = create_fig('seismic', h_vmin, h_vmax, "单帧瞬时水位形变", "水位高度 (mm)")
        im_h.set_data(h_t)
        p = os.path.join(run_dir, f'hfield_{timestamp}.jpg')
        f_h.savefig(p, dpi=150, bbox_inches='tight', pad_inches=0.02)
        saved_files.append(p)

        uv_mag = np.sqrt(u_t**2 + v_t**2)
        uv_vmax = np.percentile(uv_mag, self.p_high)
        ph_norm = np.mod(np.arctan2(v_t, u_t), 2*np.pi) / (2*np.pi)
        
        f_d, a_d, im_d = create_fig(None, None, None, "面内二维矢量位移场 (u, v)", is_2d=True, 
            legend_kwargs={'max_val': uv_vmax, 'unit': 'mm', 'label_x': '位移 u', 'label_y': '位移 v'})
        
        hsv_d = np.zeros((shape[0], shape[1], 3))
        hsv_d[..., 0] = ph_norm
        hsv_d[..., 1] = 1.0
        hsv_d[..., 2] = np.clip(uv_mag / (uv_vmax + 1e-10), 0, 1)
        im_d.set_data(plt.matplotlib.colors.hsv_to_rgb(hsv_d))
        p = os.path.join(run_dir, f'disp_{timestamp}.jpg')
        f_d.savefig(p, dpi=150, bbox_inches='tight', pad_inches=0.02)
        saved_files.append(p)

        f_dn, a_dn, im_dn = create_fig(None, None, None, "归一化位移场 (纯拓扑方向)", is_2d=True, 
            legend_kwargs={'max_val': 1.0, 'unit': '', 'label_x': 'u_norm', 'label_y': 'v_norm', 'is_norm': True})
        
        hsv_dn = np.zeros((shape[0], shape[1], 3))
        hsv_dn[..., 0] = ph_norm
        hsv_dn[..., 1] = 1.0
        hsv_dn[..., 2] = 1.0
        im_dn.set_data(plt.matplotlib.colors.hsv_to_rgb(hsv_dn))
        p = os.path.join(run_dir, f'norm_disp_{timestamp}.jpg')
        f_dn.savefig(p, dpi=150, bbox_inches='tight', pad_inches=0.02)
        saved_files.append(p)

        plt.close('all')

        np.savetxt(os.path.join(run_dir, f'hfield_matrix_{timestamp}.csv'), h_t, delimiter=',', fmt='%.5f')
        np.savetxt(os.path.join(run_dir, f'disp_u_matrix_{timestamp}.csv'), u_t, delimiter=',', fmt='%.5f')
        np.savetxt(os.path.join(run_dir, f'disp_v_matrix_{timestamp}.csv'), v_t, delimiter=',', fmt='%.5f')

        log_c = (f"===== 单帧静力学分析完成 =====\n"
                 f"静态参考图: {ref_name}\n"
                 f"动态形变图: {def_name}\n"
                 f"打包输出目录: {run_dir}\n"
                 f"已成功导出 {len(saved_files)} 张带有标准二维图例的物理图像。\n"
                 f"已成功导出 3 份原始物理矩阵 CSV 数据 (h, u, v)。")
        
        return self.write_log("SingleFrame", log_c, target_dir=run_dir)

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

        # 原代码的结尾部分：
        matplotlib.use('Agg', force=True)
        
        # 🌟 修改：提取图片名，并将日志重定向到 FindPixels_Results 文件夹
        ref_name = os.path.basename(self.ref_path) if self.ref_path else "未知"
        find_dir = os.path.join(self.out_dir, "FindPixels_Results")
        log_c = f"操作: 获取像素坐标\n静态参考图: {ref_name}\n采集点数: {len(points)}\n绝对坐标: {points}"
        
        return points, self.write_log("FindPixels", log_c, target_dir=find_dir)

    def measure_distance(self):
        import matplotlib
        matplotlib.use('TkAgg', force=True)
        import matplotlib.pyplot as plt
        import os, time

        if not self.ref_path or not self.def_path:
            raise ValueError("请先配置静态参考图和形变图路径！")

        # 1. 基础读取与像素刻度计算
        Iref = self._read_and_crop(self.ref_path)
        Idef = self._read_and_crop(self.def_path)
        mm_per_pixel = self._estimate_mm_per_pixel(Iref)
        cm_per_pixel = mm_per_pixel / 10.0

        # 2. 解调出水位场 (高度图) 用于测量背景
        u_px, v_px = self._fcd_demodulate_correct(Iref, Idef)
        # 🌟 终极修复：先切除 FCD 污染边缘，再对纯净场逆积分
        e = self.edge
        u_crop = u_px[e:-e, e:-e]
        v_crop = v_px[e:-e, e:-e]
        
        h_px_int = self._fftinvgrad(-u_crop, -v_crop)
        h_t = h_px_int * (mm_per_pixel**2) / self.H
        h = self._remove_background_surface(h_t, order=3)
        shape = h_t.shape

        # 3. 构建交互画板
        fig, ax = plt.subplots(figsize=(8, 6), num='FCD 交互式距离测量')
        h_vmin, h_vmax = np.percentile(h_t, self.p_low), np.percentile(h_t, self.p_high)
        im = ax.imshow(h_t, cmap='jet', vmin=h_vmin, vmax=h_vmax, aspect='equal')
        
        cbar = fig.colorbar(im, ax=ax, shrink=0.85, aspect=25)
        self._set_colorbar_ticks(cbar, np.array([h_vmin, h_vmax]), mm_per_pixel, h_vmin, h_vmax, label='水位高度 (mm)')
        self._set_dynamic_ticks(ax, shape, mm_per_pixel)

        # 4. 挂载交互测距工具并阻塞等待用户操作
        measurer = InteractiveMeasurer(ax, fig, cm_per_pixel)
        plt.show(block=True)
        matplotlib.use('Agg', force=True) 

        ref_name = os.path.basename(self.ref_path)
        def_name = os.path.basename(self.def_path)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.out_dir, "Measure_Results", f"Run_{timestamp}")

        # 5. 如果用户进行了测量，则保存结果
        if len(measurer.measurements) > 0:
            os.makedirs(run_dir, exist_ok=True)
            base_name = os.path.splitext(def_name)[0]
            
            ax.set_title("") 
            img_path = os.path.join(run_dir, f"{base_name}_height_measured_{timestamp}.png")
            fig.savefig(img_path, dpi=300, bbox_inches='tight')
            
            log_path = os.path.join(run_dir, f"Log_measure_cm_{base_name}_{timestamp}.txt")
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write("===== 处理日志 =====\n")
                f.write(f"处理时间: {timestamp}\n")
                f.write(f"静态参考图: {ref_name} ({self.ref_path})\n")
                f.write(f"动态形变图: {def_name} ({self.def_path})\n")
                f.write(f"裁剪参数: {self.crop}\n\n")
                
                f.write("===== 距离测量结果 =====\n")
                f.write(f"物理单位: cm (每像素 = {cm_per_pixel:.4f} cm)\n")
                f.write("测量时间\t序号\t点1坐标(px)\t点2坐标(px)\t距离(cm)\n")
                
                for m in measurer.measurements:
                    p1, p2 = m['p1'], m['p2']
                    f.write(f"{m['time']}\t{m['id']}\t[{p1[0]:.1f}, {p1[1]:.1f}]\t[{p2[0]:.1f}, {p2[1]:.1f}]\t{m['dist_cm']:.2f}\n")

            summary = [m['dist_cm'] for m in measurer.measurements]
            log_str = f"交互测距完成！\n静态参考图: {ref_name}\n动态形变图: {def_name}\n共测量 {len(summary)} 组距离。\n结果 (cm): {', '.join([f'{d:.2f}' for d in summary])}\n截图与日志已打包保存至输出目录。"
            return summary[-1] if summary else 0, self.write_log("Measure", log_str, target_dir=run_dir)
        else:
            log_str = f"交互测距已取消或未提取任何点。\n静态参考图: {ref_name}\n动态形变图: {def_name}"
            return 0, self.write_log("Measure_Cancel", log_str)

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

    def _process_frame_worker(self, def_path, Iref, mm_per_pixel):
        """线程池专属 Worker：负责极速解调单张物理场"""
        Idef = self._read_and_crop(def_path)
        u_px, v_px = self._fcd_demodulate_correct(Iref, Idef)
        
        e = self.edge
        u_crop = u_px[e:-e, e:-e]
        v_crop = v_px[e:-e, e:-e]
        
        h_px_int = self._fftinvgrad(-u_crop, -v_crop)
        
        h_t = h_px_int * (mm_per_pixel**2) / self.H
        h = self._remove_background_surface(h_t, order=3)
        
        u_t = u_crop * mm_per_pixel
        v_t = v_crop * mm_per_pixel
        
        return h, u_t, v_t

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
        frames = len(files) 
        
        # 🌟 极度重要：并发前的“缓存预热 (Cache Warm-up)”
        # 必须在主线程提前触发一次解析积分器和三阶曲面拟合器
        # 否则几打线程同时涌入生成缓存，会引发严重的竞争碰撞 (Race Condition)！
        print("正在预热矩阵缓存，激活代数加速引擎...")
        dummy_u = np.zeros((Iref.shape[0]-2*self.edge, Iref.shape[1]-2*self.edge))
        self._fftinvgrad(dummy_u, dummy_u)
        self._remove_background_surface(dummy_u, order=3)
        
        h_list = [None] * frames
        u_list = [None] * frames
        v_list = [None] * frames
        
        import concurrent.futures
        # 自动探测 CPU 核心数，设定并发上限 (留一两个核保证系统不卡死)
        max_workers = max(1, (os.cpu_count() or 4) - 1)
        print(f"🚀 启动 CPU 多核并发引擎 (并发数: {max_workers})...")
        
        # 🌟 并发处理核心
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 将每一帧封装为独立任务投入线程池
            future_to_idx = {
                executor.submit(self._process_frame_worker, os.path.join(self.seq_dir, files[i]), Iref, mm_per_pixel): i
                for i in range(frames)
            }
            
            # 动态收集结果 (谁先算完就先提取谁，但利用 idx 严格保证时序不乱)
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    h, u, v = future.result()
                    h_list[idx] = h
                    u_list[idx] = u
                    v_list[idx] = v
                except Exception as exc:
                    print(f"❌ 第 {idx} 帧处理异常: {exc}")

        import gc  # 🌟 引入垃圾回收模块

        # 🌟 内存绝杀 1：使用单精度 float32 构建基础矩阵，内存直接减半！
        h_stack = np.array(h_list, dtype=np.float32)
        u_stack = np.array(u_list, dtype=np.float32)
        v_stack = np.array(v_list, dtype=np.float32)
        shape = h_stack[0].shape

        # 极速释放多线程收集时产生的庞大临时列表
        del h_list, u_list, v_list
        gc.collect()

        # 减去均值
        h_stack -= np.mean(h_stack, axis=0)
        u_stack -= np.mean(u_stack, axis=0)
        v_stack -= np.mean(v_stack, axis=0)
        
        # 提取时域标准差静态包络
        amp_map = np.std(h_stack, axis=0) * np.sqrt(2.0)

        # 🌟 内存绝杀 2：计算 Hilbert 变换后立刻强转单精度复数 (complex64)
        # 并且算完一个立刻删除原矩阵！(因为复数矩阵的实部就是原矩阵，无需存两份)
        h_ana = np.conj(hilbert(h_stack, axis=0)).astype(np.complex64)
        del h_stack
        
        u_ana = np.conj(hilbert(u_stack, axis=0)).astype(np.complex64)
        del u_stack
        
        v_ana = np.conj(hilbert(v_stack, axis=0)).astype(np.complex64)
        del v_stack
        gc.collect()

        phase_w = np.mod(np.angle(h_ana), 2*np.pi)

        fps = getattr(self, 'fps', 30.0) 
        dt = 1.0 / fps if fps > 0 else 1.0 / 30.0

        # 🌟 内存绝杀 3：原地将单位转为米(m)，绝不创建 _m 结尾的新庞大矩阵
        h_ana /= 1000.0
        u_ana /= 1000.0
        v_ana /= 1000.0

        # 计算真实物理速度 (m/s)
        u_vel_m = np.gradient(u_ana, axis=0) / dt
        v_vel_m = np.gradient(v_ana, axis=0) / dt
        h_vel_m = np.gradient(h_ana, axis=0) / dt
        
        calc_spin = self.out_sz or self.out_s3d
        if calc_spin:
            sx = -np.imag(np.conj(v_ana) * h_vel_m - np.conj(h_ana) * v_vel_m) * 1e6
            sy =  np.imag(np.conj(h_ana) * u_vel_m - np.conj(u_ana) * h_vel_m) * 1e6
            sz = -np.imag(np.conj(u_ana) * v_vel_m - np.conj(v_ana) * u_vel_m) * 1e6
            
        # 🌟 重新提取百分位值（因为我们删了 stack，需要直接从 ana 的实部提取）
        h_real_mm = np.real(h_ana) * 1000.0
        h_vmin, h_vmax = np.percentile(h_real_mm, self.p_low), np.percentile(h_real_mm, self.p_high)
        h_abs_max = max(abs(h_vmin), abs(h_vmax))
        h_vmin, h_vmax = -h_abs_max, h_abs_max
        del h_real_mm
        gc.collect()

        amp_vmin, amp_vmax = np.percentile(amp_map, self.p_low), np.percentile(amp_map, self.p_high)
        if calc_spin:
            sz_vmin, sz_vmax = np.percentile(sz, self.p_low), np.percentile(sz, self.p_high)

        # ====== 动量流全局绝对缩放锁死逻辑 ======
        if self.out_mom:
            mpp_m = mm_per_pixel / 1000.0
            
            du_dy_m, du_dx_m = np.gradient(u_ana, axis=(1, 2))
            du_dy_m /= mpp_m; du_dx_m /= mpp_m
            
            dv_dy_m, dv_dx_m = np.gradient(v_ana, axis=(1, 2))
            dv_dy_m /= mpp_m; dv_dx_m /= mpp_m
            
            dw_dy_m, dw_dx_m = np.gradient(h_ana, axis=(1, 2))
            dw_dy_m /= mpp_m; dw_dx_m /= mpp_m
            
            Px_all = -np.real(np.conj(u_vel_m)*du_dx_m + np.conj(v_vel_m)*dv_dx_m + np.conj(h_vel_m)*dw_dx_m)
            Py_all = -np.real(np.conj(u_vel_m)*du_dy_m + np.conj(v_vel_m)*dv_dy_m + np.conj(h_vel_m)*dw_dy_m)
            
            # 🌟 内存绝杀 4：动量流算完，彻底删除极其占内存的空间梯度矩阵！
            del du_dy_m, du_dx_m, dv_dy_m, dv_dx_m, dw_dy_m, dw_dx_m
            gc.collect()
            
            global_max_p = np.percentile(np.sqrt(Px_all**2 + Py_all**2), 99.5)
            if global_max_p == 0 or np.isnan(global_max_p): global_max_p = 1e-12

            exp = np.floor(np.log10(global_max_p * 0.4))
            frac = (global_max_p * 0.4) / 10**exp
            if frac < 2: nice_frac = 1.0
            elif frac < 5: nice_frac = 2.0
            else: nice_frac = 5.0
            ref_len = nice_frac * 10**exp

            arrow_width_fraction = 0.1 * (self.q_scale / 4.0)
            fixed_scale = global_max_p / (arrow_width_fraction + 1e-12)

        def create_fig(cmap, vmin, vmax, title, label=None, is_2d=False, legend_kwargs=None):
            if is_2d:
                fig, (ax, ax_w) = plt.subplots(1, 2, figsize=(8.5, 5), gridspec_kw={'width_ratios': [4, 1.5]}, constrained_layout=True)
                im = ax.imshow(np.zeros(shape), aspect='equal')
                ax.set_title(title, fontsize=12)
                
                max_val = legend_kwargs.get('max_val', 1.0)
                unit = legend_kwargs.get('unit', '')
                lx = legend_kwargs.get('label_x', 'X')
                ly = legend_kwargs.get('label_y', 'Y')
                is_n = legend_kwargs.get('is_norm', False)
                
                lim = max_val if not is_n else 1.0
                x = np.linspace(-lim, lim, 200)
                y = np.linspace(lim, -lim, 200) 
                Xg, Yg = np.meshgrid(x, y)
                mag = np.sqrt(Xg**2 + Yg**2)
                ang = np.mod(np.arctan2(Yg, Xg), 2*np.pi) / (2*np.pi)
                
                hsv = np.zeros((200, 200, 3))
                hsv[..., 0] = ang
                if is_n:
                    hsv[..., 1] = np.where(mag <= lim, 1.0, 0.0)
                    hsv[..., 2] = np.where(mag <= lim, 1.0, 1.0)
                    rgb = plt.matplotlib.colors.hsv_to_rgb(hsv)
                    rgb[mag > lim] = 1.0

                else:
                    hsv[..., 1] = 1.0
                    hsv[..., 2] = np.clip(mag / (max_val + 1e-10), 0, 1)
                    rgb = plt.matplotlib.colors.hsv_to_rgb(hsv)
                
                ax_w.imshow(rgb, extent=[-lim, lim, -lim, lim])
                ax_w.set_xlabel(f"{lx} ({unit})" if unit else lx, fontsize=10)
                ax_w.set_ylabel(f"{ly} ({unit})" if unit else ly, fontsize=10)
                ax_w.set_title("正交分量映射图例", fontsize=10, weight='bold')
                ax_w.tick_params(labelsize=8)
                ax_w.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
            else:
                fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
                im = ax.imshow(np.zeros(shape), cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
                ax.set_title(title, fontsize=12)
                if label:
                    cbar = fig.colorbar(im, ax=ax, shrink=0.85, aspect=25)
                    if title == "解析相位角":
                        cbar.set_ticks([0, np.pi, 2*np.pi])
                        cbar.set_ticklabels(['0', 'π', '2π'])
                        cbar.set_label(label)
                    else:
                        self._set_colorbar_ticks(cbar, np.array([vmin, vmax]), mm_per_pixel, vmin, vmax, label=label)
            
            self._set_dynamic_ticks(ax, shape, mm_per_pixel)
            return fig, ax, im

        # 🌟 修改点：更新自旋相关图表的物理单位标注
        if self.out_hf: f_h, a_h, im_h = create_fig('seismic', h_vmin, h_vmax, "水位形变", "水位形变 (mm)")
        if self.out_amp:
            f_a, a_a, im_a = create_fig('hot', amp_vmin, amp_vmax, "全局水波振幅包络 (Amplitude)", "振幅强度 (mm)")
            im_a.set_data(amp_map)
            f_a.savefig(os.path.join(seq_out_dir, 'amplitude', 'Global_Amplitude_Envelope.jpg'), dpi=150, bbox_inches='tight', pad_inches=0.02)
            plt.close(f_a)
        if self.out_ph: f_p, a_p, im_p = create_fig('hsv', 0, 2*np.pi, "解析相位角", "相位 (rad)")
        if self.out_sz: f_sz, a_sz, im_sz = create_fig('viridis', sz_vmin, sz_vmax, "Z向自旋角动量", "自旋密度 ($mm^2/s$)")

        if self.out_disp or self.out_ndisp:
            # 🌟 修复：使用解析信号的实部无损还原原有的 stack 矩阵
            uv_mag_all = np.sqrt(np.real(u_ana)**2 + np.real(v_ana)**2) * 1000
            uv_vmax = np.percentile(uv_mag_all, self.p_high)
            del uv_mag_all
            gc.collect()

        if self.out_s3d:
            sxy_mag_all = np.sqrt(sx**2 + sy**2)
            sxy_vmax = np.percentile(sxy_mag_all, self.p_high)
            del sxy_mag_all
            gc.collect()

        if self.out_pa: 
            f_pa, a_pa, im_pa = create_fig(None, None, None, "彩色相位振幅复合场", is_2d=True, 
                legend_kwargs={'max_val': amp_vmax, 'unit': 'mm', 'label_x': 'Re(h)', 'label_y': 'Im(h)'})
        if self.out_disp: 
            f_d, a_d, im_d = create_fig(None, None, None, "面内二维矢量位移场 (u, v)", is_2d=True, 
                legend_kwargs={'max_val': uv_vmax, 'unit': 'mm', 'label_x': '位移 u', 'label_y': '位移 v'})
        if self.out_ndisp: 
            f_dn, a_dn, im_dn = create_fig(None, None, None, "归一化位移场 (纯拓扑方向)", is_2d=True, 
                legend_kwargs={'max_val': 1.0, 'unit': '', 'label_x': 'u_norm', 'label_y': 'v_norm', 'is_norm': True})
        if self.out_s3d: 
            f_s3, a_s3, im_s3 = create_fig(None, None, None, "横向自旋角动量场 (Sx, Sy)", is_2d=True, 
                legend_kwargs={'max_val': sxy_vmax, 'unit': '$mm^2/s$', 'label_x': 'Sx', 'label_y': 'Sy'})
        
        if self.out_mom:
            f_m, a_m, im_m = create_fig(None, None, None, "物理动量密度流场 (Momentum Flux)", is_2d=True, 
                legend_kwargs={'max_val': amp_vmax, 'unit': 'mm', 'label_x': 'Re(h)', 'label_y': 'Im(h)'})
            a_m.text(0.02, 1.06, "图例说明: \n[背景] 彩色相幅复合场h (亮度=振幅, 色相=相位)\n[箭头] 物理动量密度流方向与大小", 
                        transform=a_m.transAxes, fontsize=9, va='bottom', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # ====== 循环填入时间帧数据 ======
        for t in range(frames):
            tag = f"{t:03d}"
            
            if self.out_hf:
                im_h.set_data(np.real(h_ana[t]) * 1000.0)
                f_h.savefig(os.path.join(seq_out_dir, 'hfield', f'hfield_{tag}.jpg'), dpi=150, bbox_inches='tight', pad_inches=0.02)
            
            if self.out_ph:
                im_p.set_data(phase_w[t])
                f_p.savefig(os.path.join(seq_out_dir, 'phase', f'phase_{tag}.jpg'), dpi=150, bbox_inches='tight', pad_inches=0.02)
            
            if self.out_pa:
                pa_hsv = np.zeros((shape[0], shape[1], 3))
                pa_hsv[..., 0] = phase_w[t] / (2*np.pi)
                pa_hsv[..., 1] = 1.0
                pa_hsv[..., 2] = np.clip(amp_map / (amp_vmax + 1e-10), 0, 1) 
                im_pa.set_data(plt.matplotlib.colors.hsv_to_rgb(pa_hsv))
                f_pa.savefig(os.path.join(seq_out_dir, 'phaseamp', f'phaseamp_{tag}.jpg'), dpi=150, bbox_inches='tight', pad_inches=0.02)
            
            if self.out_disp or self.out_ndisp:
                u_r, v_r = np.real(u_ana[t]) * 1000.0, np.real(v_ana[t]) * 1000.0
                ph_norm = np.mod(np.arctan2(v_r, u_r), 2*np.pi) / (2*np.pi)
                
                if self.out_disp:
                    uv_mag = np.sqrt(u_r**2 + v_r**2)
                    hsv_d = np.zeros((shape[0], shape[1], 3))
                    hsv_d[..., 0] = ph_norm
                    hsv_d[..., 1] = 1.0
                    hsv_d[..., 2] = np.clip(uv_mag / (uv_vmax + 1e-10), 0, 1)
                    im_d.set_data(plt.matplotlib.colors.hsv_to_rgb(hsv_d))
                    f_d.savefig(os.path.join(seq_out_dir, 'displacement', f'disp_{tag}.jpg'), dpi=150, bbox_inches='tight', pad_inches=0.02)
                
                if self.out_ndisp:
                    hsv_dn = np.zeros((shape[0], shape[1], 3))
                    hsv_dn[..., 0] = ph_norm
                    hsv_dn[..., 1] = 1.0
                    hsv_dn[..., 2] = 1.0
                    im_dn.set_data(plt.matplotlib.colors.hsv_to_rgb(hsv_dn))
                    f_dn.savefig(os.path.join(seq_out_dir, 'norm_disp', f'norm_disp_{tag}.jpg'), dpi=150, bbox_inches='tight', pad_inches=0.02)
            
            if self.out_sz:
                im_sz.set_data(sz[t])
                f_sz.savefig(os.path.join(seq_out_dir, 'sz', f'sz_{tag}.jpg'), dpi=150, bbox_inches='tight', pad_inches=0.02)
            
            if self.out_s3d:
                s_ph = np.mod(np.arctan2(sy[t], sx[t]), 2*np.pi) / (2*np.pi)
                sxy_mag = np.sqrt(sx[t]**2 + sy[t]**2)
                hsv_s3 = np.zeros((shape[0], shape[1], 3))
                hsv_s3[..., 0] = s_ph
                hsv_s3[..., 1] = 1.0
                hsv_s3[..., 2] = np.clip(sxy_mag / (sxy_vmax + 1e-10), 0, 1)
                im_s3.set_data(plt.matplotlib.colors.hsv_to_rgb(hsv_s3))
                f_s3.savefig(os.path.join(seq_out_dir, 's3d', f's3d_{tag}.jpg'), dpi=150, bbox_inches='tight', pad_inches=0.02)
            
            if self.out_mom:
                pa_hsv = np.zeros((shape[0], shape[1], 3))
                pa_hsv[..., 0] = phase_w[t] / (2*np.pi)
                pa_hsv[..., 1] = 1.0
                pa_hsv[..., 2] = np.clip(amp_map / (amp_vmax + 1e-10), 0, 1)
                im_m.set_data(plt.matplotlib.colors.hsv_to_rgb(pa_hsv))
                
                a_m.set_xlim(0, shape[1]-1)
                a_m.set_ylim(shape[0]-1, 0)
                
                slc = slice(None, None, self.q_step)
                X_grid, Y_grid = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
                p_mag = np.sqrt(Px_all[t]**2 + Py_all[t]**2)
                mask = p_mag[slc, slc] > (global_max_p * 0.05) 
                
                q = a_m.quiver(X_grid[slc, slc][mask], Y_grid[slc, slc][mask], 
                            Px_all[t][slc, slc][mask], Py_all[t][slc, slc][mask], 
                            color='cyan', scale=fixed_scale, scale_units='width', angles='xy')
                
                # 🌟 修改点：动量流箭头参照的标注单位切换为 m²/s
                qk = a_m.quiverkey(q, X=0.85, Y=1.06, U=ref_len, 
                                   label=f'{ref_len:.1e} $m^2/s$', labelpos='E', coordinates='axes', color='red')
                
                f_m.savefig(os.path.join(seq_out_dir, 'momentum', f'momentum_{tag}.jpg'), dpi=150, bbox_inches='tight', pad_inches=0.02)
                
                q.remove()
                qk.remove()

        plt.close('all')

        # 原代码倒数两行：
        log_c = f"===== 序列分析完成 =====\n输出目录: {seq_out_dir}\n包含真实物理时间帧率: {fps} FPS\n共处理: {frames}帧\n"
        
        # 🌟 修改：直接传入 seq_out_dir，使日志储存在结果文件夹里
        return os.path.join(seq_out_dir, 'hfield'), self.write_log("ImageSeq", log_c, target_dir=seq_out_dir)
    
   # 🌟 修改签名：去掉了 amp, phase，加入了专属路径 calib_dir
    def run_calibration(self, calib_dir, fps, in_period):
        import matplotlib
        import matplotlib.pyplot as plt
        import json

        fps, in_period = float(fps), float(in_period)
        if not calib_dir or not os.path.exists(calib_dir):
            raise ValueError("未选择定标总目录！请选择包含 32 组子文件夹的 Calibration_RAW 根目录。")

        # 1. 解析目录结构
        levels = [0.1, 0.4, 0.7, 1.0]
        missing = []
        for ch in range(1, 9):
            for lvl in levels:
                if not os.path.exists(os.path.join(calib_dir, f"CH{ch}_Amp{lvl:.1f}")):
                    missing.append(f"CH{ch}_Amp{lvl:.1f}")
        if missing:
            raise ValueError(f"数据不完整！缺少以下文件夹:\n{missing[:5]}...")

        calib_out_dir = os.path.join(calib_dir, "Calibration_Results")
        os.makedirs(calib_out_dir, exist_ok=True)
        img_out_dir = os.path.join(calib_out_dir, "Height_Maps")
        os.makedirs(img_out_dir, exist_ok=True)
        csv_out_dir = os.path.join(calib_out_dir, "Time_Series_Data")
        os.makedirs(csv_out_dir, exist_ok=True)

        Iref = self._read_and_crop(self.ref_path)
        mm_per_pixel = self._estimate_mm_per_pixel(Iref)
        
        # ----------------------------------------------------
        # 步骤 1：全图基准平行线标定
        # ----------------------------------------------------
        matplotlib.use('TkAgg', force=True)
        
        fig, ax = plt.subplots(figsize=(8, 8), num='全局基准线设定')
        ax.imshow(Iref, cmap='gray')
        m_sel = MasterLineSelector(ax, fig)
        plt.show(block=True)
        if m_sel.start_pt is None: raise ValueError("操作取消。")
        p1, p2 = m_sel.start_pt, m_sel.end_pt

        # ----------------------------------------------------
        # 步骤 2：生成并提取 8 个喇叭的绝对声学中心
        # ----------------------------------------------------
        speaker_pts = []
        for ch in range(1, 9):
            folder = os.path.join(calib_dir, f"CH{ch}_Amp1.0")
            files = sorted([f for f in os.listdir(folder) if f.endswith(('.bmp', '.tiff'))])
            mid_idx = len(files) // 2
            
            Idef = self._read_and_crop(os.path.join(folder, files[mid_idx]))
            u_px, v_px = self._fcd_demodulate_correct(Iref, Idef)
            # 🌟 终极修复：定标时也必须先切边缘再积分，防止极值点被锯齿干扰
            e = self.edge
            u_crop = u_px[e:-e, e:-e]
            v_crop = v_px[e:-e, e:-e]
            h_px_int = self._fftinvgrad(-u_crop, -v_crop)
            h_map = h_px_int * (mm_per_pixel**2) / self.H
            h_map = self._remove_background_surface(h_map, order=3)

            h_vmin, h_vmax = h_map.max(), h_map.min()
            h_abs_max = max(abs(h_vmin), abs(h_vmax))
            h_vmin, h_vmax = -h_abs_max, h_abs_max
            
            plt.imsave(os.path.join(img_out_dir, f"CH{ch}_Amp1.0_MidFrame.png"), h_map, cmap='seismic')

            fig, ax = plt.subplots(figsize=(8, 8), num=f'标定喇叭 CH{ch}')
            ax.imshow(h_map, cmap='seismic', vmin = h_vmin, vmax = h_vmax)
            p_sel = OrthogonalPicker(ax, fig, p1, p2, f"CH{ch}")
            plt.show(block=True)
            
            if p_sel.target_pt is None: raise ValueError("标定中断。")
            speaker_pts.append((int(p_sel.target_pt[0]), int(p_sel.target_pt[1])))
            
        matplotlib.use('Agg', force=True) 

        # ----------------------------------------------------
        # 步骤 3：32组时序数据全量提取与滤波拟合引擎 (含锚点去抖动)
        # ----------------------------------------------------
        self.write_log("Calibration", "正在进行 32 组高速时序波形解调与正弦拟合，这需要一些时间，请稍候...")
        
        f_drive = 1000.0 / in_period
        omega_drive = 2.0 * np.pi * f_drive
        
        f_cutoff = f_drive * 0.3
        b, a = butter(4, f_cutoff, btype='high', fs=fps)

        amps_out = np.zeros((8, 4))
        phases_out = np.zeros((8, 4))

        for ch_idx in range(8):
            # 🌟 获取当前测试喇叭与锚点喇叭的精确像素坐标
            anchor_idx = 0 if ch_idx == 7 else 7
            px, py = speaker_pts[ch_idx]
            ax_px, ay_py = speaker_pts[anchor_idx]
            
            for lvl_idx, lvl in enumerate(levels):
                folder = os.path.join(calib_dir, f"CH{ch_idx+1}_Amp{lvl:.1f}")
                files = sorted([f for f in os.listdir(folder) if f.endswith(('.bmp', '.tiff'))])
                
                N_frames = len(files)
                t_arr = np.arange(N_frames) / fps
                h_arr = np.zeros(N_frames)
                h_anchor_arr = np.zeros(N_frames)
                
                # 极速逐帧提取单点高度
                for i, fname in enumerate(files):
                    Idef = self._read_and_crop(os.path.join(folder, fname))
                    u_px, v_px = self._fcd_demodulate_correct(Iref, Idef)
                    
                    e = self.edge
                    u_crop = u_px[e:-e, e:-e]
                    v_crop = v_px[e:-e, e:-e]
                    h_px_int = self._fftinvgrad(-u_crop, -v_crop)
                    h_full = h_px_int * (mm_per_pixel**2) / self.H
                    h_full = self._remove_background_surface(h_full, order=3)
                    
                    # 同时提取测试点与锚点的高度
                    h_arr[i] = h_full[py, px]
                    h_anchor_arr[i] = h_full[ay_py, ax_px]
                
                # 零相移高通滤波
                h_filt = filtfilt(b, a, h_arr)
                h_anchor_filt = filtfilt(b, a, h_anchor_arr)
                
                csv_path = os.path.join(csv_out_dir, f"CH{ch_idx+1}_Amp{lvl:.1f}_TimeSeries.csv")
                np.savetxt(csv_path, np.column_stack((t_arr, h_arr, h_filt)), 
                           delimiter=',', header="Time(s),Raw_Height(mm),Filtered_Height(mm)", comments='')

                # 🌟 内部拟合函数
                def get_fit(h_data):
                    p0 = [np.std(h_data)*np.sqrt(2), omega_drive, 0.0, 0.0]
                    bounds = ([0, omega_drive*0.95, -np.pi*2, -10], [100, omega_drive*1.05, np.pi*2, 10])
                    try:
                        popt, _ = curve_fit(sine_fit_func, t_arr, h_data, p0=p0, bounds=bounds)
                        return abs(popt[0]), popt[2]
                    except RuntimeError:
                        lo = np.exp(-1j * omega_drive * t_arr)
                        C = 2.0 * np.mean(h_data * lo)
                        return np.abs(C), np.angle(C)

                # 分别拟合测试信号与锚点信号
                fit_amp, fit_phase_raw = get_fit(h_filt)
                _, anchor_phase = get_fit(h_anchor_filt)

                # 🌟 共模差分去抖动：完美消除操作系统的随机软触发延迟！
                fit_phase = (fit_phase_raw - anchor_phase) % (2*np.pi)
                if fit_phase > np.pi: fit_phase -= 2*np.pi

                amps_out[ch_idx, lvl_idx] = fit_amp
                phases_out[ch_idx, lvl_idx] = fit_phase

        # ----------------------------------------------------
        # 步骤 4：生成 LUT 与报告
        # ----------------------------------------------------
        phases_out = np.unwrap(phases_out, axis=1)

        # 🌟 统一基准面：将 CH8 (以CH1为锚) 的时间基准转换到与其他通道 (以CH8为锚) 绝对一致
        delta_18_05 = np.interp(0.5, levels, phases_out[0, :])
        phases_out[7, :] += delta_18_05
        phases_out = np.unwrap(phases_out, axis=1)

        for ch in range(8):
            for lvl_idx in range(1, 4):
                if amps_out[ch, lvl_idx] <= amps_out[ch, lvl_idx-1]:
                    amps_out[ch, lvl_idx] = amps_out[ch, lvl_idx-1] + 1e-4

        global_max_amp = np.min(amps_out[:, 3])

        calib_data = {
            "Algorithm": "Time-Domain Sine Fitting (32-Step Sync)",
            "Global_Max_Amp_mm": float(global_max_amp),
            "Speakers": {}
        }

        for ch in range(8):
            calib_data["Speakers"][f"CH{ch+1}"] = {
                "v_in": [0.0] + levels,
                "amp_out": [0.0] + [float(x) for x in amps_out[ch]],
                "phase_out": [float(phases_out[ch, 0])] + [float(x) for x in phases_out[ch]] 
            }

        out_json = os.path.join(calib_out_dir, "speaker_lut_calibration.json")
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(calib_data, f, indent=4, ensure_ascii=False)
            
        log_c = f"===== 【时域正弦拟合】32阶梯定标完成 =====\n"
        # log_c += f"所有中间态热力图已存入: {img_out_dir}\n"
        # log_c += f"所有时域拟合数据已存入: {csv_out_dir}\n"
        # log_c += f"阵列不失真物理极限振幅: {global_max_amp:.4f} mm\n"
        log_c += f"定标 JSON 已导出至: {out_json}\n\n"
        log_c += f"系统定标完成！请在上位机重新加载此 JSON。"
        return self.write_log("Calibration", log_c)