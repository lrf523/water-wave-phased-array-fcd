import os
import time
import threading
import cv2
import numpy as np
import platform

try:
    import mvsdk
except ImportError:
    raise ImportError("❌ 找不到 mvsdk.py！请确保将其放入与此脚本相同的目录下。")

class MindVisionCamera:
    def __init__(self):
        self.hCamera = 0
        self.pFrameBuffer = 0
        self.is_opened = False
        
        self.frame_cache = []      
        self.is_recording = False
        self.grab_thread = None

    def open_camera(self):
        DevList = mvsdk.CameraEnumerateDevice()
        nDev = len(DevList)
        if nDev < 1:
            raise Exception("未找到任何迈德威视相机连接，请检查 USB/网线！")

        DevInfo = DevList[0]
        
        try:
            self.hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
            cap = mvsdk.CameraGetCapability(self.hCamera)
            
            monoCamera = (cap.sIspCapacity.bMonoSensor != 0)
            if monoCamera:
                mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
            else:
                mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
            
            self.pFrameBuffer = mvsdk.CameraAlignMalloc(cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * 3, 16)
            
            mvsdk.CameraSetTriggerMode(self.hCamera, 0)
            
            # 默认启动时开启自动曝光
            mvsdk.CameraSetAeState(self.hCamera, 1)
            
            mvsdk.CameraPlay(self.hCamera)
            self.is_opened = True
            print("📷 迈德威视相机初始化成功！")
            
        except Exception as e:
            raise Exception(f"相机初始化失败: {str(e)}")

    # 🌟 新增：动态曝光控制引擎
    def set_exposure(self, exposure_ms):
        if not self.is_opened: return
        try:
            if exposure_ms <= 0:
                mvsdk.CameraSetAeState(self.hCamera, 1) # 开启自动曝光
                print("📷 相机已切换至 [自动曝光] 模式。")
            else:
                mvsdk.CameraSetAeState(self.hCamera, 0) # 关闭自动曝光
                mvsdk.CameraSetExposureTime(self.hCamera, int(exposure_ms * 1000)) # 迈德威视底层单位为微秒
                print(f"📷 相机曝光已强力锁定为 [{exposure_ms} ms]。")
        except Exception as e:
            print(f"⚠️ 曝光设置失败: {e}")

    def close_camera(self):
        if self.is_opened:
            mvsdk.CameraUnInit(self.hCamera)
            mvsdk.CameraAlignFree(self.pFrameBuffer)
            self.is_opened = False
            print("📷 相机已安全关闭。")

    def _grab_loop(self, target_fps, duration_sec):
        self.frame_cache.clear()
        target_frames = int(target_fps * duration_sec)
        
        # 🌟 新增：计算每一帧的标准间隔时间 (例如 30FPS 就是每帧 0.0333秒)
        frame_interval = 1.0 / target_fps 
        
        start_time = time.perf_counter()
        frames_grabbed = 0
        timeout_count = 0
        
        while self.is_recording and frames_grabbed < target_frames:
            # 🌟 新增：记录当前帧开始抓取的时刻
            loop_start = time.perf_counter() 
            
            try:
                pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self.hCamera, 200)
                mvsdk.CameraImageProcess(self.hCamera, pRawData, self.pFrameBuffer, FrameHead)
                mvsdk.CameraReleaseImageBuffer(self.hCamera, pRawData)
                
                if platform.system() == "Windows":
                    mvsdk.CameraFlipFrameBuffer(self.pFrameBuffer, FrameHead, 1)
                
                channels = 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3
                w = FrameHead.iWidth
                h = FrameHead.iHeight
                
                frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(self.pFrameBuffer)
                frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((h, w, channels))
                
                if channels == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                elif channels == 1:
                    frame = frame.reshape((h, w))
                
                self.frame_cache.append(frame.copy())
                frames_grabbed += 1
                timeout_count = 0 
                
            except mvsdk.CameraException as e:
                timeout_count += 1
                if timeout_count > 15: 
                    print(f"⚠️ 相机连续获取失败，抓图意外终止。已获取 {frames_grabbed} 帧。")
                    break
            
            # =========================================================
            # 🌟 核心修复：严格的物理帧率节拍器！
            # 算算抓这一帧和处理它花了多少时间。如果这台电脑处理得太快，
            # 就强行让线程睡一会儿，把剩下的时间补齐，凑够完整的 1/30 秒！
            # =========================================================
            elapsed = time.perf_counter() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            
        actual_duration = time.perf_counter() - start_time
        print(f"✅ 获取完毕。实际获取 {len(self.frame_cache)} 帧，耗时 {actual_duration:.3f} 秒。")
        self.is_recording = False

    def start_recording(self, target_fps=30, duration_sec=1.5):
        if not self.is_opened:
            raise Exception("相机未打开！")
        
        self.is_recording = True
        self.grab_thread = threading.Thread(target=self._grab_loop, args=(target_fps, duration_sec), daemon=True)
        self.grab_thread.start()

    def wait_and_save(self, save_dir, prefix="frame"):
        if self.grab_thread and self.grab_thread.is_alive():
            self.grab_thread.join() 
            
        if not self.frame_cache: return
            
        os.makedirs(save_dir, exist_ok=True)
        print(f"💾 后台落盘 {len(self.frame_cache)} 张图片至 {save_dir} ...")
        
        for i, frame in enumerate(self.frame_cache):
            filename = os.path.join(save_dir, f"{i:04d}.tiff") 
            encode_params = [int(cv2.IMWRITE_TIFF_COMPRESSION), 5] 
            is_success, im_buf = cv2.imencode(".tiff", frame, encode_params)
            if is_success: im_buf.tofile(filename)
            
        self.frame_cache.clear()