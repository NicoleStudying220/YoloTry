"""
视频采集服务
负责获取摄像头流、处理视频数据、提供实时视频流
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import cv2
import asyncio
import threading
import queue
import os
from datetime import datetime
import numpy as np
from typing import Optional
import time

app = FastAPI(title="Video Capture Service", version="1.0.0")

class VideoCapture:
    def __init__(self, source=0):
        self.source = source
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=10)
        self.is_running = False
        self.current_frame = None
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.lock = threading.Lock()
        
    def start(self):
        """启动视频采集"""
        try:
            # 尝试打开摄像头
            if isinstance(self.source, str) and self.source.startswith(('rtsp://', 'http://')):
                # RTSP或HTTP流
                self.cap = cv2.VideoCapture(self.source)
            else:
                # 本地摄像头
                self.cap = cv2.VideoCapture(int(self.source))
            
            if not self.cap.isOpened():
                raise Exception(f"无法打开视频源: {self.source}")
            
            # 设置摄像头参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_running = True
            
            # 启动采集线程
            self.capture_thread = threading.Thread(target=self._capture_frames)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            
            print(f"视频采集已启动，源: {self.source}")
            return True
            
        except Exception as e:
            print(f"启动视频采集失败: {e}")
            return False
    
    def stop(self):
        """停止视频采集"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        print("视频采集已停止")
    
    def _capture_frames(self):
        """采集帧的后台线程"""
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.current_frame = frame.copy()
                    self.frame_count += 1
                    
                    # 计算FPS
                    current_time = time.time()
                    if current_time - self.last_fps_time >= 1.0:
                        self.fps = self.frame_count / (current_time - self.last_fps_time)
                        self.frame_count = 0
                        self.last_fps_time = current_time
                
                # 将帧添加到队列（非阻塞）
                try:
                    self.frame_queue.put(frame, block=False)
                except queue.Full:
                    # 队列满了，丢弃最旧的帧
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put(frame, block=False)
                    except queue.Empty:
                        pass
            else:
                print("无法读取视频帧")
                time.sleep(0.1)
    
    def get_frame(self):
        """获取最新帧"""
        with self.lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def get_frame_from_queue(self):
        """从队列获取帧"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_stream_frame(self):
        """获取用于流传输的帧（JPEG编码）"""
        frame = self.get_frame()
        if frame is not None:
            # 编码为JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                return buffer.tobytes()
        return None
    
    def get_info(self):
        """获取视频信息"""
        if self.cap and self.cap.isOpened():
            return {
                "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": self.fps,
                "source": self.source,
                "is_running": self.is_running,
                "frame_count": self.frame_count
            }
        return None

# 全局视频采集实例
video_capture = VideoCapture(source=os.getenv('CAMERA_URL', '0'))

@app.on_event("startup")
async def startup_event():
    """启动时初始化视频采集"""
    if not video_capture.start():
        print("警告: 视频采集启动失败")

@app.on_event("shutdown")
async def shutdown_event():
    """关闭时清理资源"""
    video_capture.stop()

@app.get("/health")
async def health_check():
    """健康检查"""
    info = video_capture.get_info()
    return {
        "status": "healthy" if info and info["is_running"] else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "video_info": info
    }

@app.get("/stream")
async def video_stream():
    """实时视频流"""
    def generate_frames():
        while True:
            frame_bytes = video_capture.get_stream_frame()
            if frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                # 如果没有帧，等待一下
                time.sleep(0.03)  # 约30fps
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/frame/current")
async def get_current_frame():
    """获取当前帧（JSON格式，包含base64编码的图像）"""
    frame = video_capture.get_frame()
    if frame is not None:
        # 编码为JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            import base64
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            return {
                "timestamp": datetime.now().isoformat(),
                "image": img_base64,
                "width": frame.shape[1],
                "height": frame.shape[0]
            }
    
    raise HTTPException(status_code=404, detail="无法获取当前帧")

@app.get("/info")
async def get_video_info():
    """获取视频信息"""
    info = video_capture.get_info()
    if info:
        return {
            "timestamp": datetime.now().isoformat(),
            **info
        }
    
    raise HTTPException(status_code=503, detail="视频采集服务不可用")

@app.post("/camera/switch")
async def switch_camera(source: str):
    """切换摄像头源"""
    try:
        # 停止当前采集
        video_capture.stop()
        
        # 切换到新源
        global video_capture
        video_capture = VideoCapture(source)
        
        if video_capture.start():
            return {
                "status": "success",
                "message": f"已切换到摄像头源: {source}",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail=f"无法切换到摄像头源: {source}")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"切换摄像头失败: {str(e)}")

@app.post("/capture/snapshot")
async def capture_snapshot():
    """拍摄快照"""
    frame = video_capture.get_frame()
    if frame is not None:
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{timestamp}.jpg"
        filepath = os.path.join("snapshots", filename)
        
        # 确保目录存在
        os.makedirs("snapshots", exist_ok=True)
        
        # 保存图像
        cv2.imwrite(filepath, frame)
        
        return {
            "status": "success",
            "filename": filename,
            "filepath": filepath,
            "timestamp": datetime.now().isoformat()
        }
    
    raise HTTPException(status_code=404, detail="无法获取当前帧进行快照")

@app.get("/camera/settings")
async def get_camera_settings():
    """获取摄像头设置"""
    if video_capture.cap and video_capture.cap.isOpened():
        settings = {
            "brightness": video_capture.cap.get(cv2.CAP_PROP_BRIGHTNESS),
            "contrast": video_capture.cap.get(cv2.CAP_PROP_CONTRAST),
            "saturation": video_capture.cap.get(cv2.CAP_PROP_SATURATION),
            "hue": video_capture.cap.get(cv2.CAP_PROP_HUE),
            "gain": video_capture.cap.get(cv2.CAP_PROP_GAIN),
            "exposure": video_capture.cap.get(cv2.CAP_PROP_EXPOSURE)
        }
        return settings
    
    raise HTTPException(status_code=503, detail="摄像头不可用")

@app.post("/camera/settings")
async def update_camera_settings(settings: dict):
    """更新摄像头设置"""
    if not video_capture.cap or not video_capture.cap.isOpened():
        raise HTTPException(status_code=503, detail="摄像头不可用")
    
    updated = {}
    setting_map = {
        "brightness": cv2.CAP_PROP_BRIGHTNESS,
        "contrast": cv2.CAP_PROP_CONTRAST,
        "saturation": cv2.CAP_PROP_SATURATION,
        "hue": cv2.CAP_PROP_HUE,
        "gain": cv2.CAP_PROP_GAIN,
        "exposure": cv2.CAP_PROP_EXPOSURE
    }
    
    for setting_name, value in settings.items():
        if setting_name in setting_map:
            try:
                video_capture.cap.set(setting_map[setting_name], float(value))
                updated[setting_name] = value
            except Exception as e:
                print(f"无法设置 {setting_name}: {e}")
    
    return {
        "status": "success",
        "updated_settings": updated,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.getenv("VIDEO_CAPTURE_HOST", "localhost"),
        port=int(os.getenv("VIDEO_CAPTURE_PORT", "8001"))
    )
