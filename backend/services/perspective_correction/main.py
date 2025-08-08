"""
透视校正服务
负责透视变换、网格坐标转换、像素-米比例转换
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import os
import asyncio
import httpx
from datetime import datetime
import base64
from typing import List, Dict, Any, Optional, Tuple
import json
import threading
import redis

app = FastAPI(title="Perspective Correction Service", version="1.0.0")

class PerspectiveCorrector:
    def __init__(self):
        self.transformation_matrix = None
        self.inverse_matrix = None
        self.calibration_points = []
        self.real_world_points = []
        self.pixel_to_meter_ratio = 1.0
        self.grid_size = 1.0  # 网格大小(米)
        self.is_calibrated = False
        self.lock = threading.Lock()
        
        # 默认的校准点（需要根据实际情况调整）
        self.default_calibration = {
            "image_points": [
                [100, 400],  # 左下
                [540, 400],  # 右下  
                [540, 200],  # 右上
                [100, 200]   # 左上
            ],
            "real_world_points": [
                [0, 0],      # 左下 (0,0)
                [4, 0],      # 右下 (4米,0)
                [4, 3],      # 右上 (4米,3米)
                [0, 3]       # 左上 (0,3米)
            ]
        }
        
    def set_calibration_points(self, image_points: List[List[float]], real_world_points: List[List[float]]):
        """设置校准点"""
        if len(image_points) != 4 or len(real_world_points) != 4:
            raise ValueError("需要提供4个校准点")
        
        with self.lock:
            self.calibration_points = np.array(image_points, dtype=np.float32)
            self.real_world_points = np.array(real_world_points, dtype=np.float32)
            
            # 计算透视变换矩阵
            self.transformation_matrix = cv2.getPerspectiveTransform(
                self.calibration_points, self.real_world_points
            )
            self.inverse_matrix = cv2.getPerspectiveTransform(
                self.real_world_points, self.calibration_points
            )
            
            # 计算像素到米的比例
            self._calculate_pixel_ratio()
            
            self.is_calibrated = True
            print("透视校正矩阵已更新")
    
    def _calculate_pixel_ratio(self):
        """计算像素到米的比例"""
        if len(self.calibration_points) >= 2 and len(self.real_world_points) >= 2:
            # 使用第一条边计算比例
            pixel_dist = np.linalg.norm(self.calibration_points[1] - self.calibration_points[0])
            real_dist = np.linalg.norm(self.real_world_points[1] - self.real_world_points[0])
            
            if pixel_dist > 0:
                self.pixel_to_meter_ratio = real_dist / pixel_dist
    
    def pixel_to_real_world(self, pixel_points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """将像素坐标转换为真实世界坐标"""
        if not self.is_calibrated:
            raise ValueError("透视校正未校准")
        
        pixel_array = np.array(pixel_points, dtype=np.float32).reshape(-1, 1, 2)
        real_world_array = cv2.perspectiveTransform(pixel_array, self.transformation_matrix)
        
        return [(float(pt[0][0]), float(pt[0][1])) for pt in real_world_array]
    
    def real_world_to_pixel(self, real_world_points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """将真实世界坐标转换为像素坐标"""
        if not self.is_calibrated:
            raise ValueError("透视校正未校准")
        
        real_array = np.array(real_world_points, dtype=np.float32).reshape(-1, 1, 2)
        pixel_array = cv2.perspectiveTransform(real_array, self.inverse_matrix)
        
        return [(float(pt[0][0]), float(pt[0][1])) for pt in pixel_array]
    
    def apply_perspective_transform(self, image: np.ndarray, output_size: Tuple[int, int] = (800, 600)) -> np.ndarray:
        """应用透视变换到图像"""
        if not self.is_calibrated:
            raise ValueError("透视校正未校准")
        
        # 定义输出图像的角点（鸟瞰图）
        output_points = np.array([
            [0, output_size[1]],           # 左下
            [output_size[0], output_size[1]], # 右下
            [output_size[0], 0],           # 右上
            [0, 0]                         # 左上
        ], dtype=np.float32)
        
        # 计算从输入图像到输出图像的变换矩阵
        transform_matrix = cv2.getPerspectiveTransform(self.calibration_points, output_points)
        
        # 应用透视变换
        warped_image = cv2.warpPerspective(image, transform_matrix, output_size)
        
        return warped_image
    
    def draw_grid(self, image: np.ndarray, grid_size: float = 1.0) -> np.ndarray:
        """在图像上绘制网格"""
        result_image = image.copy()
        
        if not self.is_calibrated:
            return result_image
        
        # 获取图像尺寸
        height, width = image.shape[:2]
        
        # 计算网格范围
        max_x = max(self.real_world_points[:, 0])
        max_y = max(self.real_world_points[:, 1])
        
        # 绘制垂直网格线
        for x in np.arange(0, max_x + grid_size, grid_size):
            start_point = (x, 0)
            end_point = (x, max_y)
            
            # 转换为像素坐标
            pixel_points = self.real_world_to_pixel([start_point, end_point])
            
            if len(pixel_points) == 2:
                pt1 = (int(pixel_points[0][0]), int(pixel_points[0][1]))
                pt2 = (int(pixel_points[1][0]), int(pixel_points[1][1]))
                
                # 确保点在图像范围内
                if (0 <= pt1[0] < width and 0 <= pt1[1] < height and
                    0 <= pt2[0] < width and 0 <= pt2[1] < height):
                    cv2.line(result_image, pt1, pt2, (0, 255, 0), 1)
        
        # 绘制水平网格线
        for y in np.arange(0, max_y + grid_size, grid_size):
            start_point = (0, y)
            end_point = (max_x, y)
            
            # 转换为像素坐标
            pixel_points = self.real_world_to_pixel([start_point, end_point])
            
            if len(pixel_points) == 2:
                pt1 = (int(pixel_points[0][0]), int(pixel_points[0][1]))
                pt2 = (int(pixel_points[1][0]), int(pixel_points[1][1]))
                
                # 确保点在图像范围内
                if (0 <= pt1[0] < width and 0 <= pt1[1] < height and
                    0 <= pt2[0] < width and 0 <= pt2[1] < height):
                    cv2.line(result_image, pt1, pt2, (0, 255, 0), 1)
        
        # 绘制校准点
        for i, point in enumerate(self.calibration_points):
            pt = (int(point[0]), int(point[1]))
            cv2.circle(result_image, pt, 5, (0, 0, 255), -1)
            cv2.putText(result_image, f"P{i+1}", (pt[0]+10, pt[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return result_image
    
    def get_grid_coordinates(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        """获取像素点对应的网格坐标"""
        if not self.is_calibrated:
            raise ValueError("透视校正未校准")
        
        real_coords = self.pixel_to_real_world([(pixel_x, pixel_y)])
        return real_coords[0] if real_coords else (0, 0)

# 全局校正器实例
corrector = PerspectiveCorrector()

# Redis连接
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', '6379')),
    decode_responses=True
)

@app.on_event("startup")
async def startup_event():
    """启动时加载保存的校准数据"""
    try:
        # 尝试从Redis加载校准数据
        calibration_data = redis_client.get("perspective_calibration")
        if calibration_data:
            data = json.loads(calibration_data)
            corrector.set_calibration_points(
                data["image_points"],
                data["real_world_points"]
            )
            print("已加载保存的透视校正配置")
        else:
            # 使用默认校准
            corrector.set_calibration_points(
                corrector.default_calibration["image_points"],
                corrector.default_calibration["real_world_points"]
            )
            print("使用默认透视校正配置")
    except Exception as e:
        print(f"加载透视校正配置失败: {e}")

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "is_calibrated": corrector.is_calibrated,
        "pixel_to_meter_ratio": corrector.pixel_to_meter_ratio,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/calibration/set_points")
async def set_calibration_points(calibration_data: Dict[str, Any]):
    """设置校准点"""
    try:
        image_points = calibration_data.get("image_points", [])
        real_world_points = calibration_data.get("real_world_points", [])
        
        if len(image_points) != 4 or len(real_world_points) != 4:
            raise HTTPException(status_code=400, detail="需要提供4个校准点")
        
        corrector.set_calibration_points(image_points, real_world_points)
        
        # 保存到Redis
        redis_client.set("perspective_calibration", json.dumps(calibration_data))
        
        return {
            "status": "success",
            "message": "校准点已设置",
            "pixel_to_meter_ratio": corrector.pixel_to_meter_ratio,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"设置校准点失败: {str(e)}")

@app.get("/calibration/points")
async def get_calibration_points():
    """获取当前校准点"""
    if not corrector.is_calibrated:
        raise HTTPException(status_code=404, detail="未设置校准点")
    
    return {
        "image_points": corrector.calibration_points.tolist(),
        "real_world_points": corrector.real_world_points.tolist(),
        "pixel_to_meter_ratio": corrector.pixel_to_meter_ratio,
        "is_calibrated": corrector.is_calibrated,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/calibration/matrix")
async def get_transformation_matrix():
    """获取透视变换矩阵"""
    if not corrector.is_calibrated:
        raise HTTPException(status_code=404, detail="未校准透视变换")
    
    return {
        "transformation_matrix": corrector.transformation_matrix.tolist(),
        "inverse_matrix": corrector.inverse_matrix.tolist(),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/grid_view")
async def get_grid_view():
    """获取当前帧的网格视图"""
    try:
        # 从视频采集服务获取当前帧
        video_service_url = f"http://{os.getenv('VIDEO_CAPTURE_HOST', 'localhost')}:{os.getenv('VIDEO_CAPTURE_PORT', '8001')}"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{video_service_url}/frame/current")
            
            if response.status_code == 200:
                frame_data = response.json()
                
                # 解码base64图像
                image_data = base64.b64decode(frame_data["image"])
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # 绘制网格
                grid_image = corrector.draw_grid(image, corrector.grid_size)
                
                # 编码为base64
                ret, buffer = cv2.imencode('.jpg', grid_image)
                if ret:
                    grid_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    return {
                        "grid_image": grid_base64,
                        "original_timestamp": frame_data["timestamp"],
                        "grid_timestamp": datetime.now().isoformat(),
                        "grid_size": corrector.grid_size,
                        "is_calibrated": corrector.is_calibrated
                    }
                else:
                    raise HTTPException(status_code=500, detail="图像编码失败")
            else:
                raise HTTPException(status_code=503, detail="无法获取视频帧")
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成网格视图失败: {str(e)}")

@app.get("/bird_eye_view")
async def get_bird_eye_view():
    """获取鸟瞰图视图"""
    try:
        if not corrector.is_calibrated:
            raise HTTPException(status_code=400, detail="透视校正未校准")
        
        # 从视频采集服务获取当前帧
        video_service_url = f"http://{os.getenv('VIDEO_CAPTURE_HOST', 'localhost')}:{os.getenv('VIDEO_CAPTURE_PORT', '8001')}"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{video_service_url}/frame/current")
            
            if response.status_code == 200:
                frame_data = response.json()
                
                # 解码base64图像
                image_data = base64.b64decode(frame_data["image"])
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # 应用透视变换
                bird_eye_image = corrector.apply_perspective_transform(image)
                
                # 编码为base64
                ret, buffer = cv2.imencode('.jpg', bird_eye_image)
                if ret:
                    bird_eye_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    return {
                        "bird_eye_image": bird_eye_base64,
                        "original_timestamp": frame_data["timestamp"],
                        "transform_timestamp": datetime.now().isoformat(),
                        "output_size": bird_eye_image.shape[:2][::-1]  # (width, height)
                    }
                else:
                    raise HTTPException(status_code=500, detail="图像编码失败")
            else:
                raise HTTPException(status_code=503, detail="无法获取视频帧")
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成鸟瞰图失败: {str(e)}")

@app.post("/convert/pixel_to_world")
async def pixel_to_world_coordinates(points: List[Dict[str, float]]):
    """将像素坐标转换为世界坐标"""
    try:
        if not corrector.is_calibrated:
            raise HTTPException(status_code=400, detail="透视校正未校准")
        
        pixel_points = [(point["x"], point["y"]) for point in points]
        world_points = corrector.pixel_to_real_world(pixel_points)
        
        result = []
        for i, (world_x, world_y) in enumerate(world_points):
            result.append({
                "original_pixel": points[i],
                "world_coordinates": {"x": world_x, "y": world_y},
                "grid_cell": {
                    "row": int(world_y // corrector.grid_size),
                    "col": int(world_x // corrector.grid_size)
                }
            })
        
        return {
            "conversions": result,
            "grid_size": corrector.grid_size,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"坐标转换失败: {str(e)}")

@app.post("/convert/world_to_pixel")
async def world_to_pixel_coordinates(points: List[Dict[str, float]]):
    """将世界坐标转换为像素坐标"""
    try:
        if not corrector.is_calibrated:
            raise HTTPException(status_code=400, detail="透视校正未校准")
        
        world_points = [(point["x"], point["y"]) for point in points]
        pixel_points = corrector.real_world_to_pixel(world_points)
        
        result = []
        for i, (pixel_x, pixel_y) in enumerate(pixel_points):
            result.append({
                "original_world": points[i],
                "pixel_coordinates": {"x": pixel_x, "y": pixel_y}
            })
        
        return {
            "conversions": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"坐标转换失败: {str(e)}")

@app.post("/settings/grid_size")
async def update_grid_size(grid_size: float):
    """更新网格大小"""
    if grid_size <= 0:
        raise HTTPException(status_code=400, detail="网格大小必须大于0")
    
    corrector.grid_size = grid_size
    
    return {
        "status": "success",
        "new_grid_size": grid_size,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/settings")
async def get_current_settings():
    """获取当前设置"""
    return {
        "grid_size": corrector.grid_size,
        "pixel_to_meter_ratio": corrector.pixel_to_meter_ratio,
        "is_calibrated": corrector.is_calibrated,
        "calibration_points_count": len(corrector.calibration_points) if corrector.is_calibrated else 0,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/calibration/reset")
async def reset_calibration():
    """重置校准到默认值"""
    try:
        corrector.set_calibration_points(
            corrector.default_calibration["image_points"],
            corrector.default_calibration["real_world_points"]
        )
        
        # 清除Redis中的校准数据
        redis_client.delete("perspective_calibration")
        
        return {
            "status": "success",
            "message": "校准已重置为默认值",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重置校准失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.getenv("PERSPECTIVE_CORRECTION_HOST", "localhost"),
        port=int(os.getenv("PERSPECTIVE_CORRECTION_PORT", "8003"))
    )
