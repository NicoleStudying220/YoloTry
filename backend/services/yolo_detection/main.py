"""
YOLO检测服务
负责物体检测、障碍物识别和分类
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from ultralytics import YOLO
import os
import asyncio
import httpx
from datetime import datetime
import base64
from typing import List, Dict, Any, Optional
import threading
import time
import redis
import json

app = FastAPI(title="YOLO Detection Service", version="1.0.0")

class YOLODetector:
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.5):
        self.model_path = model_path or "yolov8n.pt"
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.is_initialized = False
        self.detection_history = []
        self.lock = threading.Lock()
        
        # 导航相关的物体类别映射
        self.navigation_classes = {
            0: "人员",
            1: "自行车", 
            2: "汽车",
            3: "摩托车",
            5: "公交车",
            7: "卡车",
            9: "交通灯",
            11: "停止标志",
            15: "椅子",
            16: "沙发",
            17: "盆栽植物",
            18: "床",
            19: "餐桌",
            20: "马桶",
            39: "瓶子",
            41: "杯子",
            56: "椅子",
            57: "沙发",
            58: "盆栽植物",
            59: "床",
            60: "餐桌",
            61: "马桶",
            62: "电视",
            63: "笔记本电脑",
            64: "鼠标",
            65: "遥控器",
            66: "键盘",
            67: "手机"
        }
        
        # 危险等级定义
        self.danger_levels = {
            "高危险": ["汽车", "卡车", "公交车", "摩托车"],
            "中危险": ["人员", "自行车", "交通灯", "停止标志"],
            "低危险": ["椅子", "桌子", "盆栽植物", "瓶子"],
            "可忽略": ["杯子", "遥控器", "键盘", "鼠标"]
        }
    
    def initialize(self):
        """初始化YOLO模型"""
        try:
            print(f"正在加载YOLO模型: {self.model_path}")
            self.model = YOLO(self.model_path)
            self.is_initialized = True
            print("YOLO模型加载成功")
            return True
        except Exception as e:
            print(f"YOLO模型加载失败: {e}")
            return False
    
    def detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """检测图像中的物体"""
        if not self.is_initialized:
            if not self.initialize():
                return []
        
        try:
            # 运行YOLO检测
            results = self.model(image, conf=self.confidence_threshold)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # 获取检测结果
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # 获取类别名称
                        class_name = self.navigation_classes.get(class_id, f"未知类别({class_id})")
                        
                        # 计算中心点和尺寸
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1
                        
                        # 估算距离（基于物体大小，需要根据实际情况调整）
                        estimated_distance = self._estimate_distance(width, height, class_name)
                        
                        # 确定危险等级
                        danger_level = self._get_danger_level(class_name)
                        
                        detection = {
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": round(confidence, 3),
                            "bbox": {
                                "x1": int(x1), "y1": int(y1),
                                "x2": int(x2), "y2": int(y2),
                                "center_x": int(center_x), "center_y": int(center_y),
                                "width": int(width), "height": int(height)
                            },
                            "estimated_distance": round(estimated_distance, 2),
                            "danger_level": danger_level,
                            "timestamp": datetime.now().isoformat()
                        }
                        detections.append(detection)
            
            # 按危险等级和距离排序
            detections.sort(key=lambda x: (
                self._danger_priority(x["danger_level"]),
                x["estimated_distance"]
            ))
            
            return detections
            
        except Exception as e:
            print(f"物体检测失败: {e}")
            return []
    
    def _estimate_distance(self, width: float, height: float, class_name: str) -> float:
        """基于物体大小估算距离（简化算法）"""
        # 这里是一个简化的距离估算算法
        # 实际应用中需要根据相机参数和物体实际尺寸进行校准
        
        object_sizes = {
            "人员": 1.7,  # 平均身高(米)
            "汽车": 4.5,  # 平均长度(米)
            "自行车": 1.8,
            "椅子": 0.8,
            "桌子": 1.2
        }
        
        # 假设的焦距和传感器尺寸（需要根据实际相机调整）
        focal_length = 800  # 像素
        real_size = object_sizes.get(class_name, 1.0)  # 默认1米
        
        # 简单的距离估算公式
        if max(width, height) > 0:
            distance = (real_size * focal_length) / max(width, height)
            return max(0.5, min(distance, 50))  # 限制在0.5-50米范围内
        
        return 5.0  # 默认距离
    
    def _get_danger_level(self, class_name: str) -> str:
        """获取物体的危险等级"""
        for level, classes in self.danger_levels.items():
            if class_name in classes:
                return level
        return "未知"
    
    def _danger_priority(self, danger_level: str) -> int:
        """危险等级优先级（用于排序）"""
        priority_map = {"高危险": 1, "中危险": 2, "低危险": 3, "可忽略": 4, "未知": 5}
        return priority_map.get(danger_level, 5)
    
    def annotate_image(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """在图像上标注检测结果"""
        annotated_image = image.copy()
        
        # 危险等级颜色映射
        color_map = {
            "高危险": (0, 0, 255),    # 红色
            "中危险": (0, 165, 255),  # 橙色
            "低危险": (0, 255, 255),  # 黄色
            "可忽略": (0, 255, 0),    # 绿色
            "未知": (128, 128, 128)    # 灰色
        }
        
        for detection in detections:
            bbox = detection["bbox"]
            color = color_map.get(detection["danger_level"], (128, 128, 128))
            
            # 绘制边界框
            cv2.rectangle(
                annotated_image,
                (bbox["x1"], bbox["y1"]),
                (bbox["x2"], bbox["y2"]),
                color, 2
            )
            
            # 准备标签文本
            label = f"{detection['class_name']} ({detection['confidence']:.2f})"
            distance_text = f"{detection['estimated_distance']:.1f}m"
            
            # 绘制标签背景
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(
                annotated_image,
                (bbox["x1"], bbox["y1"] - label_size[1] - 10),
                (bbox["x1"] + label_size[0], bbox["y1"]),
                color, -1
            )
            
            # 绘制标签文本
            cv2.putText(
                annotated_image, label,
                (bbox["x1"], bbox["y1"] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            
            # 绘制距离信息
            cv2.putText(
                annotated_image, distance_text,
                (bbox["x1"], bbox["y2"] + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
        
        return annotated_image

# 全局检测器实例
detector = YOLODetector(
    model_path=os.getenv('YOLO_MODEL_PATH', 'yolov8n.pt'),
    confidence_threshold=float(os.getenv('YOLO_CONFIDENCE_THRESHOLD', '0.5'))
)

# Redis连接
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', '6379')),
    decode_responses=True
)

@app.on_event("startup")
async def startup_event():
    """启动时初始化模型"""
    print("正在初始化YOLO检测服务...")
    if detector.initialize():
        print("YOLO检测服务初始化成功")
    else:
        print("警告: YOLO检测服务初始化失败")

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy" if detector.is_initialized else "unhealthy",
        "model_path": detector.model_path,
        "confidence_threshold": detector.confidence_threshold,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/detect/current")
async def detect_current_frame():
    """检测当前视频帧中的物体"""
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
                
                # 运行检测
                detections = detector.detect_objects(image)
                
                # 缓存结果到Redis
                result = {
                    "detections": detections,
                    "frame_info": {
                        "width": frame_data["width"],
                        "height": frame_data["height"],
                        "timestamp": frame_data["timestamp"]
                    },
                    "detection_timestamp": datetime.now().isoformat(),
                    "total_objects": len(detections)
                }
                
                redis_client.setex("current_detections", 30, json.dumps(result))
                
                return result
            else:
                raise HTTPException(status_code=503, detail="无法获取视频帧")
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")

@app.post("/detect/image")
async def detect_uploaded_image(file: UploadFile = File(...)):
    """检测上传图像中的物体"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="只支持图像文件")
    
    try:
        # 读取上传的图像
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="无法解析图像文件")
        
        # 运行检测
        detections = detector.detect_objects(image)
        
        # 生成标注图像
        annotated_image = detector.annotate_image(image, detections)
        
        # 编码标注图像为base64
        ret, buffer = cv2.imencode('.jpg', annotated_image)
        if ret:
            annotated_base64 = base64.b64encode(buffer).decode('utf-8')
        else:
            annotated_base64 = None
        
        return {
            "detections": detections,
            "total_objects": len(detections),
            "annotated_image": annotated_base64,
            "image_info": {
                "width": image.shape[1],
                "height": image.shape[0]
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图像检测失败: {str(e)}")

@app.get("/analyze")
async def analyze_current_scene():
    """分析当前场景并返回导航相关信息"""
    try:
        # 获取当前检测结果
        detection_data = redis_client.get("current_detections")
        if not detection_data:
            # 如果缓存中没有，执行新的检测
            current_result = await detect_current_frame()
            detections = current_result["detections"]
        else:
            current_result = json.loads(detection_data)
            detections = current_result["detections"]
        
        # 分析场景
        analysis = {
            "scene_safety": "安全",
            "immediate_threats": [],
            "path_obstacles": [],
            "recommended_action": "前进",
            "confidence_level": "高"
        }
        
        high_danger_objects = [d for d in detections if d["danger_level"] == "高危险"]
        medium_danger_objects = [d for d in detections if d["danger_level"] == "中危险"]
        
        # 判断即时威胁（距离小于3米的高危险物体）
        immediate_threats = [d for d in high_danger_objects if d["estimated_distance"] < 3.0]
        
        if immediate_threats:
            analysis["scene_safety"] = "危险"
            analysis["immediate_threats"] = immediate_threats
            analysis["recommended_action"] = "停止"
            analysis["confidence_level"] = "高"
        elif high_danger_objects:
            # 有高危险物体但距离较远
            closest_danger = min(high_danger_objects, key=lambda x: x["estimated_distance"])
            if closest_danger["estimated_distance"] < 5.0:
                analysis["scene_safety"] = "警告"
                analysis["recommended_action"] = "减速"
                analysis["confidence_level"] = "中"
        elif medium_danger_objects:
            # 只有中等危险物体
            closest_medium = min(medium_danger_objects, key=lambda x: x["estimated_distance"])
            if closest_medium["estimated_distance"] < 2.0:
                analysis["scene_safety"] = "注意"
                analysis["recommended_action"] = "谨慎前进"
        
        # 识别路径障碍物（前方中央区域的物体）
        image_center_x = current_result["frame_info"]["width"] // 2
        path_obstacles = []
        
        for detection in detections:
            bbox = detection["bbox"]
            obj_center_x = bbox["center_x"]
            
            # 判断物体是否在前方路径上（图像中央1/3区域）
            if abs(obj_center_x - image_center_x) < image_center_x // 3:
                if detection["estimated_distance"] < 10.0:  # 10米内
                    path_obstacles.append(detection)
        
        analysis["path_obstacles"] = path_obstacles
        
        # 综合分析结果
        result = {
            **current_result,
            "scene_analysis": analysis,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        # 缓存分析结果
        redis_client.setex("scene_analysis", 15, json.dumps(result))
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"场景分析失败: {str(e)}")

@app.get("/statistics")
async def get_detection_statistics():
    """获取检测统计信息"""
    try:
        # 从Redis获取历史数据进行统计
        stats = {
            "total_detections_today": 0,
            "most_common_objects": {},
            "danger_level_distribution": {
                "高危险": 0, "中危险": 0, "低危险": 0, "可忽略": 0
            },
            "average_confidence": 0.0,
            "timestamp": datetime.now().isoformat()
        }
        
        # 这里可以添加更复杂的统计逻辑
        # 从数据库或Redis中获取历史检测数据
        
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")

@app.post("/settings/confidence")
async def update_confidence_threshold(threshold: float):
    """更新置信度阈值"""
    if not 0.1 <= threshold <= 1.0:
        raise HTTPException(status_code=400, detail="置信度阈值必须在0.1-1.0之间")
    
    detector.confidence_threshold = threshold
    return {
        "status": "success",
        "new_threshold": threshold,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/classes")
async def get_supported_classes():
    """获取支持的检测类别"""
    return {
        "navigation_classes": detector.navigation_classes,
        "danger_levels": detector.danger_levels,
        "total_classes": len(detector.navigation_classes)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.getenv("YOLO_DETECTION_HOST", "localhost"),
        port=int(os.getenv("YOLO_DETECTION_PORT", "8002"))
    )
