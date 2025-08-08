"""
API网关服务 - 统一入口
负责路由分发、负载均衡、认证授权等
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import httpx
import asyncio
import json
import redis
from typing import List, Dict, Any
import os
from pydantic import BaseModel
import websockets
from datetime import datetime

app = FastAPI(title="AI Navigation API Gateway", version="1.0.0")

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 服务配置
SERVICES = {
    "video_capture": f"http://{os.getenv('VIDEO_CAPTURE_HOST', 'localhost')}:{os.getenv('VIDEO_CAPTURE_PORT', '8001')}",
    "yolo_detection": f"http://{os.getenv('YOLO_DETECTION_HOST', 'localhost')}:{os.getenv('YOLO_DETECTION_PORT', '8002')}",
    "perspective_correction": f"http://{os.getenv('PERSPECTIVE_CORRECTION_HOST', 'localhost')}:{os.getenv('PERSPECTIVE_CORRECTION_PORT', '8003')}",
    "llm_decision": f"http://{os.getenv('LLM_DECISION_HOST', 'localhost')}:{os.getenv('LLM_DECISION_PORT', '8004')}"
}

# Redis连接
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', '6379')),
    decode_responses=True
)

# WebSocket连接管理
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # 移除断开的连接
                self.active_connections.remove(connection)

manager = ConnectionManager()

# 数据模型
class NavigationCommand(BaseModel):
    action: str
    parameters: Dict[str, Any]
    reason: str
    priority: int = 1
    timestamp: str = None

class ObstacleInfo(BaseModel):
    type: str
    coordinates: Dict[str, float]
    confidence: float
    distance: float

# 健康检查
@app.get("/health")
async def health_check():
    """系统健康检查"""
    services_status = {}
    
    async with httpx.AsyncClient() as client:
        for service_name, service_url in SERVICES.items():
            try:
                response = await client.get(f"{service_url}/health", timeout=5.0)
                services_status[service_name] = {
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "response_time": response.elapsed.total_seconds()
                }
            except Exception as e:
                services_status[service_name] = {
                    "status": "unreachable",
                    "error": str(e)
                }
    
    # Redis健康检查
    try:
        redis_client.ping()
        services_status["redis"] = {"status": "healthy"}
    except Exception as e:
        services_status["redis"] = {"status": "unhealthy", "error": str(e)}
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": services_status
    }

# 视频流相关API
@app.get("/api/video/stream")
async def get_video_stream():
    """获取实时视频流"""
    async def video_generator():
        async with httpx.AsyncClient() as client:
            async with client.stream("GET", f"{SERVICES['video_capture']}/stream") as response:
                async for chunk in response.aiter_bytes():
                    yield chunk
    
    return StreamingResponse(video_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/video/grid_view")
async def get_grid_view():
    """获取透视校正后的网格视图"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{SERVICES['perspective_correction']}/grid_view")
        return response.json()

# YOLO检测相关API
@app.get("/api/obstacles/current")
async def get_current_obstacles():
    """获取当前检测到的障碍物"""
    try:
        # 从Redis缓存获取最新的检测结果
        obstacles_data = redis_client.get("current_obstacles")
        if obstacles_data:
            return json.loads(obstacles_data)
        
        # 如果缓存中没有，直接调用YOLO服务
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{SERVICES['yolo_detection']}/detect/current")
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取障碍物信息失败: {str(e)}")

@app.post("/api/obstacles/analyze")
async def analyze_frame():
    """分析当前帧并返回检测结果"""
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{SERVICES['yolo_detection']}/analyze")
        if response.status_code == 200:
            result = response.json()
            # 缓存结果到Redis
            redis_client.setex("current_obstacles", 30, json.dumps(result))
            
            # 广播检测结果给所有WebSocket连接
            await manager.broadcast(json.dumps({
                "type": "obstacles_update",
                "data": result
            }))
            
            return result
        else:
            raise HTTPException(status_code=response.status_code, detail="分析失败")

# LLM决策相关API
@app.get("/api/navigation/current_command")
async def get_current_command():
    """获取当前导航指令"""
    try:
        command_data = redis_client.get("current_command")
        if command_data:
            return json.loads(command_data)
        return {"action": "wait", "parameters": {}, "reason": "等待分析结果"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取导航指令失败: {str(e)}")

@app.post("/api/navigation/generate_command")
async def generate_navigation_command(obstacles: List[ObstacleInfo]):
    """基于障碍物信息生成导航指令"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{SERVICES['llm_decision']}/generate_command",
            json=[obstacle.dict() for obstacle in obstacles]
        )
        
        if response.status_code == 200:
            command = response.json()
            command["timestamp"] = datetime.now().isoformat()
            
            # 缓存指令到Redis
            redis_client.setex("current_command", 300, json.dumps(command))
            
            # 广播指令给所有WebSocket连接
            await manager.broadcast(json.dumps({
                "type": "navigation_command",
                "data": command
            }))
            
            return command
        else:
            raise HTTPException(status_code=response.status_code, detail="生成导航指令失败")

@app.post("/api/navigation/manual_command")
async def manual_navigation_command(command: NavigationCommand):
    """手动导航指令（用户干预）"""
    command.timestamp = datetime.now().isoformat()
    command.priority = 10  # 手动指令优先级最高
    
    # 缓存到Redis
    redis_client.setex("current_command", 300, json.dumps(command.dict()))
    
    # 广播指令
    await manager.broadcast(json.dumps({
        "type": "manual_command",
        "data": command.dict()
    }))
    
    return {"status": "success", "message": "手动指令已发送"}

# 透视校正相关API
@app.post("/api/calibration/set_grid_points")
async def set_grid_points(points: List[Dict[str, float]]):
    """设置网格校正点"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{SERVICES['perspective_correction']}/calibration/set_points",
            json=points
        )
        return response.json()

@app.get("/api/calibration/get_matrix")
async def get_transformation_matrix():
    """获取透视变换矩阵"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{SERVICES['perspective_correction']}/calibration/matrix")
        return response.json()

# WebSocket端点
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket连接端点，用于实时数据推送"""
    await manager.connect(websocket)
    try:
        while True:
            # 保持连接活跃
            data = await websocket.receive_text()
            # 可以处理客户端发送的消息
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# 系统状态API
@app.get("/api/system/status")
async def get_system_status():
    """获取系统整体状态"""
    # 获取各服务状态
    health_response = await health_check()
    
    # 获取当前活跃连接数
    active_connections = len(manager.active_connections)
    
    # 获取Redis中的统计信息
    try:
        redis_info = redis_client.info()
        redis_stats = {
            "connected_clients": redis_info.get("connected_clients", 0),
            "used_memory_human": redis_info.get("used_memory_human", "0B"),
            "total_commands_processed": redis_info.get("total_commands_processed", 0)
        }
    except:
        redis_stats = {"error": "无法获取Redis统计信息"}
    
    return {
        "timestamp": datetime.now().isoformat(),
        "services_health": health_response["services"],
        "websocket_connections": active_connections,
        "redis_stats": redis_stats,
        "system_load": "正常"  # 可以添加更多系统监控指标
    }

# 启动后台任务
@app.on_event("startup")
async def startup_event():
    """启动时执行的任务"""
    print("API网关服务启动中...")
    print(f"已配置的服务: {list(SERVICES.keys())}")

@app.on_event("shutdown")
async def shutdown_event():
    """关闭时执行的清理任务"""
    print("API网关服务正在关闭...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.getenv("API_GATEWAY_HOST", "localhost"),
        port=int(os.getenv("API_GATEWAY_PORT", "8000"))
    )
