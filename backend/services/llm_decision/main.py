"""
LLM决策服务
基于YOLO检测结果生成AI导航指令
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import openai
import os
import json
from datetime import datetime
import asyncio
import redis
from enum import Enum

app = FastAPI(title="LLM Decision Service", version="1.0.0")

# 数据模型
class DangerLevel(str, Enum):
    HIGH = "高危险"
    MEDIUM = "中危险"
    LOW = "低危险"
    IGNORE = "可忽略"

class NavigationAction(str, Enum):
    STOP = "停止"
    SLOW_DOWN = "减速"
    TURN_LEFT = "左转"
    TURN_RIGHT = "右转"
    MOVE_FORWARD = "前进"
    MOVE_BACKWARD = "后退"
    WAIT = "等待"
    CAREFUL_FORWARD = "谨慎前进"

class ObstacleDetection(BaseModel):
    class_name: str
    confidence: float
    bbox: Dict[str, int]
    estimated_distance: float
    danger_level: str
    timestamp: str

class NavigationCommand(BaseModel):
    action: str
    parameters: Dict[str, Any]
    reason: str
    priority: int
    confidence: float
    timestamp: str

class LLMDecisionMaker:
    def __init__(self, api_key: str, model: str = "deepseek-chat", base_url: str = None):
        if api_key:
            # 如果提供了base_url，使用DeepSeek API
            if base_url or "deepseek" in model.lower():
                self.client = openai.OpenAI(
                    api_key=api_key,
                    base_url=base_url or "https://api.deepseek.com"
                )
            else:
                # 默认使用OpenAI
                self.client = openai.OpenAI(api_key=api_key)
        else:
            self.client = None
        self.model = model
        self.decision_history = []
=======
        
        # 决策规则模板
        self.system_prompt = """
你是一个智能导航系统的决策引擎。根据提供的物体检测信息，你需要生成安全的导航指令。

决策原则：
1. 安全第一：遇到高危险物体时必须停止或避让
2. 效率其次：在安全的前提下选择最高效的路径
3. 保持距离：与移动物体保持足够安全距离
4. 预判风险：考虑潜在的危险情况

危险等级定义：
- 高危险：汽车、卡车、公交车、摩托车等移动车辆
- 中危险：人员、自行车、交通信号等
- 低危险：固定障碍物如椅子、桌子等
- 可忽略：小物件如杯子、遥控器等

距离判断：
- 0-2米：即时危险区域
- 2-5米：警告区域  
- 5-10米：注意区域
- 10米以上：安全区域

请根据检测到的物体信息，生成JSON格式的导航指令，包含：
- action: 具体动作
- parameters: 动作参数（如速度、角度等）
- reason: 决策理由
- priority: 优先级(1-10)
- confidence: 决策置信度(0-1)
"""
    
    def _create_rule_based_decision(self, obstacles: List[ObstacleDetection]) -> NavigationCommand:
        """基于规则的决策（备用方案）"""
        if not obstacles:
            return NavigationCommand(
                action=NavigationAction.MOVE_FORWARD,
                parameters={"speed": 0.5},
                reason="未检测到障碍物，正常前进",
                priority=1,
                confidence=0.9,
                timestamp=datetime.now().isoformat()
            )
        
        # 分析即时威胁
        immediate_threats = [obs for obs in obstacles 
                           if obs.estimated_distance < 2.0 and obs.danger_level == DangerLevel.HIGH]
        
        if immediate_threats:
            return NavigationCommand(
                action=NavigationAction.STOP,
                parameters={"emergency": True},
                reason=f"检测到{len(immediate_threats)}个即时威胁：{', '.join([obs.class_name for obs in immediate_threats])}",
                priority=10,
                confidence=0.95,
                timestamp=datetime.now().isoformat()
            )
        
        # 分析警告区域威胁
        warning_threats = [obs for obs in obstacles 
                         if 2.0 <= obs.estimated_distance < 5.0 and obs.danger_level in [DangerLevel.HIGH, DangerLevel.MEDIUM]]
        
        if warning_threats:
            # 检查威胁位置，决定是减速还是转向
            center_threats = [obs for obs in warning_threats 
                            if abs(obs.bbox["center_x"] - 640) < 200]  # 假设图像宽度1280
            
            if center_threats:
                return NavigationCommand(
                    action=NavigationAction.SLOW_DOWN,
                    parameters={"speed": 0.3, "reason": "前方有障碍"},
                    reason=f"前方{center_threats[0].estimated_distance:.1f}米处有{center_threats[0].class_name}",
                    priority=7,
                    confidence=0.8,
                    timestamp=datetime.now().isoformat()
                )
        
        # 分析注意区域
        attention_obstacles = [obs for obs in obstacles 
                             if 5.0 <= obs.estimated_distance < 10.0]
        
        if attention_obstacles:
            return NavigationCommand(
                action=NavigationAction.CAREFUL_FORWARD,
                parameters={"speed": 0.4, "monitoring": True},
                reason=f"注意区域有{len(attention_obstacles)}个物体，谨慎前进",
                priority=3,
                confidence=0.7,
                timestamp=datetime.now().isoformat()
            )
        
        # 默认情况
        return NavigationCommand(
            action=NavigationAction.MOVE_FORWARD,
            parameters={"speed": 0.5},
            reason="路径相对安全，正常前进",
            priority=1,
            confidence=0.8,
            timestamp=datetime.now().isoformat()
        )
    
    async def make_decision(self, obstacles: List[ObstacleDetection]) -> NavigationCommand:
        """生成导航决策"""
        try:
            if self.client:
                return await self._llm_decision(obstacles)
            else:
                return self._create_rule_based_decision(obstacles)
        except Exception as e:
            print(f"LLM决策失败，使用规则决策: {e}")
            return self._create_rule_based_decision(obstacles)
    
    async def _llm_decision(self, obstacles: List[ObstacleDetection]) -> NavigationCommand:
        """使用LLM生成决策"""
        # 构建输入数据
        obstacles_data = []
        for obs in obstacles:
            obstacles_data.append({
                "物体类型": obs.class_name,
                "置信度": obs.confidence,
                "距离": f"{obs.estimated_distance:.1f}米",
                "危险等级": obs.danger_level,
                "位置": f"({obs.bbox['center_x']}, {obs.bbox['center_y']})",
                "大小": f"{obs.bbox['width']}x{obs.bbox['height']}"
            })
        
        user_message = f"""
当前检测到的障碍物信息：
{json.dumps(obstacles_data, ensure_ascii=False, indent=2)}

图像信息：
- 分辨率：1280x720（假设）
- 中心点：(640, 360)
- 当前速度：0.5m/s

请分析当前情况并生成导航指令。返回JSON格式：
{{
    "action": "动作名称",
    "parameters": {{"speed": 速度值, "其他参数": "值"}},
    "reason": "决策理由",
    "priority": 优先级数字,
    "confidence": 置信度数值
}}
"""
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            content = response.choices[0].message.content.strip()
            
            # 尝试解析JSON响应
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
            
            decision_data = json.loads(content)
            
            return NavigationCommand(
                action=decision_data.get("action", NavigationAction.WAIT),
                parameters=decision_data.get("parameters", {}),
                reason=decision_data.get("reason", "LLM生成的决策"),
                priority=decision_data.get("priority", 5),
                confidence=decision_data.get("confidence", 0.7),
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            print(f"LLM API调用失败: {e}")
            return self._create_rule_based_decision(obstacles)
    
    def get_decision_history(self, limit: int = 10) -> List[NavigationCommand]:
        """获取决策历史"""
        return self.decision_history[-limit:]
    
    def add_to_history(self, decision: NavigationCommand):
        """添加决策到历史"""
        self.decision_history.append(decision)
        # 保持历史记录在合理范围内
        if len(self.decision_history) > 100:
            self.decision_history = self.decision_history[-50:]

# 全局决策器实例
decision_maker = LLMDecisionMaker(
    api_key=os.getenv('DEEPSEEK_API_KEY') or os.getenv('OPENAI_API_KEY'),
    model=os.getenv('LLM_MODEL', 'deepseek-chat'),
    base_url=os.getenv('LLM_BASE_URL')
)
=======

# Redis连接
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', '6379')),
    decode_responses=True
)

@app.get("/health")
async def health_check():
    """健康检查"""
    llm_available = decision_maker.client is not None
    
    return {
        "status": "healthy",
        "llm_available": llm_available,
        "model": decision_maker.model,
        "decision_history_count": len(decision_maker.decision_history),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/generate_command")
async def generate_navigation_command(obstacles: List[ObstacleDetection]):
    """基于障碍物信息生成导航指令"""
    try:
        # 生成决策
        decision = await decision_maker.make_decision(obstacles)
        
        # 添加到历史
        decision_maker.add_to_history(decision)
        
        # 缓存到Redis
        redis_client.setex("latest_decision", 60, decision.json())
        
        return decision.dict()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成导航指令失败: {str(e)}")

@app.post("/analyze_scene")
async def analyze_scene_safety(obstacles: List[ObstacleDetection]):
    """分析场景安全性"""
    try:
        # 分类统计
        danger_stats = {
            "高危险": 0, "中危险": 0, "低危险": 0, "可忽略": 0
        }
        
        distance_stats = {
            "immediate": [],  # 0-2米
            "warning": [],    # 2-5米
            "attention": [],  # 5-10米
            "safe": []        # 10米以上
        }
        
        for obs in obstacles:
            danger_stats[obs.danger_level] = danger_stats.get(obs.danger_level, 0) + 1
            
            if obs.estimated_distance < 2.0:
                distance_stats["immediate"].append(obs)
            elif obs.estimated_distance < 5.0:
                distance_stats["warning"].append(obs)
            elif obs.estimated_distance < 10.0:
                distance_stats["attention"].append(obs)
            else:
                distance_stats["safe"].append(obs)
        
        # 计算安全等级
        safety_score = 100
        safety_level = "安全"
        
        # 扣分规则
        safety_score -= len(distance_stats["immediate"]) * 50
        safety_score -= len(distance_stats["warning"]) * 20
        safety_score -= len(distance_stats["attention"]) * 5
        safety_score -= danger_stats["高危险"] * 30
        safety_score -= danger_stats["中危险"] * 15
        
        safety_score = max(0, safety_score)
        
        if safety_score < 30:
            safety_level = "危险"
        elif safety_score < 60:
            safety_level = "警告"
        elif safety_score < 80:
            safety_level = "注意"
        
        analysis = {
            "safety_level": safety_level,
            "safety_score": safety_score,
            "danger_statistics": danger_stats,
            "distance_analysis": {
                "immediate_threats": len(distance_stats["immediate"]),
                "warning_objects": len(distance_stats["warning"]),
                "attention_objects": len(distance_stats["attention"]),
                "safe_objects": len(distance_stats["safe"])
            },
            "recommendations": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # 生成建议
        if distance_stats["immediate"]:
            analysis["recommendations"].append("立即停止，存在即时威胁")
        elif distance_stats["warning"]:
            analysis["recommendations"].append("减速慢行，注意前方障碍")
        elif danger_stats["高危险"] > 0:
            analysis["recommendations"].append("保持警惕，附近有高危险物体")
        else:
            analysis["recommendations"].append("可以正常行进")
        
        return analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"场景分析失败: {str(e)}")

@app.get("/decision/history")
async def get_decision_history(limit: int = 10):
    """获取决策历史"""
    history = decision_maker.get_decision_history(limit)
    return {
        "history": [decision.dict() for decision in history],
        "total_count": len(decision_maker.decision_history),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/decision/latest")
async def get_latest_decision():
    """获取最新决策"""
    try:
        latest_data = redis_client.get("latest_decision")
        if latest_data:
            return json.loads(latest_data)
        
        if decision_maker.decision_history:
            return decision_maker.decision_history[-1].dict()
        
        return {
            "action": "wait",
            "parameters": {},
            "reason": "暂无决策历史",
            "priority": 1,
            "confidence": 0.5,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取最新决策失败: {str(e)}")

@app.post("/decision/manual_override")
async def manual_override(command: NavigationCommand):
    """手动覆盖决策"""
    try:
        # 设置为手动决策
        command.reason = f"手动干预: {command.reason}"
        command.priority = 10  # 最高优先级
        command.timestamp = datetime.now().isoformat()
        
        # 添加到历史
        decision_maker.add_to_history(command)
        
        # 缓存到Redis
        redis_client.setex("latest_decision", 60, command.json())
        redis_client.setex("manual_override", 300, "true")  # 5分钟标记
        
        return {
            "status": "success",
            "message": "手动决策已生效",
            "command": command.dict(),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"手动覆盖失败: {str(e)}")

@app.get("/decision/patterns")
async def get_decision_patterns():
    """分析决策模式"""
    try:
        if not decision_maker.decision_history:
            return {"message": "暂无决策历史"}
        
        # 统计动作分布
        action_counts = {}
        priority_distribution = {}
        confidence_levels = []
        
        for decision in decision_maker.decision_history:
            action = decision.action
            action_counts[action] = action_counts.get(action, 0) + 1
            
            priority = decision.priority
            priority_distribution[priority] = priority_distribution.get(priority, 0) + 1
            
            confidence_levels.append(decision.confidence)
        
        avg_confidence = sum(confidence_levels) / len(confidence_levels) if confidence_levels else 0
        
        return {
            "action_distribution": action_counts,
            "priority_distribution": priority_distribution,
            "average_confidence": round(avg_confidence, 3),
            "total_decisions": len(decision_maker.decision_history),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析决策模式失败: {str(e)}")

@app.post("/settings/model")
async def update_llm_model(model_name: str):
    """更新LLM模型"""
    try:
        decision_maker.model = model_name
        return {
            "status": "success",
            "new_model": model_name,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新模型失败: {str(e)}")

@app.get("/actions/available")
async def get_available_actions():
    """获取可用的导航动作"""
    return {
        "actions": [action.value for action in NavigationAction],
        "danger_levels": [level.value for level in DangerLevel],
        "priority_range": "1-10 (10为最高优先级)",
        "confidence_range": "0.0-1.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.getenv("LLM_DECISION_HOST", "localhost"),
        port=int(os.getenv("LLM_DECISION_PORT", "8004"))
    )
