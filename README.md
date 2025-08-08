# AI导航系统 - 实时环境感知与智能导航

基于YOLOv8和LLM的智能导航系统，提供实时物体检测、透视校正、AI决策和可视化交互界面。

## 🌟 主要特性

### 🎯 核心功能
- **实时环境感知**: 使用YOLOv8进行物体检测和分类
- **透视校正**: 建立网格坐标系统，实现像素-米转换
- **AI智能决策**: 集成LLM模型生成导航指令
- **可视化界面**: 实时视频监控、障碍物列表、AI指令中心
- **手动干预**: 支持用户手动控制和紧急停止
- **语音交互**: 语音命令控制和指令播报

### 🏗️ 系统架构
```
前端层 (Frontend)
├── 视频监控面板
├── 障碍物检测列表  
├── AI指令中心
└── 系统状态面板

后端服务层 (Backend Services)
├── API网关服务 (统一入口)
├── 视频采集服务 (摄像头流)
├── YOLO检测服务 (物体识别)
├── 透视校正服务 (坐标转换)
└── LLM决策服务 (智能决策)

数据存储层 (Data Layer)  
├── Redis (实时缓存)
├── PostgreSQL (结构化数据)
└── MinIO (文件存储)
```

## 🚀 快速开始

### 系统要求
- Python 3.8+
- 摄像头设备 (USB/IP摄像头)
- 8GB+ 内存
- NVIDIA GPU (可选，用于YOLO加速)

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd YoloTry2
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **配置环境**
```bash
cp .env.example .env
# 编辑 .env 文件，配置API密钥等
```

4. **启动系统**
```bash
python start_system.py
```

5. **访问界面**
- 打开浏览器访问: http://localhost:3000
- API文档: http://localhost:8000/docs

### Docker部署

1. **使用Docker Compose启动**
```bash
docker-compose up -d
```

2. **查看服务状态**
```bash
docker-compose ps
```

3. **查看日志**
```bash
docker-compose logs -f
```

## 📱 使用指南

### 前端界面说明

#### 1. 视频监控面板
- **原始画面**: 显示摄像头实时视频流
- **网格视图**: 透视校正后的网格坐标视图
- **检测结果**: 标注了检测物体的视频流

#### 2. AI指令中心
- 显示最新的AI导航指令
- 支持手动控制按钮
- 紧急停止功能
- 语音控制支持

#### 3. 障碍物列表
- 实时显示检测到的物体
- 按危险等级分类显示
- 包含距离和置信度信息

#### 4. 系统状态面板
- 服务健康状态
- 连接状态监控
- 性能指标显示

### API接口说明

#### 视频相关
```bash
GET /api/video/stream          # 获取视频流
GET /api/video/grid_view       # 获取网格视图
```

#### 检测相关
```bash
GET /api/obstacles/current     # 获取当前障碍物
POST /api/obstacles/analyze    # 分析当前帧
```

#### 导航相关
```bash
GET /api/navigation/current_command    # 获取当前指令
POST /api/navigation/manual_command    # 发送手动指令
```

#### 透视校正
```bash
POST /api/calibration/set_grid_points  # 设置校准点
GET /api/calibration/get_matrix        # 获取变换矩阵
```

## 🔧 配置说明

### 环境变量配置

主要配置项说明：

```bash
# 摄像头配置
CAMERA_URL=0                    # 摄像头设备号或RTSP地址

# YOLO模型配置
YOLO_MODEL_PATH=./models/yolov8n.pt
YOLO_CONFIDENCE_THRESHOLD=0.5

# LLM配置
OPENAI_API_KEY=your_api_key     # OpenAI API密钥
LLM_MODEL=gpt-3.5-turbo

# 数据库配置
REDIS_HOST=localhost
POSTGRES_HOST=localhost
```

### 透视校正校准

1. **设置校准点**
   - 在实际环境中标记4个已知坐标的点
   - 通过API设置像素坐标和真实世界坐标的对应关系

2. **校准示例**
```python
calibration_data = {
    "image_points": [
        [100, 400],  # 左下角像素坐标
        [540, 400],  # 右下角像素坐标  
        [540, 200],  # 右上角像素坐标
        [100, 200]   # 左上角像素坐标
    ],
    "real_world_points": [
        [0, 0],      # 左下角实际坐标(米)
        [4, 0],      # 右下角实际坐标(米)
        [4, 3],      # 右上角实际坐标(米)
        [0, 3]       # 左上角实际坐标(米)
    ]
}
```

## 🎛️ 高级功能

### 自定义物体检测

1. **训练自定义模型**
```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8n.pt')

# 使用自定义数据集训练
model.train(data='path/to/dataset.yaml', epochs=100)
```

2. **更新检测类别**
```python
# 在 yolo_detection/main.py 中修改
navigation_classes = {
    0: "自定义物体1",
    1: "自定义物体2",
    # ...
}
```

### 扩展LLM决策逻辑

1. **自定义决策规则**
```python
def custom_decision_logic(obstacles):
    # 实现自定义决策逻辑
    if condition:
        return NavigationCommand(
            action="custom_action",
            parameters={"speed": 0.5},
            reason="自定义理由"
        )
```

2. **集成其他LLM模型**
```python
# 支持本地模型、其他API等
from transformers import pipeline

# 使用Hugging Face模型
llm = pipeline("text-generation", model="model_name")
```

## 📊 监控与日志

### 系统监控
- Prometheus + Grafana 监控仪表板
- 实时性能指标
- 服务健康状态

### 日志系统
- 结构化日志记录
- 多级别日志（INFO, WARNING, ERROR）
- 日志轮转和归档

## 🔒 安全性

### 访问控制
- API网关认证
- 服务间通信加密
- 前端访问控制

### 数据安全
- 敏感数据加密存储
- 定期备份
- 访问审计日志

## 🐛 故障排除

### 常见问题

1. **摄像头无法访问**
```bash
# 检查摄像头权限
ls /dev/video*
# 确认摄像头未被其他程序占用
```

2. **YOLO模型加载失败**
```bash
# 下载YOLOv8模型
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

3. **Redis连接失败**
```bash
# 启动Redis服务
redis-server
# 检查连接
redis-cli ping
```

4. **端口被占用**
```bash
# 查看端口占用
netstat -tulpn | grep :8000
# 修改端口配置或终止占用进程
```

### 调试模式

启用详细日志：
```bash
export LOG_LEVEL=DEBUG
python start_system.py
```

## 🤝 贡献指南

### 开发环境设置
1. Fork项目仓库
2. 创建开发分支
3. 安装开发依赖
```bash
pip install -r requirements-dev.txt
```

### 代码规范
- 使用Black进行代码格式化
- 遵循PEP 8规范
- 添加类型注解
- 编写单元测试

### 提交流程
1. 创建功能分支
2. 编写测试用例
3. 确保所有测试通过
4. 提交Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 支持与反馈

- **问题反馈**: [GitHub Issues](https://github.com/your-repo/issues)
- **功能建议**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **技术交流**: 加入我们的社区群组

## 🔮 后续计划

### 短期目标
- [ ] 增加更多物体检测类别
- [ ] 优化检测精度和速度
- [ ] 完善Web界面交互
- [ ] 添加移动端支持

### 长期规划
- [ ] 支持多摄像头融合
- [ ] 3D环境建模
- [ ] 强化学习优化决策
- [ ] 边缘设备部署

---

## 🙏 致谢

感谢以下开源项目的支持：
- [YOLOv8](https://github.com/ultralytics/ultralytics) - 物体检测模型
- [FastAPI](https://fastapi.tiangolo.com/) - Web框架
- [OpenCV](https://opencv.org/) - 计算机视觉库
- [Redis](https://redis.io/) - 内存数据库

---

**如果这个项目对您有帮助，请给我们一个⭐星标支持！**
