#!/usr/bin/env python3
"""
AI导航系统启动脚本
负责启动所有微服务和前端界面
"""

import os
import sys
import subprocess
import time
import signal
import threading
import webbrowser
from pathlib import Path
import psutil
import argparse
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class SystemManager:
    def __init__(self):
        self.processes = {}
        self.base_dir = Path(__file__).parent
        self.running = True
        self.log_dir = self.base_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # 配置日志系统
        self.setup_logging()
        
        # 服务配置
        self.services = {
            "api_gateway": {
                "path": "backend/services/api_gateway/main.py",
                "port": int(os.getenv("API_GATEWAY_PORT", "8000")),
                "name": "API网关",
                "required": True,  # 必需服务
                "dependencies": []  # 无依赖
            },
            "video_capture": {
                "path": "backend/services/video_capture/main.py",
                "port": int(os.getenv("VIDEO_CAPTURE_PORT", "8001")),
                "name": "视频采集服务",
                "required": True,
                "dependencies": ["api_gateway"]
            },
            "yolo_detection": {
                "path": "backend/services/yolo_detection/main.py",
                "port": int(os.getenv("YOLO_DETECTION_PORT", "8002")),
                "name": "YOLO检测服务",
                "required": True,
                "dependencies": ["video_capture"]
            },
            "perspective_correction": {
                "path": "backend/services/perspective_correction/main.py",
                "port": int(os.getenv("PERSPECTIVE_CORRECTION_PORT", "8003")),
                "name": "透视校正服务",
                "required": False,  # 非必需服务
                "dependencies": ["video_capture"]
            },
            "llm_decision": {
                "path": "backend/services/llm_decision/main.py",
                "port": int(os.getenv("LLM_DECISION_PORT", "8004")),
                "name": "LLM决策服务",
                "required": False,
                "dependencies": ["yolo_detection"]
            }
        }
        
        # 系统状态
        self.system_status = {
            "start_time": None,
            "total_restarts": 0,
            "service_stats": {},
            "last_error": None
        }
    
    def setup_logging(self):
        """配置日志系统"""
        import logging
        from logging.handlers import RotatingFileHandler
        
        # 创建主日志文件
        main_log = self.log_dir / "system.log"
        main_handler = RotatingFileHandler(
            main_log,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        
        # 创建错误日志文件
        error_log = self.log_dir / "error.log"
        error_handler = RotatingFileHandler(
            error_log,
            maxBytes=10*1024*1024,
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        
        # 配置日志格式
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        main_handler.setFormatter(formatter)
        error_handler.setFormatter(formatter)
        
        # 配置根日志器
        self.logger = logging.getLogger('ai_navigation')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(main_handler)
        self.logger.addHandler(error_handler)
        
        # 添加控制台输出
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        self.logger.info("日志系统初始化完成")
    
    def log_error(self, error_msg, exc_info=None):
        """记录错误信息"""
        self.logger.error(error_msg, exc_info=exc_info)
        self.system_status["last_error"] = {
            "message": error_msg,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    class YourServiceClass:
        class YourSystemClass:  # 假设这是你的类名
            def __init__(self):
                # 初始化系统状态
                self.system_status = {
                    "service_stats": {}  # 用于记录各个服务的状态
                }

                # 定义所有服务的配置（注意变量名是 services_config）
                self.services_config = {
                    "video_capture": {
                        "path": "backend/services/video_capture/main.py",
                        "port": int(os.getenv("VIDEO_CAPTURE_PORT", "8001")),
                        "name": "视频采集服务"
                    },
                    "yolo_detection": {
                        "path": "backend/services/yolo_detection/main.py",
                        "port": int(os.getenv("YOLO_DETECTION_PORT", "8002")),
                        "name": "YOLO检测服务"
                    },
                    "perspective_correction": {
                        "path": "backend/services/perspective_correction/main.py",
                        "port": int(os.getenv("PERSPECTIVE_CORRECTION_PORT", "8003")),
                        "name": "透视校正服务"
                    },
                    "llm_decision": {
                        "path": "backend/services/llm_decision/main.py",
                        "port": int(os.getenv("LLM_DECISION_PORT", "8004")),
                        "name": "LLM决策服务"
                    }
                }

                # 前端配置
                self.frontend_port = int(os.getenv("FRONTEND_PORT", "3000"))

            def log_service_status(self, service_name, status, details=None):
                """记录服务状态"""
                self.logger.info(f"服务 {service_name}: {status}")
                # 只为单个服务记录状态，不包含所有服务的配置
                self.system_status["service_stats"][service_name] = {
                    "status": status,
                    "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "details": details
                }

            def check_dependencies(self):
                """检查系统依赖（所有检查逻辑都在函数内部）"""
                print("🔍 检查系统依赖...")

                # 1. 检查Python版本
                if sys.version_info < (3, 8):
                    print("❌ 需要Python 3.8或更高版本")
                    return False  # 正确：在函数内部返回

                # 2. 检查必要的Python包（缩进正确，在函数内部）
                required_packages = {
                    "fastapi": "fastapi",
                    "uvicorn": "uvicorn",
                    "opencv-python": "cv2",
                    "ultralytics": "ultralytics",
                    "redis": "redis",
                    "psycopg2-binary": "psycopg2",
                    "openai": "openai",
                    "numpy": "numpy",
                    "requests": "requests"
                }

                missing_packages = []
                for package_name, import_name in required_packages.items():
                    try:
                        __import__(import_name)
                        print(f"✅ {package_name}: 已安装")
                    except ImportError:
                        missing_packages.append(package_name)
                        print(f"❌ {package_name}: 未找到")

                if missing_packages:
                    print(f"\n❌ 缺少必要的Python包: {', '.join(missing_packages)}")
                    print("\n请按顺序执行以下命令:")
                    print("pip install --upgrade pip")
                    print("pip install -r requirements.txt")
                    print("\n如果遇到问题，尝试:")
                    print("pip install opencv-python-headless  # 替代opencv-python")
                    print("pip install psycopg2-binary --no-cache-dir")
                    return False  # 正确：在函数内部返回

                # 3. 检查端口占用（使用正确的变量名 services_config）
                for service_name, config in self.services_config.items():  # 修正：self.services → self.services_config
                    if self.is_port_in_use(config["port"]):
                        print(f"⚠️  端口 {config['port']} 被占用 ({config['name']})")
                        # 如果端口占用需要阻止启动，可在此处 return False

                print("✅ 依赖检查完成")
                return True  # 所有检查通过，返回True
    
    def is_port_in_use(self, port):
        """检查端口是否被占用"""
        try:
            # 使用更简单的方法检查端口
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                # 尝试绑定端口
                s.bind(('localhost', port))
                return False
        except socket.error:
            return True
        except Exception as e:
            print(f"⚠️  检查端口 {port} 时出错: {e}")
            return False  # 如果无法确定，假设端口可用
    
    def start_service(self, service_name, config):
        """启动单个服务"""
        service_path = self.base_dir / config["path"]
        
        if not service_path.exists():
            print(f"❌ 服务文件不存在: {service_path}")
            return False
        
        # 检查端口是否被占用
        if self.is_port_in_use(config["port"]):
            print(f"⚠️  端口 {config['port']} 已被占用，尝试终止现有进程...")
            try:
                # 在Windows上可以使用: netstat -ano | findstr <port>
                # 在Unix上可以使用: lsof -i :<port>
                if os.name == 'nt':
                    os.system(f'for /f "tokens=5" %a in (\'netstat -aon ^| findstr {config["port"]}\') do taskkill /F /PID %a')
                else:
                    os.system(f'lsof -ti:{config["port"]} | xargs kill -9')
                time.sleep(1)  # 等待端口释放
            except Exception as e:
                print(f"⚠️  无法释放端口: {e}")
        
        print(f"🚀 启动 {config['name']} (端口: {config['port']})")
        
        try:
            # 设置环境变量
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.base_dir)
            
            # 启动进程
            process = subprocess.Popen(
                [sys.executable, str(service_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.base_dir),
                env=env,
                bufsize=1,
                universal_newlines=True
            )
            
            self.processes[service_name] = {
                "process": process,
                "config": config,
                "start_time": time.time()
            }
            
            # 等待服务启动并检查状态
            max_wait = 10  # 最长等待10秒
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                if process.poll() is not None:
                    # 进程已退出
                    stdout, stderr = process.communicate()
                    print(f"❌ {config['name']} 启动失败")
                    if stdout:
                        print(f"输出: {stdout.strip()}")
                    if stderr:
                        print(f"错误: {stderr.strip()}")
                    return False
                
                # 检查服务是否正常响应
                try:
                    if not self.is_port_in_use(config["port"]):
                        print(f"✅ {config['name']} 启动成功")
                        return True
                except Exception:
                    pass
                
                time.sleep(0.5)
            
            print(f"⚠️  {config['name']} 启动超时，但进程仍在运行")
            return True
                
        except Exception as e:
            print(f"❌ 启动 {config['name']} 时发生错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def start_frontend(self):
        """启动前端服务"""
        print(f"🌐 启动前端服务 (端口: {self.frontend_port})")
        
        try:
            # 使用Python内置的HTTP服务器
            process = subprocess.Popen([
                sys.executable, "-m", "http.server", str(self.frontend_port)
            ],
            cwd=str(self.base_dir / "frontend"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
            )
            
            self.processes["frontend"] = {
                "process": process,
                "config": {"name": "前端服务", "port": self.frontend_port},
                "start_time": time.time()
            }
            
            time.sleep(2)
            
            if process.poll() is None:
                print(f"✅ 前端服务启动成功")
                return True
            else:
                print(f"❌ 前端服务启动失败")
                return False
                
        except Exception as e:
            print(f"❌ 启动前端服务时发生错误: {e}")
            return False
    
    def check_service_health(self):
        """检查服务健康状态"""
        print("\n🔍 检查服务健康状态...")
        
        import requests
        
        healthy_services = 0
        total_services = len(self.services)
        
        for service_name, config in self.services.items():
            try:
                response = requests.get(
                    f"http://localhost:{config['port']}/health",
                    timeout=5
                )
                if response.status_code == 200:
                    print(f"✅ {config['name']}: 健康")
                    healthy_services += 1
                else:
                    print(f"⚠️  {config['name']}: 响应异常 ({response.status_code})")
            except requests.exceptions.RequestException:
                print(f"❌ {config['name']}: 无响应")
        
        print(f"\n📊 服务状态: {healthy_services}/{total_services} 正常")
        return healthy_services == total_services
    
    def open_browser(self):
        """打开浏览器"""
        url = f"http://localhost:{self.frontend_port}"
        print(f"🌐 打开浏览器: {url}")
        
        try:
            webbrowser.open(url)
        except Exception as e:
            print(f"无法自动打开浏览器: {e}")
            print(f"请手动访问: {url}")
    
    def monitor_services(self):
        """监控服务状态"""
        log_dir = self.base_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        while self.running:
            try:
                time.sleep(10)  # 每10秒检查一次
                
                # 检查服务状态
                failed_services = []
                for service_name, service_info in self.processes.items():
                    process = service_info["process"]
                    config = service_info["config"]
                    
                    # 检查进程状态
                    if process.poll() is not None:
                        failed_services.append(service_name)
                        continue
                    
                    # 检查资源使用情况
                    try:
                        p = psutil.Process(process.pid)
                        cpu_percent = p.cpu_percent(interval=1)
                        memory_info = p.memory_info()
                        
                        # 记录资源使用情况
                        log_file = log_dir / f"{service_name}_stats.log"
                        with open(log_file, "a") as f:
                            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                            f.write(f"{timestamp} CPU: {cpu_percent}% MEM: {memory_info.rss / 1024 / 1024:.1f}MB\n")
                        
                        # 检查是否占用过多资源
                        if cpu_percent > 90:  # CPU使用率超过90%
                            print(f"\n⚠️  {config['name']} CPU使用率过高: {cpu_percent}%")
                        if memory_info.rss > 1024 * 1024 * 1024:  # 内存使用超过1GB
                            print(f"\n⚠️  {config['name']} 内存使用过高: {memory_info.rss / 1024 / 1024:.1f}MB")
                            
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        failed_services.append(service_name)
                    
                    # 检查服务响应
                    if service_name != "frontend":
                        try:
                            import requests
                            response = requests.get(
                                f"http://localhost:{config['port']}/health",
                                timeout=2
                            )
                            if response.status_code != 200:
                                failed_services.append(service_name)
                        except Exception:
                            failed_services.append(service_name)
                
                # 处理故障服务
                if failed_services:
                    print(f"\n⚠️  检测到服务故障: {', '.join(failed_services)}")
                    
                    for service_name in failed_services:
                        config = self.services[service_name]
                        print(f"\n🔄 正在重启 {config['name']}...")
                        
                        # 获取服务日志
                        if service_name in self.processes:
                            process = self.processes[service_name]["process"]
                            stdout, stderr = process.communicate()
                            
                            # 记录错误日志
                            error_log = log_dir / f"{service_name}_error.log"
                            with open(error_log, "a") as f:
                                f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                                if stdout:
                                    f.write(f"STDOUT:\n{stdout.decode()}\n")
                                if stderr:
                                    f.write(f"STDERR:\n{stderr.decode()}\n")
                        
                        # 尝试重启服务
                        if self.restart_service(service_name):
                            print(f"✅ {config['name']} 重启成功")
                        else:
                            print(f"❌ {config['name']} 重启失败")
                            
                            # 发送警告
                            warning_msg = f"服务 {config['name']} 重启失败，请检查日志文件: {error_log}"
                            print(f"\n⚠️  {warning_msg}")
                            
                            # 如果是关键服务，可以选择重启整个系统
                            if service_name in ["api_gateway", "video_capture"]:
                                print("\n❗ 关键服务故障，建议重启系统")
                                user_choice = input("是否重启整个系统? (y/N): ").strip().lower()
                                if user_choice == 'y':
                                    self.stop_all_services()
                                    time.sleep(2)
                                    self.start_all(skip_health_check=True)
                
            except Exception as e:
                print(f"\n❌ 监控服务出错: {e}")
                import traceback
                traceback.print_exc()
                
                # 记录监控错误
                monitor_log = log_dir / "monitor_error.log"
                with open(monitor_log, "a") as f:
                    f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                    f.write(f"监控错误: {str(e)}\n")
                    traceback.print_exc(file=f)
    
    def restart_service(self, service_name, force=False):
        """重启服务"""
        if service_name not in self.processes and not force:
            self.log_error(f"尝试重启不存在的服务: {service_name}")
            return False

        self.logger.info(f"准备重启服务: {service_name}")
        
        try:
            # 检查依赖服务
            if service_name in self.services:
                for dep in self.services[service_name]["dependencies"]:
                    if dep not in self.processes or self.processes[dep]["process"].poll() is not None:
                        self.logger.warning(f"依赖服务 {dep} 未运行，尝试启动...")
                        if not self.restart_service(dep, force=True):
                            self.log_error(f"无法启动依赖服务 {dep}")
                            return False
            
            # 停止旧进程
            if service_name in self.processes:
                old_process = self.processes[service_name]["process"]
                if old_process.poll() is None:
                    self.logger.info(f"终止旧进程: {service_name}")
                    try:
                        old_process.terminate()
                        old_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.logger.warning(f"进程未响应，强制终止: {service_name}")
                        old_process.kill()
                        old_process.wait()
            
            # 清理端口（如果需要）
            if service_name in self.services:
                port = self.services[service_name]["port"]
                if self.is_port_in_use(port):
                    self.logger.warning(f"端口 {port} 仍被占用，尝试清理...")
                    self._force_free_port(port)
            
            # 重新启动
            if service_name == "frontend":
                success = self.start_frontend()
            else:
                config = self.services[service_name]
                success = self.start_service(service_name, config)
            
            if success:
                self.system_status["total_restarts"] += 1
                self.log_service_status(service_name, "重启成功")
                return True
            else:
                self.log_error(f"服务 {service_name} 重启失败")
                return False
                
        except Exception as e:
            self.log_error(f"重启服务 {service_name} 时发生错误", exc_info=True)
            return False
    
    def _force_free_port(self, port):
        """强制释放端口"""
        try:
            if os.name == 'nt':  # Windows
                cmd = f'for /f "tokens=5" %a in (\'netstat -aon ^| findstr :{port}\') do taskkill /F /PID %a'
                subprocess.run(cmd, shell=True, stderr=subprocess.PIPE)
            else:  # Unix/Linux/macOS
                cmd = f'lsof -ti:{port} | xargs kill -9'
                subprocess.run(cmd, shell=True, stderr=subprocess.PIPE)
            time.sleep(1)  # 等待端口释放
        except Exception as e:
            self.logger.warning(f"清理端口 {port} 时出错: {e}")
    
    def get_system_status(self):
        """获取系统状态报告"""
        status = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "uptime": time.time() - self.system_status["start_time"] if self.system_status["start_time"] else 0,
            "total_restarts": self.system_status["total_restarts"],
            "services": {},
            "resources": {},
            "last_error": self.system_status["last_error"]
        }
        
        # 收集服务状态
        for service_name, process_info in self.processes.items():
            process = process_info["process"]
            config = process_info["config"]
            
            service_status = {
                "running": process.poll() is None,
                "pid": process.pid if process.poll() is None else None,
                "start_time": time.strftime(
                    "%Y-%m-%d %H:%M:%S", 
                    time.localtime(process_info["start_time"])
                ),
                "port": config["port"] if service_name in self.services else None
            }
            
            # 尝试获取进程资源使用情况
            try:
                if service_status["running"]:
                    p = psutil.Process(process.pid)
                    service_status.update({
                        "cpu_percent": p.cpu_percent(),
                        "memory_mb": p.memory_info().rss / 1024 / 1024,
                        "threads": p.num_threads()
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            
            status["services"][service_name] = service_status
        
        # 系统资源使用情况
        try:
            status["resources"] = {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent
            }
        except Exception as e:
            self.logger.warning(f"获取系统资源信息失败: {e}")
        
        return status
    
    def check_system_health(self):
        """检查系统健康状态"""
        status = self.get_system_status()
        issues = []
        
        # 检查必需服务
        for service_name, config in self.services.items():
            if config.get("required", True):  # 默认服务都是必需的
                service_status = status["services"].get(service_name, {})
                if not service_status.get("running"):
                    issues.append(f"必需服务 {config['name']} 未运行")
                elif service_status.get("cpu_percent", 0) > 90:
                    issues.append(f"服务 {config['name']} CPU使用率过高")
                elif service_status.get("memory_mb", 0) > 1024:  # 超过1GB
                    issues.append(f"服务 {config['name']} 内存使用过高")
        
        # 检查系统资源
        if status["resources"].get("cpu_percent", 0) > 90:
            issues.append("系统CPU使用率过高")
        if status["resources"].get("memory_percent", 0) > 90:
            issues.append("系统内存使用率过高")
        if status["resources"].get("disk_percent", 0) > 90:
            issues.append("磁盘使用率过高")
        
        if issues:
            self.logger.warning("发现系统问题:")
            for issue in issues:
                self.logger.warning(f"- {issue}")
            return False
        
        return True
    
    def stop_all_services(self):
        """停止所有服务"""
        print("\n🛑 正在停止所有服务...")
        self.running = False
        
        for service_name, service_info in self.processes.items():
            process = service_info["process"]
            config = service_info["config"]
            
            print(f"停止 {config['name']}...")
            
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                    print(f"✅ {config['name']} 已停止")
                except subprocess.TimeoutExpired:
                    print(f"⚠️  强制终止 {config['name']}")
                    process.kill()
                    process.wait()

    def check_dependencies(self):
        """检查系统依赖（Python版本、必要包、模型文件、端口占用等）"""
        print("🔍 开始检查系统依赖...")

        # 1. 检查 Python 版本（要求 3.8+）
        if sys.version_info < (3, 8):
            print("❌ 需要 Python 3.8 或更高版本！当前版本：", sys.version)
            return False

        # 2. 检查必要 Python 包（根据你的项目需求调整）
        required_packages = {
            "fastapi": "fastapi",
            "uvicorn": "uvicorn",
            "opencv-python": "cv2",
            "ultralytics": "ultralytics",
            "redis": "redis",
            "psycopg2-binary": "psycopg2",
            "openai": "openai",
            "numpy": "numpy",
            "requests": "requests"
        }
        missing_packages = []
        for pkg_name, import_name in required_packages.items():
            try:
                __import__(import_name)
                print(f"✅ {pkg_name} 已安装")
            except ImportError:
                missing_packages.append(pkg_name)
                print(f"❌ {pkg_name} 未安装")

        if missing_packages:
            print(f"\n❌ 缺少必要 Python 包：{', '.join(missing_packages)}")
            print("请按以下步骤修复：")
            print(" 1. 升级 pip：pip install --upgrade pip")
            print(" 2. 安装依赖：pip install -r requirements.txt")
            print("\n（可选）遇到 OpenCV 问题时，尝试：")
            print(" pip install opencv-python-headless  # 无 GUI 依赖版本")
            print(" pip install psycopg2-binary --no-cache-dir  # 解决 PostgreSQL 安装问题")
            return False

        # 3. 检查必要文件（YOLO 模型示例）
        model_path = self.base_dir / "models" / "yolov8n.pt"
        if not model_path.exists():
            print("❌ 未找到 YOLO 模型文件：", model_path)
            print("请运行：python download_models.py --setup  自动下载模型")
            return False

        # 4. 检查服务端口是否被占用（示例逻辑，可根据 self.services 扩展）
        def is_port_used(port):
            """检查指定端口是否被占用"""
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(("localhost", port)) == 0

        # 遍历服务配置，检查端口
        for service_name, config in self.services.items():
            port = config["port"]
            if is_port_used(port):
                print(f"❌ 端口 {port} 被占用（{config['name']} 无法启动）")
                return False

        # 5. 检查必要目录（models、logs、frontend 等）
        required_dirs = ['models', 'logs', 'frontend']
        for dir_name in required_dirs:
            dir_path = self.base_dir / dir_name
            if not dir_path.exists():
                print(f"📌 自动创建缺失目录：{dir_name}")
                dir_path.mkdir(exist_ok=True)

        print("✅ 所有依赖检查通过！")
        return True



    def start_all(self, skip_health_check=False):
        """启动所有服务"""
        print("🚀 启动AI导航系统")
        print("=" * 50)


        # 检查依赖
        if not self.check_dependencies():
            return False
        
        # 检查必要的文件和目录
        required_dirs = ['models', 'logs', 'frontend']
        for dir_name in required_dirs:
            dir_path = self.base_dir / dir_name
            if not dir_path.exists():
                print(f"📁 创建目录: {dir_name}")
                dir_path.mkdir(exist_ok=True)
        
        # 检查模型文件
        model_path = self.base_dir / "models" / "yolov8n.pt"
        if not model_path.exists():
            print("⚠️  未找到YOLO模型文件")
            print("请运行: python download_models.py --setup")
            return False
        
        # 启动顺序和依赖关系
        service_order = [
            "api_gateway",      # 首先启动API网关
            "video_capture",    # 然后是视频采集服务
            "yolo_detection",   # YOLO检测依赖视频服务
            "perspective_correction",  # 依赖视频服务
            "llm_decision"      # 最后启动决策服务
        ]
        
        # 启动后端服务
        print("\n📡 启动后端服务...")
        success_count = 0
        failed_services = []
        
        for service_name in service_order:
            config = self.services[service_name]
            print(f"\n正在启动 {config['name']}...")
            
            # 检查依赖服务是否正常
            if service_name in ["yolo_detection", "perspective_correction"]:
                if "video_capture" not in self.processes:
                    print(f"⚠️  {config['name']} 依赖的视频采集服务未启动")
                    failed_services.append(service_name)
                    continue
            
            # 尝试启动服务
            retry_count = 3  # 最多重试3次
            for attempt in range(retry_count):
                if attempt > 0:
                    print(f"第 {attempt + 1} 次尝试启动 {config['name']}...")
                    time.sleep(2)  # 等待之前的进程完全终止
                
                if self.start_service(service_name, config):
                    success_count += 1
                    break
                elif attempt == retry_count - 1:
                    failed_services.append(service_name)
            
            time.sleep(1)  # 服务启动间隔
        
        # 报告启动状态
        if failed_services:
            print(f"\n⚠️  以下服务启动失败:")
            for service_name in failed_services:
                print(f"- {self.services[service_name]['name']}")
        
        if success_count != len(self.services):
            print(f"\n⚠️  部分服务启动失败 ({success_count}/{len(self.services)})")
            user_choice = input("是否继续启动前端? (y/N): ").strip().lower()
            if user_choice != 'y':
                return False
        
        # 启动前端
        print("\n🌐 启动前端服务...")
        if not self.start_frontend():
            print("❌ 前端服务启动失败")
            user_choice = input("是否继续运行系统? (y/N): ").strip().lower()
            if user_choice != 'y':
                return False
        
        # 健康检查
        if not skip_health_check:
            print("\n🔍 执行健康检查...")
            time.sleep(5)  # 等待服务完全启动
            health_status = self.check_service_health()
            
            if not health_status:
                print("\n⚠️  健康检查发现问题")
                print("建议操作:")
                print("1. 检查日志文件获取详细错误信息")
                print("2. 运行 'python install_dependencies.py' 验证依赖")
                print("3. 确保所有必要的端口未被占用")
                user_choice = input("\n是否继续运行系统? (y/N): ").strip().lower()
                if user_choice != 'y':
                    return False
        
        # 打开浏览器
        time.sleep(2)
        self.open_browser()
        
        # 启动监控
        monitor_thread = threading.Thread(target=self.monitor_services, daemon=True)
        monitor_thread.start()
        
        print("\n" + "=" * 50)
        print("🎉 AI导航系统启动完成!")
        print(f"🌐 前端地址: http://localhost:{self.frontend_port}")
        print(f"📡 API网关: http://localhost:{self.services['api_gateway']['port']}")
        
        if failed_services:
            print("\n⚠️  部分服务未正常启动，系统功能可能受限")
        
        print("\n按 Ctrl+C 停止系统")
        
        return True
    
    def show_status(self):
        """显示系统状态"""
        print("\n📊 系统状态")
        print("-" * 40)
        
        for service_name, service_info in self.processes.items():
            process = service_info["process"]
            config = service_info["config"]
            
            if process.poll() is None:
                runtime = time.time() - service_info["start_time"]
                status = f"✅ 运行中 ({runtime:.0f}s)"
            else:
                status = "❌ 已停止"
            
            print(f"{config['name']}: {status}")

def signal_handler(signum, frame):
    """处理系统信号"""
    print("\n\n收到停止信号...")
    if 'manager' in globals():
        manager.stop_all_services()
    sys.exit(0)


class SystemManager:
    def __init__(self):
        # 其他已有属性（保持不变）
        self.processes = {}
        self.base_dir = Path(__file__).parent
        self.running = True
        self.log_dir = self.base_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)

        # 配置日志系统（现在会调用下面定义的 setup_logging 方法）
        self.setup_logging()

        # 前端端口（之前已添加）
        self.frontend_port = 3000

        # 服务配置、系统状态等其他代码...
        # ...

    def check_dependencies(self):
        """检查系统依赖（Python版本、必要包、模型文件、端口等）"""
        print("🔍 开始检查系统依赖...")

        # 1. 检查Python版本（要求3.8+）
        import sys
        if sys.version_info < (3, 8):
            print("❌ 需要Python 3.8或更高版本！当前版本：", sys.version)
            return False

        # 2. 检查必要Python包
        required_packages = {
            "fastapi": "fastapi",
            "uvicorn": "uvicorn",
            "opencv-python": "cv2",
            "ultralytics": "ultralytics",
            "redis": "redis",
            "psycopg2-binary": "psycopg2",
            "openai": "openai",
            "numpy": "numpy",
            "requests": "requests"
        }
        missing_packages = []
        for pkg_name, import_name in required_packages.items():
            try:
                __import__(import_name)
                print(f"✅ {pkg_name} 已安装")
            except ImportError:
                missing_packages.append(pkg_name)
                print(f"❌ {pkg_name} 未安装")

        if missing_packages:
            print(f"\n❌ 缺少必要Python包：{', '.join(missing_packages)}")
            print("请执行以下命令安装：")
            print("pip install --upgrade pip")
            print("pip install -r requirements.txt")
            return False

        # 3. 检查YOLO模型文件
        model_path = self.base_dir / "models" / "yolov8n.pt"
        if not model_path.exists():
            print(f"❌ 未找到YOLO模型文件：{model_path}")
            print("请运行：python download_models.py --setup")
            return False

        # 4. 检查服务端口是否被占用
        def is_port_used(port):
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(("localhost", port)) == 0

        for service_name, config in self.services.items():
            port = config["port"]
            if is_port_used(port):
                print(f"❌ 端口 {port} 被占用（{config['name']} 无法启动）")
                return False

        # 5. 检查必要目录
        required_dirs = ['models', 'logs', 'frontend']
        for dir_name in required_dirs:
            dir_path = self.base_dir / dir_name
            if not dir_path.exists():
                print(f"📌 创建缺失目录：{dir_name}")
                dir_path.mkdir(exist_ok=True)

        print("✅ 所有依赖检查通过！")
        return True

    def __init__(self):
        # 基础属性初始化
        self.processes = {}  # 用于存储服务进程
        self.base_dir = Path(__file__).parent  # 项目根目录
        self.running = True  # 系统运行状态
        self.frontend_port = 3000  # 前端端口

        # 日志目录
        self.log_dir = self.base_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)

        # 初始化日志
        self.setup_logging()

        # 👇 必须添加的服务配置（关键！）
        self.services = {
            "api_gateway": {
                "path": "backend/services/api_gateway/main.py",
                "port": int(os.getenv("API_GATEWAY_PORT", "8000")),
                "name": "API网关",
                "required": True,
                "dependencies": []
            },
            "video_capture": {
                "path": "backend/services/video_capture/main.py",
                "port": int(os.getenv("VIDEO_CAPTURE_PORT", "8001")),
                "name": "视频采集服务",
                "required": True,
                "dependencies": ["api_gateway"]
            },
            "yolo_detection": {
                "path": "backend/services/yolo_detection/main.py",
                "port": int(os.getenv("YOLO_DETECTION_PORT", "8002")),
                "name": "YOLO检测服务",
                "required": True,
                "dependencies": ["video_capture"]
            },
            "perspective_correction": {
                "path": "backend/services/perspective_correction/main.py",
                "port": int(os.getenv("PERSPECTIVE_CORRECTION_PORT", "8003")),
                "name": "透视校正服务",
                "required": False,
                "dependencies": ["video_capture"]
            },
            "llm_decision": {
                "path": "backend/services/llm_decision/main.py",
                "port": int(os.getenv("LLM_DECISION_PORT", "8004")),
                "name": "LLM决策服务",
                "required": False,
                "dependencies": ["yolo_detection"]
            }
        }

        # 系统状态
        self.system_status = {
            "start_time": None,
            "total_restarts": 0,
            "service_stats": {},
            "last_error": None
        }

    def start_service(self, service_name, config):
        """启动单个服务进程"""
        try:
            # 构建服务文件路径
            service_path = self.base_dir / config["path"]
            if not service_path.exists():
                print(f"❌ 服务文件不存在: {service_path}")
                return False

            # 启动命令（假设使用Python运行服务）
            cmd = [
                sys.executable,  # 当前Python解释器路径
                str(service_path),
                "--port", str(config["port"])
            ]

            # 启动进程（后台运行，重定向输出到日志）
            log_file = self.log_dir / f"{service_name}.log"
            with open(log_file, "a") as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=str(self.base_dir)
                )

            # 记录进程信息
            self.processes[service_name] = {
                "process": process,
                "config": config,
                "start_time": time.time()
            }

            # 简单检查进程是否启动成功
            time.sleep(1)
            if process.poll() is None:  # None表示进程仍在运行
                print(f"✅ {config['name']} 启动成功 (PID: {process.pid})")
                return True
            else:
                print(f"❌ {config['name']} 启动失败（进程已退出）")
                return False

        except Exception as e:
            print(f"❌ 启动{config['name']}时出错: {str(e)}")
            return False

    def start_frontend(self):
        """启动前端服务（假设是Node.js项目）"""
        try:
            # 前端目录路径
            frontend_dir = self.base_dir / "frontend"
            if not frontend_dir.exists():
                print(f"❌ 前端目录不存在: {frontend_dir}")
                return False

            # 检查前端依赖是否安装（package.json是否存在）
            package_file = frontend_dir / "package.json"
            if not package_file.exists():
                print(f"❌ 前端项目配置文件缺失: {package_file}")
                print("请确认前端项目已正确部署到 frontend 目录")
                return False

            # 启动命令（假设使用npm启动，默认脚本为start）
            cmd = ["npm", "start"]
            if sys.platform.startswith("win32"):
                cmd = ["npm.cmd", "start"]  # Windows系统适配

            # 启动前端进程，输出重定向到日志
            log_file = self.log_dir / "frontend.log"
            with open(log_file, "a") as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=str(frontend_dir),
                    env={**os.environ, "PORT": str(self.frontend_port)}  # 传递端口环境变量
                )

            # 记录前端进程信息
            self.processes["frontend"] = {
                "process": process,
                "config": {"name": "前端服务", "port": self.frontend_port},
                "start_time": time.time()
            }

            # 简单检查启动状态
            time.sleep(2)
            if process.poll() is None:
                print(f"✅ 前端服务启动成功 (端口: {self.frontend_port})")
                return True
            else:
                print("❌ 前端服务启动失败（进程已退出）")
                print("请查看日志文件了解详情: logs/frontend.log")
                return False

        except Exception as e:
            print(f"❌ 启动前端服务时出错: {str(e)}")
            return False

    def start_all(self, skip_health_check=False):
        """启动所有服务"""
        print("🚀 启动AI导航系统")
        print("=" * 50)

        # 检查依赖
        if not self.check_dependencies():
            return False

        # 检查必要的文件和目录
        required_dirs = ['models', 'logs', 'frontend']
        for dir_name in required_dirs:
            dir_path = self.base_dir / dir_name
            if not dir_path.exists():
                print(f"📁 创建目录: {dir_name}")
                dir_path.mkdir(exist_ok=True)

        # 检查模型文件
        model_path = self.base_dir / "models" / "yolov8n.pt"
        if not model_path.exists():
            print("⚠️  未找到YOLO模型文件")
            print("请运行: python download_models.py --setup")
            return False

        # 启动顺序和依赖关系
        service_order = [
            "api_gateway",  # 首先启动API网关
            "video_capture",  # 然后是视频采集服务
            "yolo_detection",  # YOLO检测依赖视频服务
            "perspective_correction",  # 依赖视频服务
            "llm_decision"  # 最后启动决策服务
        ]

        # 启动后端服务
        print("\n📡 启动后端服务...")
        success_count = 0
        failed_services = []

        for service_name in service_order:
            config = self.services[service_name]
            print(f"\n正在启动 {config['name']}...")

            # 检查依赖服务是否正常
            if service_name in ["yolo_detection", "perspective_correction"]:
                if "video_capture" not in self.processes:
                    print(f"⚠️  {config['name']} 依赖的视频采集服务未启动")
                    failed_services.append(service_name)
                    continue

            # 尝试启动服务
            retry_count = 3  # 最多重试3次
            for attempt in range(retry_count):
                if attempt > 0:
                    print(f"第 {attempt + 1} 次尝试启动 {config['name']}...")
                    time.sleep(2)  # 等待之前的进程完全终止

                if self.start_service(service_name, config):
                    success_count += 1
                    break
                elif attempt == retry_count - 1:
                    failed_services.append(service_name)

            time.sleep(1)  # 服务启动间隔

        # 报告启动状态
        if failed_services:
            print(f"\n⚠️  以下服务启动失败:")
            for service_name in failed_services:
                print(f"- {self.services[service_name]['name']}")

        if success_count != len(self.services):
            print(f"\n⚠️  部分服务启动失败 ({success_count}/{len(self.services)})")
            user_choice = input("是否继续启动前端? (y/N): ").strip().lower()
            if user_choice != 'y':
                return False

        # 启动前端
        print("\n🌐 启动前端服务...")
        if not self.start_frontend():
            print("❌ 前端服务启动失败")
            user_choice = input("是否继续运行系统? (y/N): ").strip().lower()
            if user_choice != 'y':
                return False

        # 健康检查
        if not skip_health_check:
            print("\n🔍 执行健康检查...")
            time.sleep(5)  # 等待服务完全启动
            health_status = self.check_service_health()

            if not health_status:
                print("\n⚠️  健康检查发现问题")
                print("建议操作:")
                print("1. 检查日志文件获取详细错误信息")
                print("2. 运行 'python install_dependencies.py' 验证依赖")
                print("3. 确保所有必要的端口未被占用")
                user_choice = input("\n是否继续运行系统? (y/N): ").strip().lower()
                if user_choice != 'y':
                    return False

        # 打开浏览器
        time.sleep(2)
        self.open_browser()

        # 启动监控
        monitor_thread = threading.Thread(target=self.monitor_services, daemon=True)
        monitor_thread.start()

        print("\n" + "=" * 50)
        print("🎉 AI导航系统启动完成!")
        print(f"🌐 前端地址: http://localhost:{self.frontend_port}")
        print(f"📡 API网关: http://localhost:{self.services['api_gateway']['port']}")

        if failed_services:
            print("\n⚠️  部分服务未正常启动，系统功能可能受限")

        print("\n按 Ctrl+C 停止系统")

        return True

    # 还需要确保以下方法也存在于类中（如果之前未定义）：
    # - check_dependencies (已添加)
    # - start_service (启动单个服务的方法)
    # - start_frontend (启动前端的方法)
    # - check_service_health (健康检查方法)
    # - open_browser (打开浏览器的方法)
    # - monitor_services (监控服务的方法)
    # 👇 添加日志配置方法
    def setup_logging(self):
        """配置系统日志（输出到文件和控制台）"""
        import logging
        from logging.handlers import RotatingFileHandler

        # 日志格式
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(log_format)

        # 日志级别（DEBUG/INFO/WARNING/ERROR）
        log_level = logging.INFO

        # 初始化 logger
        self.logger = logging.getLogger("SystemManager")
        self.logger.setLevel(log_level)
        self.logger.handlers = []  # 清除已有处理器

        # 1. 添加控制台日志处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # 2. 添加文件日志处理器（按大小切割，保留备份）
        log_file = self.log_dir / "system.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=1024 * 1024 * 5,  # 5MB
            backupCount=5,  # 保留5个备份
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self.logger.info("日志系统初始化完成")









def main():
    parser = argparse.ArgumentParser(description="AI导航系统启动器")
    parser.add_argument("--skip-health-check", action="store_true", 
                       help="跳过健康检查")
    parser.add_argument("--no-browser", action="store_true",
                       help="不自动打开浏览器")
    parser.add_argument("--status", action="store_true",
                       help="显示系统状态")
    
    args = parser.parse_args()
    
    global manager
    manager = SystemManager()
    
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    if args.status:
        manager.show_status()
        return
    
    # 启动系统
    if manager.start_all(skip_health_check=args.skip_health_check):
        # 启动监控线程
        monitor_thread = threading.Thread(target=manager.monitor_services, daemon=True)
        monitor_thread.start()
        
        try:
            # 保持主线程运行
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n用户中断...")
        finally:
            manager.stop_all_services()
    else:
        print("❌ 系统启动失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
