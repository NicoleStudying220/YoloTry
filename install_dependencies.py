#!/usr/bin/env python3
"""
依赖安装脚本 - 专为macOS优化
解决常见的包安装问题
"""

import subprocess
import sys
import os
import platform

def run_command(command, description=""):
    """运行命令并显示输出"""
    print(f"\n🔄 {description}")
    print(f"执行命令: {command}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True
        )
        print("✅ 成功")
        if result.stdout:
            print(f"输出: {result.stdout[:200]}...")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 失败: {e}")
        if e.stderr:
            print(f"错误: {e.stderr}")
        return False

def check_system():
    """检查系统环境"""
    print("🔍 检查系统环境...")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    
    # 检查是否在虚拟环境中
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ 在虚拟环境中")
    else:
        print("⚠️  不在虚拟环境中，建议使用conda或venv")

def upgrade_pip():
    """升级pip"""
    print("\n📦 升级pip...")
    commands = [
        "python -m pip install --upgrade pip",
        "pip install --upgrade setuptools wheel"
    ]
    
    for cmd in commands:
        run_command(cmd, f"执行: {cmd}")

def install_basic_packages():
    """安装基础包"""
    print("\n📦 安装基础依赖包...")
    
    # 基础包列表
    basic_packages = [
        "numpy>=1.21.0",
        "pillow>=9.0.0",
        "requests>=2.28.0",
        "python-dotenv>=0.19.0",
        "tqdm>=4.64.0",
        "psutil>=5.9.0"
    ]
    
    for package in basic_packages:
        run_command(f"pip install {package}", f"安装 {package}")

def install_web_framework():
    """安装Web框架"""
    print("\n🌐 安装Web框架...")
    
    web_packages = [
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.20.0", 
        "websockets>=11.0",
        "httpx>=0.24.0",
        "python-multipart>=0.0.5",
        "pydantic>=1.10.0"
    ]
    
    for package in web_packages:
        run_command(f"pip install {package}", f"安装 {package}")

def install_opencv():
    """安装OpenCV - macOS优化版本"""
    print("\n📷 安装OpenCV...")
    
    # 先尝试安装headless版本（更稳定）
    opencv_commands = [
        "pip uninstall opencv-python opencv-contrib-python -y",
        "pip install opencv-python-headless>=4.8.0",
        "pip install opencv-contrib-python-headless>=4.8.0"
    ]
    
    for cmd in opencv_commands:
        run_command(cmd, f"OpenCV: {cmd}")

def install_ai_packages():
    """安装AI相关包"""
    print("\n🤖 安装AI/ML包...")
    
    # PyTorch (macOS优化)
    if platform.processor() == 'arm':  # Apple Silicon
        torch_cmd = "pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
    else:  # Intel Mac
        torch_cmd = "pip install torch>=1.13.0 torchvision>=0.14.0"
    
    run_command(torch_cmd, "安装PyTorch")
    
    # 其他AI包
    ai_packages = [
        "ultralytics>=8.0.0",
        "openai>=1.0.0",
        "matplotlib>=3.5.0",
        "scipy>=1.9.0"
    ]
    
    for package in ai_packages:
        run_command(f"pip install {package}", f"安装 {package}")

def install_database_packages():
    """安装数据库相关包"""
    print("\n🗄️ 安装数据库包...")
    
    # 对于macOS，psycopg2-binary可能需要特殊处理
    db_commands = [
        "pip install redis>=4.0.0",
        "pip install sqlalchemy>=1.4.0"
    ]
    
    for cmd in db_commands:
        run_command(cmd, f"数据库: {cmd}")
    
    # PostgreSQL驱动特殊处理
    print("\n🔧 安装PostgreSQL驱动...")
    psycopg2_commands = [
        "pip install psycopg2-binary>=2.9.0 --no-cache-dir",
        # 如果上面失败，尝试编译版本
        "brew install postgresql || echo 'Homebrew PostgreSQL not installed'",
        "pip install psycopg2>=2.9.0 || echo 'psycopg2 compilation failed'"
    ]
    
    for cmd in psycopg2_commands:
        run_command(cmd, f"PostgreSQL: {cmd}")

def verify_installation():
    """验证安装"""
    print("\n✅ 验证安装...")
    
    test_imports = {
        "cv2": "OpenCV",
        "fastapi": "FastAPI", 
        "uvicorn": "Uvicorn",
        "ultralytics": "YOLOv8",
        "torch": "PyTorch",
        "redis": "Redis",
        "openai": "OpenAI",
        "numpy": "NumPy",
        "requests": "Requests"
    }
    
    success_count = 0
    total_count = len(test_imports)
    
    for module, name in test_imports.items():
        try:
            __import__(module)
            print(f"✅ {name}: 成功")
            success_count += 1
        except ImportError as e:
            print(f"❌ {name}: 失败 - {e}")
    
    print(f"\n📊 安装结果: {success_count}/{total_count} 包可用")
    
    if success_count == total_count:
        print("🎉 所有依赖安装成功！")
        return True
    else:
        print("⚠️  部分包安装失败，请检查错误信息")
        return False

def create_env_file():
    """创建环境配置文件"""
    env_file = ".env"
    env_example = ".env.example"
    
    if not os.path.exists(env_file) and os.path.exists(env_example):
        print(f"\n📝 创建环境配置文件...")
        try:
            with open(env_example, 'r') as src:
                content = src.read()
            
            with open(env_file, 'w') as dst:
                dst.write(content)
            
            print("✅ 环境配置文件创建成功")
            print("⚠️  请编辑 .env 文件，填入您的API密钥")
        except Exception as e:
            print(f"❌ 创建环境配置文件失败: {e}")

def main():
    print("🚀 AI导航系统 - 依赖安装脚本")
    print("=" * 50)
    
    # 检查系统
    check_system()
    
    print("\n选择安装模式:")
    print("1. 完整安装（推荐）")
    print("2. 基础安装（仅核心包）")
    print("3. 修复安装（重新安装问题包）")
    print("4. 仅验证当前安装")
    
    try:
        choice = input("\n请选择 (1-4): ").strip()
    except KeyboardInterrupt:
        print("\n\n安装已取消")
        return
    
    if choice == "1":
        # 完整安装
        print("\n🚀 开始完整安装...")
        upgrade_pip()
        install_basic_packages()
        install_web_framework()
        install_opencv()
        install_ai_packages()
        install_database_packages()
        verify_installation()
        create_env_file()
        
    elif choice == "2":
        # 基础安装
        print("\n🚀 开始基础安装...")
        upgrade_pip()
        install_basic_packages()
        install_web_framework()
        install_opencv()
        verify_installation()
        
    elif choice == "3":
        # 修复安装
        print("\n🔧 修复安装...")
        print("正在重新安装常见问题包...")
        
        fix_commands = [
            "pip uninstall opencv-python opencv-contrib-python -y",
            "pip install opencv-python-headless",
            "pip install psycopg2-binary --no-cache-dir --force-reinstall",
            "pip install ultralytics --upgrade",
            "pip install fastapi uvicorn --upgrade"
        ]
        
        for cmd in fix_commands:
            run_command(cmd, f"修复: {cmd}")
        
        verify_installation()
        
    elif choice == "4":
        # 仅验证
        verify_installation()
        
    else:
        print("无效选择")
        return
    
    print("\n" + "=" * 50)
    print("🎉 依赖安装完成！")
    print("\n接下来的步骤:")
    print("1. 编辑 .env 文件，添加您的DeepSeek API密钥")
    print("2. 运行: python download_models.py --setup")
    print("3. 运行: python start_system.py")

if __name__ == "__main__":
    main()
