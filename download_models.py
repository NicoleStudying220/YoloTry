#!/usr/bin/env python3
"""
模型下载脚本
下载和配置AI导航系统所需的各种模型
"""

import os
import sys
import requests
import hashlib
from pathlib import Path
from tqdm import tqdm
import argparse

class ModelDownloader:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.models_dir = self.base_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # 可用的YOLO模型
        self.yolo_models = {
            "yolov8n.pt": {
                "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
                "size": "6.2MB",
                "description": "YOLOv8 Nano - 最快速度，适合实时检测",
                "md5": "f3f3d8e1234567890abcdef123456789"  # 示例MD5
            },
            "yolov8s.pt": {
                "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt", 
                "size": "22MB",
                "description": "YOLOv8 Small - 平衡速度和精度",
                "md5": "a1b2c3d4e5f6789012345678901234ab"
            },
            "yolov8m.pt": {
                "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt",
                "size": "50MB", 
                "description": "YOLOv8 Medium - 更高精度",
                "md5": "b2c3d4e5f6789012345678901234abcd"
            },
            "yolov8l.pt": {
                "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt",
                "size": "87MB",
                "description": "YOLOv8 Large - 高精度检测",
                "md5": "c3d4e5f6789012345678901234abcdef"
            },
            "yolov8x.pt": {
                "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt",
                "size": "136MB",
                "description": "YOLOv8 Extra Large - 最高精度",
                "md5": "d4e5f6789012345678901234abcdef12"
            }
        }
        
        # 自定义导航模型（如果有）
        self.custom_models = {
            "navigation_yolo.pt": {
                "url": "https://example.com/custom/navigation_yolo.pt",
                "size": "25MB",
                "description": "专门训练的导航物体检测模型",
                "md5": "e5f6789012345678901234abcdef1234"
            }
        }
    
    def calculate_md5(self, file_path):
        """计算文件MD5值"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def download_file(self, url, file_path, expected_md5=None):
        """下载文件并显示进度"""
        try:
            print(f"正在下载: {file_path.name}")
            print(f"URL: {url}")
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(file_path, 'wb') as file:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=file_path.name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file.write(chunk)
                            pbar.update(len(chunk))
            
            # 验证MD5（如果提供）
            if expected_md5:
                actual_md5 = self.calculate_md5(file_path)
                if actual_md5.lower() != expected_md5.lower():
                    print(f"⚠️  MD5校验失败！期望: {expected_md5}, 实际: {actual_md5}")
                    return False
                else:
                    print("✅ MD5校验通过")
            
            print(f"✅ 下载完成: {file_path}")
            return True
            
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            if file_path.exists():
                file_path.unlink()
            return False
    
    def check_model_exists(self, model_name):
        """检查模型是否已存在"""
        model_path = self.models_dir / model_name
        return model_path.exists()
    
    def list_available_models(self):
        """列出所有可用的模型"""
        print("\n📦 可用的YOLO模型:")
        print("-" * 80)
        
        for model_name, info in self.yolo_models.items():
            status = "✅ 已下载" if self.check_model_exists(model_name) else "⬇️  未下载"
            print(f"{model_name:<15} | {info['size']:<8} | {status:<8} | {info['description']}")
        
        if self.custom_models:
            print("\n🎯 自定义导航模型:")
            print("-" * 80)
            
            for model_name, info in self.custom_models.items():
                status = "✅ 已下载" if self.check_model_exists(model_name) else "⬇️  未下载"
                print(f"{model_name:<20} | {info['size']:<8} | {status:<8} | {info['description']}")
    
    def download_yolo_model(self, model_name):
        """下载指定的YOLO模型"""
        if model_name not in self.yolo_models:
            print(f"❌ 模型 {model_name} 不存在")
            return False
        
        model_path = self.models_dir / model_name
        
        if model_path.exists():
            print(f"ℹ️  模型 {model_name} 已存在，跳过下载")
            return True
        
        model_info = self.yolo_models[model_name]
        return self.download_file(
            model_info["url"], 
            model_path, 
            model_info.get("md5")
        )
    
    def download_all_models(self):
        """下载所有模型"""
        print("🚀 开始下载所有模型...")
        
        success_count = 0
        total_count = len(self.yolo_models) + len(self.custom_models)
        
        # 下载YOLO模型
        for model_name in self.yolo_models:
            if self.download_yolo_model(model_name):
                success_count += 1
        
        # 下载自定义模型
        for model_name, model_info in self.custom_models.items():
            model_path = self.models_dir / model_name
            
            if model_path.exists():
                print(f"ℹ️  模型 {model_name} 已存在，跳过下载")
                success_count += 1
                continue
            
            if self.download_file(model_info["url"], model_path, model_info.get("md5")):
                success_count += 1
        
        print(f"\n📊 下载完成: {success_count}/{total_count} 个模型")
        return success_count == total_count
    
    def download_recommended_models(self):
        """下载推荐的模型组合"""
        recommended = ["yolov8n.pt", "yolov8s.pt"]  # 推荐用于实时检测
        
        print("🎯 下载推荐模型组合...")
        print("推荐模型: " + ", ".join(recommended))
        
        success_count = 0
        for model_name in recommended:
            if self.download_yolo_model(model_name):
                success_count += 1
        
        print(f"\n📊 推荐模型下载完成: {success_count}/{len(recommended)}")
        return success_count == len(recommended)
    
    def verify_models(self):
        """验证已下载的模型"""
        print("\n🔍 验证模型文件...")
        
        verified_count = 0
        total_count = 0
        
        all_models = {**self.yolo_models, **self.custom_models}
        
        for model_name, model_info in all_models.items():
            model_path = self.models_dir / model_name
            
            if not model_path.exists():
                continue
            
            total_count += 1
            print(f"验证: {model_name}")
            
            # 检查文件大小
            file_size = model_path.stat().st_size
            print(f"  文件大小: {file_size / (1024*1024):.1f}MB")
            
            # MD5验证
            if "md5" in model_info:
                actual_md5 = self.calculate_md5(model_path)
                expected_md5 = model_info["md5"]
                
                if actual_md5.lower() == expected_md5.lower():
                    print("  ✅ MD5校验通过")
                    verified_count += 1
                else:
                    print(f"  ❌ MD5校验失败: {actual_md5}")
            else:
                print("  ⚠️  无MD5校验信息")
                verified_count += 1
        
        print(f"\n📊 验证结果: {verified_count}/{total_count} 个模型通过验证")
    
    def clean_models(self):
        """清理下载的模型"""
        print("🧹 清理模型文件...")
        
        if not self.models_dir.exists():
            print("模型目录不存在")
            return
        
        model_files = list(self.models_dir.glob("*.pt"))
        
        if not model_files:
            print("没有找到模型文件")
            return
        
        print(f"找到 {len(model_files)} 个模型文件:")
        for model_file in model_files:
            print(f"  - {model_file.name}")
        
        confirm = input("\n确认删除所有模型文件? (y/N): ")
        if confirm.lower() == 'y':
            for model_file in model_files:
                model_file.unlink()
                print(f"✅ 删除: {model_file.name}")
            print("🧹 清理完成")
        else:
            print("取消清理")
    
    def setup_default_model(self):
        """设置默认模型"""
        default_model = "yolov8n.pt"
        model_path = self.models_dir / default_model
        
        if not model_path.exists():
            print(f"🎯 下载默认模型: {default_model}")
            if not self.download_yolo_model(default_model):
                print("❌ 默认模型下载失败")
                return False
        
        # 创建环境配置
        env_path = self.base_dir / ".env"
        
        if not env_path.exists():
            # 复制示例配置
            example_env = self.base_dir / ".env.example"
            if example_env.exists():
                import shutil
                shutil.copy(example_env, env_path)
                print("✅ 创建环境配置文件")
        
        # 更新模型路径
        if env_path.exists():
            with open(env_path, 'r') as f:
                content = f.read()
            
            # 更新YOLO模型路径
            new_content = content.replace(
                "YOLO_MODEL_PATH=./models/yolov8n.pt",
                f"YOLO_MODEL_PATH=./models/{default_model}"
            )
            
            with open(env_path, 'w') as f:
                f.write(new_content)
        
        print(f"✅ 默认模型设置完成: {default_model}")
        return True

def main():
    parser = argparse.ArgumentParser(description="AI导航系统模型下载工具")
    parser.add_argument("--list", action="store_true", help="列出所有可用模型")
    parser.add_argument("--download", type=str, help="下载指定模型")
    parser.add_argument("--download-all", action="store_true", help="下载所有模型") 
    parser.add_argument("--download-recommended", action="store_true", help="下载推荐模型")
    parser.add_argument("--verify", action="store_true", help="验证已下载的模型")
    parser.add_argument("--clean", action="store_true", help="清理所有模型")
    parser.add_argument("--setup", action="store_true", help="设置默认模型和配置")
    
    args = parser.parse_args()
    
    downloader = ModelDownloader()
    
    if args.list:
        downloader.list_available_models()
    elif args.download:
        downloader.download_yolo_model(args.download)
    elif args.download_all:
        downloader.download_all_models()
    elif args.download_recommended:
        downloader.download_recommended_models()
    elif args.verify:
        downloader.verify_models()
    elif args.clean:
        downloader.clean_models()
    elif args.setup:
        downloader.setup_default_model()
    else:
        # 默认行为：设置系统
        print("🚀 AI导航系统模型配置")
        print("=" * 50)
        
        downloader.list_available_models()
        
        print("\n选择操作:")
        print("1. 设置默认配置（推荐）")
        print("2. 下载推荐模型") 
        print("3. 下载所有模型")
        print("4. 下载指定模型")
        print("5. 验证模型")
        print("6. 退出")
        
        while True:
            choice = input("\n请选择 (1-6): ").strip()
            
            if choice == "1":
                downloader.setup_default_model()
                break
            elif choice == "2":
                downloader.download_recommended_models()
                break
            elif choice == "3":
                downloader.download_all_models()
                break
            elif choice == "4":
                downloader.list_available_models()
                model_name = input("\n输入模型名称: ").strip()
                downloader.download_yolo_model(model_name)
                break
            elif choice == "5":
                downloader.verify_models()
                break
            elif choice == "6":
                print("退出")
                break
            else:
                print("无效选择，请重试")

if __name__ == "__main__":
    main()
