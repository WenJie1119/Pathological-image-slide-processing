#!/usr/bin/env python
"""
启动病理图像切片处理工具 GUI
双击此文件或运行: python run_gui.py
"""

import sys
from pathlib import Path

# 添加src目录到路径
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from gui import main

if __name__ == "__main__":
    main()
