#!/usr/bin/env python
"""
启动切片审核工具
双击此文件或运行: python run_reviewer.py
"""

import sys
from pathlib import Path

# 添加src目录到路径
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from tile_reviewer import main

if __name__ == "__main__":
    main()
