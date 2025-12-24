# 图像切片预处理工具

用于将大尺寸病理图像切片成固定大小的小图像块，支持 NDPI、SVS 等全扫描切片格式。

## 项目结构

```
DP/
├── docs/              # 文档目录
│   ├── README.md      # 详细使用文档
│   └── 使用指南.md    # 中文使用指南
├── src/               # 源代码目录
│   ├── image_slicer.py      # 核心切片器
│   ├── visualize_tiles.py   # 可视化工具
│   ├── batch_process.py     # 批量处理脚本
│   └── quick_start.py       # 快速开始示例
├── data/              # 数据目录（存放原始图像）
│   ├── *.ndpi
│   └── *.svs
├── output/            # 输出目录
│   └── tiles/         # 切片输出
│       ├── [image_name]/    # 每个图像一个子目录
│       │   ├── tile_*.png
│       │   ├── metadata.json
│       │   └── visualizations/  # 可视化图输出
├── config.yaml        # 配置文件
├── requirements.txt   # Python 依赖
└── README.md          # 本文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 放置数据

将你的病理图像文件（.ndpi, .svs 等）放在 `data/` 目录下。

### 3. 运行示例

最简单的方式是使用快速开始脚本：

```bash
cd src
python quick_start.py
```

### 4. 批量处理

如果需要批量处理多个图像：

```bash
cd src
python batch_process.py --batch
```

或处理单个图像：

```bash
cd src
python batch_process.py --image ../data/your_image.ndpi
```

## 配置说明

编辑根目录的 `config.yaml` 文件来调整参数：

- `tile_size`: 切片大小（默认 512x512）
- `overlap`: 重叠像素数（默认 64）
- `target_magnification`: 目标放大倍数（默认 20x）
- `background_threshold`: 背景阈值（默认 0.8）
- `min_tissue_ratio`: 最小组织占比（默认 0.1）

## 可视化

切片分布可视化图会自动保存到输出目录的 `visualizations/` 子目录中，不再弹出显示窗口。

可视化包括：
- 切片位置分布图
- 切片密度热图
- 样本切片预览
- 统计信息总览

## 更多信息

详细文档请查看 `docs/README.md` 和 `docs/使用指南.md`。
