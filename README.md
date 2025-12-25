# 病理图像切片预处理工具

用于将大尺寸病理图像切片成固定大小的小图像块，支持 NDPI、SVS、MRXS 等全扫描切片格式。包含切片生成和切片审核两大功能模块。

## 功能特性

- **图形化界面**: 提供友好的 GUI 界面，支持单文件和批量处理
- **智能背景过滤**: 自动识别并跳过背景区域，只保留有效组织切片
- **切片审核工具**: 支持缩略图预览、框选删除、颜色筛选、样本学习等功能
- **跨平台支持**: 支持 Windows 和 Linux 系统
- **灵活参数配置**: 可调整切片大小、重叠像素、放大倍数等参数

## 项目结构

```
Pathological-image-slide-processing/
├── docs/                    # 文档目录
│   ├── README.md            # 详细使用文档
│   └── 使用指南.md          # 中文使用指南
├── src/                     # 源代码目录
│   ├── image_slicer.py      # 核心切片器
│   ├── gui.py               # 切片处理可视化界面
│   ├── tile_reviewer.py     # 切片审核工具
│   ├── visualize_tiles.py   # 可视化工具
│   ├── batch_process.py     # 批量处理脚本
│   └── quick_start.py       # 快速开始示例
├── data/                    # 数据目录（存放原始图像）
│   ├── *.ndpi
│   ├── *.svs
│   └── *.mrxs
├── output/                  # 输出目录
│   └── tiles/               # 切片输出
│       └── [image_name]/    # 每个图像一个子目录
│           ├── tile_*.png
│           └── metadata.json
├── openslide-win64/         # Windows OpenSlide 库（仅Windows需要）
├── config.yaml              # 配置文件
├── run_gui.py               # 切片处理GUI启动脚本
├── run_gui.bat              # Windows双击启动（切片处理）
├── run_reviewer.py          # 切片审核工具启动脚本
├── run_reviewer.bat         # Windows双击启动（切片审核）
├── requirements.txt         # Python 依赖
└── README.md                # 本文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

**Windows 用户**：需要 OpenSlide 库
- 项目已包含 `openslide-win64/` 目录，程序会自动检测
- 或者手动下载：https://openslide.org/download/

**Linux 用户**：参见下方 [Linux 部署指南](#linux-部署指南)

### 2. 使用可视化界面（推荐）

**切片处理工具**：
- Windows: 双击 `run_gui.bat`
- Linux/命令行: `python run_gui.py`

**切片审核工具**：
- Windows: 双击 `run_reviewer.bat`
- Linux/命令行: `python run_reviewer.py`

GUI界面功能：
- 选择单个文件或整个文件夹进行处理
- 显示待处理文件列表和状态
- 可调整参数：切片大小、重叠像素、放大倍数、背景阈值等
- 实时显示处理进度
- 支持随时停止处理

切片审核工具功能：
- 缩略图网格预览，虚拟滚动支持大量切片
- 框选、全选、反选等批量操作
- 颜色筛选：根据特定颜色占比筛选切片
- 样本学习：选择坏样本后自动学习并筛选相似切片
- 保存/加载筛选规则，便于批量应用

### 3. 放置数据

将你的病理图像文件（.ndpi, .svs 等）放在 `data/` 目录下。

### 4. 命令行方式（可选）

最简单的方式是使用快速开始脚本：

```bash
cd src
python quick_start.py
```

批量处理多个图像：

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

## Linux 部署指南

### 1. 系统要求

- Python 3.8+
- OpenSlide 库
- Tkinter（用于 GUI）

### 2. 安装 OpenSlide

**Ubuntu / Debian:**
```bash
sudo apt-get update
sudo apt-get install openslide-tools python3-openslide
```

**CentOS / RHEL / Fedora:**
```bash
# CentOS 7/8
sudo yum install epel-release
sudo yum install openslide

# Fedora
sudo dnf install openslide
```

**Arch Linux:**
```bash
sudo pacman -S openslide
```

**从源码编译（如果包管理器中没有）:**
```bash
# 安装依赖
sudo apt-get install build-essential cmake libjpeg-dev libpng-dev libtiff-dev \
    libopenjp2-7-dev libcairo2-dev libgdk-pixbuf2.0-dev libxml2-dev libsqlite3-dev

# 下载并编译
wget https://github.com/openslide/openslide/releases/download/v4.0.0/openslide-4.0.0.tar.xz
tar xf openslide-4.0.0.tar.xz
cd openslide-4.0.0
meson setup builddir
meson compile -C builddir
sudo meson install -C builddir
sudo ldconfig
```

### 3. 安装 Tkinter（GUI 支持）

**Ubuntu / Debian:**
```bash
sudo apt-get install python3-tk
```

**CentOS / RHEL:**
```bash
sudo yum install python3-tkinter
```

**Fedora:**
```bash
sudo dnf install python3-tkinter
```

### 4. 克隆项目并安装 Python 依赖

```bash
# 克隆项目
git clone https://github.com/yourusername/Pathological-image-slide-processing.git
cd Pathological-image-slide-processing

# 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 5. 验证安装

```bash
# 验证 OpenSlide
python3 -c "import openslide; print('OpenSlide version:', openslide.__library_version__)"

# 验证 Tkinter
python3 -c "import tkinter; print('Tkinter OK')"
```

### 6. 运行程序

```bash
# 切片处理 GUI
python run_gui.py

# 切片审核工具
python run_reviewer.py

# 命令行批处理
python src/batch_process.py --batch
```

### 7. 无图形界面的服务器环境

如果在没有 X11 显示的服务器上运行，只能使用命令行模式：

```bash
# 批量处理
python src/batch_process.py --batch

# 处理单个文件
python src/batch_process.py --image /path/to/image.ndpi
```

如需在服务器上使用 GUI，可以通过 X11 转发：
```bash
# 在本地终端连接服务器时启用 X11 转发
ssh -X user@server

# 然后运行 GUI
python run_gui.py
```

### 8. Docker 部署（可选）

创建 `Dockerfile`:
```dockerfile
FROM python:3.10-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    openslide-tools \
    python3-openslide \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "src/batch_process.py", "--batch"]
```

构建和运行：
```bash
docker build -t pathology-slicer .
docker run -v /path/to/data:/app/data -v /path/to/output:/app/output pathology-slicer
```

## 支持的文件格式

| 格式 | 扩展名 | 厂商 |
|------|--------|------|
| NDPI | .ndpi | Hamamatsu NanoZoomer |
| SVS | .svs | Aperio ScanScope |
| MRXS | .mrxs | 3DHISTECH Pannoramic |
| TIFF | .tiff, .tif | 通用格式 |
| 其他 | - | 所有 OpenSlide 支持的格式 |

## 更多信息

详细文档请查看 `docs/README.md` 和 `docs/使用指南.md`。

## 许可证

MIT License
