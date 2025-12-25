"""
图像切片预处理工具
用于将大尺寸病理图像切片成固定大小的小图像块
支持NDPI等全扫描切片格式
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from openslide import OpenSlide
import json
from tqdm import tqdm


def _setup_openslide():
    """设置 OpenSlide 库路径（Windows 需要）"""
    if sys.platform == 'win32':
        # 常见的 OpenSlide 安装路径
        possible_paths = [
            Path(os.environ.get('OPENSLIDE_PATH', '')) / 'bin',
            Path('C:/openslide/bin'),
            Path('C:/openslide-win64/bin'),
            Path('C:/Program Files/openslide/bin'),
            Path(__file__).parent.parent / 'openslide' / 'bin',
            Path(__file__).parent.parent / 'openslide-win64' / 'bin',
        ]

        for path in possible_paths:
            if path.exists():
                # 添加到 DLL 搜索路径
                dll_path = str(path.resolve())
                if hasattr(os, 'add_dll_directory'):
                    os.add_dll_directory(dll_path)
                if dll_path not in os.environ.get('PATH', ''):
                    os.environ['PATH'] = dll_path + os.pathsep + os.environ.get('PATH', '')
                return True
    return False


# 尝试设置 OpenSlide
_setup_openslide()

try:
    import openslide
    from openslide import OpenSlide
    OPENSLIDE_AVAILABLE = True
except ImportError as e:
    OPENSLIDE_AVAILABLE = False
    print("Warning: openslide-python not installed. Install with: pip install openslide-python")
except OSError as e:
    OPENSLIDE_AVAILABLE = False
    print(f"Warning: OpenSlide library not found. {e}")
    print("Please download OpenSlide from https://openslide.org/download/")
    print("Extract to C:\\openslide and add C:\\openslide\\bin to your PATH")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not installed. Install with: pip install Pillow")


class ImageSlicer:
    """图像切片器"""

    def __init__(
        self,
        tile_size: int = 512,
        overlap: int = 64,
        target_magnification: float = 20.0,
        background_threshold: float = 0.8,
        min_tissue_ratio: float = 0.1
    ):
        """
        初始化切片器

        Args:
            tile_size: 切片大小（正方形）
            overlap: 重叠像素数
            target_magnification: 目标放大倍数
            background_threshold: 背景判定阈值（灰度值归一化后）
            min_tissue_ratio: 最小组织占比，低于此值的切片将被丢弃
        """
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap  # 滑动步长
        self.target_magnification = target_magnification
        self.background_threshold = background_threshold
        self.min_tissue_ratio = min_tissue_ratio

    def get_level_for_magnification(self, slide: "OpenSlide") -> Tuple[int, float]:
        """
        根据目标放大倍数获取最接近的金字塔层级

        Args:
            slide: OpenSlide对象

        Returns:
            (level, actual_magnification): 层级索引和实际放大倍数
        """
        try:
            # 尝试读取基础放大倍数
            if 'openslide.objective-power' in slide.properties:
                base_mag = float(slide.properties['openslide.objective-power'])
            elif 'aperio.AppMag' in slide.properties:
                base_mag = float(slide.properties['aperio.AppMag'])
            else:
                print("Warning: Cannot determine base magnification, assuming 40x")
                base_mag = 40.0
        except:
            base_mag = 40.0

        # 计算每个层级的放大倍数
        best_level = 0
        best_diff = float('inf')

        for level in range(slide.level_count):
            downsample = slide.level_downsamples[level]
            level_mag = base_mag / downsample
            diff = abs(level_mag - self.target_magnification)

            if diff < best_diff:
                best_diff = diff
                best_level = level

        actual_mag = base_mag / slide.level_downsamples[best_level]
        return best_level, actual_mag

    def is_background(self, tile: np.ndarray) -> bool:
        """
        判断切片是否为背景

        Args:
            tile: 图像切片 (H, W, C)

        Returns:
            True if背景, False if组织
        """
        # 转换为灰度
        if len(tile.shape) == 3:
            gray = np.mean(tile, axis=2)
        else:
            gray = tile

        # 归一化到[0, 1]
        gray_norm = gray / 255.0

        # 计算背景像素占比（接近白色）
        background_pixels = np.sum(gray_norm > self.background_threshold)
        total_pixels = gray_norm.size
        background_ratio = background_pixels / total_pixels

        # 如果组织占比低于阈值，认为是背景
        tissue_ratio = 1 - background_ratio
        return tissue_ratio < self.min_tissue_ratio

    def generate_tiles(
        self,
        image_path: str,
        output_dir: str,
        save_metadata: bool = True
    ) -> dict:
        """
        生成图像切片

        Args:
            image_path: 输入图像路径
            output_dir: 输出目录
            save_metadata: 是否保存元数据

        Returns:
            统计信息字典
        """
        if not OPENSLIDE_AVAILABLE:
            raise ImportError("openslide-python is required. Install with: pip install openslide-python")

        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 打开切片
        print(f"Opening slide: {image_path}")
        slide = OpenSlide(image_path)

        # 获取目标层级
        level, actual_mag = self.get_level_for_magnification(slide)
        print(f"Target magnification: {self.target_magnification}x")
        print(f"Selected level: {level}, Actual magnification: {actual_mag:.2f}x")
        print(f"Level dimensions: {slide.level_dimensions[level]}")

        # 获取层级尺寸
        level_width, level_height = slide.level_dimensions[level]
        downsample = slide.level_downsamples[level]

        # 计算切片位置
        tiles_info = []
        total_tiles = 0
        saved_tiles = 0
        skipped_background = 0

        # 计算总的切片数量（完整覆盖，包括需要填充的边界）
        n_tiles_x = (level_width + self.stride - 1) // self.stride
        n_tiles_y = (level_height + self.stride - 1) // self.stride
        total_expected = n_tiles_x * n_tiles_y

        print(f"Tile size: {self.tile_size}x{self.tile_size}")
        print(f"Overlap: {self.overlap}px, Stride: {self.stride}px")
        print(f"Expected tiles: {total_expected} ({n_tiles_x}x{n_tiles_y})")
        print(f"Processing tiles with padding for complete coverage...")

        # 提取切片（完整覆盖，边界使用填充）
        row = 0
        for y in tqdm(range(0, level_height, self.stride)):
            row += 1
            col = 0
            for x in range(0, level_width, self.stride):
                col += 1
                total_tiles += 1

                # 计算实际可读取的尺寸
                actual_w = min(self.tile_size, level_width - x)
                actual_h = min(self.tile_size, level_height - y)

                # 在level 0坐标系中的位置
                x_level0 = int(x * downsample)
                y_level0 = int(y * downsample)

                # 读取切片（从level 0读取，但指定目标层级）
                tile = slide.read_region(
                    (x_level0, y_level0),
                    level,
                    (actual_w, actual_h)
                )

                # 转换为RGB（去除alpha通道）
                tile_rgb = tile.convert('RGB')

                # 如果尺寸不足，使用白色填充
                if actual_w < self.tile_size or actual_h < self.tile_size:
                    # 创建白色背景
                    padded = Image.new('RGB', (self.tile_size, self.tile_size), (255, 255, 255))
                    # 将实际图像粘贴到左上角
                    padded.paste(tile_rgb, (0, 0))
                    tile_rgb = padded

                tile_array = np.array(tile_rgb)

                # 检查是否为背景
                if self.is_background(tile_array):
                    skipped_background += 1
                    continue

                # 保存切片
                tile_filename = f"tile_{row}_{col}_{y}_{x}.png"
                tile_path = output_path / tile_filename
                tile_rgb.save(tile_path)

                # 记录元数据
                tiles_info.append({
                    'filename': tile_filename,
                    'row': row,
                    'col': col,
                    'x': x,
                    'y': y,
                    'x_level0': x_level0,
                    'y_level0': y_level0,
                    'level': level,
                    'size': self.tile_size
                })

                saved_tiles += 1

        # 保存元数据
        metadata = {
            'source_image': os.path.basename(image_path),
            'tile_size': self.tile_size,
            'overlap': self.overlap,
            'stride': self.stride,
            'target_magnification': self.target_magnification,
            'actual_magnification': actual_mag,
            'level': level,
            'downsample': downsample,
            'level_dimensions': slide.level_dimensions[level],
            'total_tiles_processed': total_tiles,
            'saved_tiles': saved_tiles,
            'skipped_background': skipped_background,
            'tiles': tiles_info
        }

        if save_metadata:
            metadata_path = output_path / 'metadata.json'
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"\nMetadata saved to: {metadata_path}")

        # 打印统计信息
        print(f"\n{'='*50}")
        print(f"Processing complete!")
        print(f"Total tiles processed: {total_tiles}")
        print(f"Saved tiles: {saved_tiles}")
        print(f"Skipped (background): {skipped_background}")
        print(f"Tissue ratio: {saved_tiles/total_tiles*100:.2f}%")
        print(f"Output directory: {output_path}")
        print(f"{'='*50}")

        slide.close()
        return metadata


def main():
    """主函数示例"""
    from pathlib import Path

    # 配置参数
    config = {
        'tile_size': 512,           # 切片大小
        'overlap': 64,              # 重叠像素
        'target_magnification': 20.0,  # 目标放大倍数
        'background_threshold': 0.8,   # 背景阈值
        'min_tissue_ratio': 0.1        # 最小组织占比
    }

    # 创建切片器
    slicer = ImageSlicer(**config)

    # 处理图像
    data_dir = Path(__file__).parent.parent / 'data'
    input_image = data_dir / "DC2200025 A4 CD34.ndpi"  # 修改为你的图像路径
    output_base = Path(__file__).parent.parent / 'output' / 'tiles'
    output_dir = output_base / input_image.stem  # 输出目录

    if input_image.exists():
        metadata = slicer.generate_tiles(
            image_path=str(input_image),
            output_dir=str(output_dir),
            save_metadata=True
        )
    else:
        print(f"Error: Image file not found: {input_image}")
        print(f"\nAvailable files in {data_dir}:")
        if data_dir.exists():
            for f in data_dir.iterdir():
                if f.suffix in ['.ndpi', '.svs', '.tiff', '.tif']:
                    print(f"  - {f.name}")
        else:
            print(f"  Data directory not found: {data_dir}")


if __name__ == "__main__":
    main()
