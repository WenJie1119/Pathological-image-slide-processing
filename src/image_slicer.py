"""
图像切片预处理工具
用于将大尺寸病理图像切片成固定大小的小图像块
支持NDPI等全扫描切片格式

优化版本：
- 多进程并行保存切片
- 支持 JPEG/PNG 输出格式
- Otsu 阈值自动背景检测
- 进度回调支持
"""

import os
import sys
import gc
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Callable, Dict, Any, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

if TYPE_CHECKING:
    from openslide import OpenSlide
import json
from tqdm import tqdm


def _setup_openslide():
    """设置 OpenSlide 库路径（Windows 需要）"""
    if sys.platform == 'win32':
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
                dll_path = str(path.resolve())
                if hasattr(os, 'add_dll_directory'):
                    os.add_dll_directory(dll_path)
                if dll_path not in os.environ.get('PATH', ''):
                    os.environ['PATH'] = dll_path + os.pathsep + os.environ.get('PATH', '')
                return True
    return False


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

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not installed. Install with: pip install Pillow")


class OutputFormat(Enum):
    """输出格式枚举"""
    PNG = "png"
    JPEG = "jpg"


class BackgroundMethod(Enum):
    """背景检测方法枚举"""
    THRESHOLD = "threshold"  # 简单阈值
    OTSU = "otsu"  # Otsu 自动阈值


class ImageSlicer:
    """图像切片器（优化版）"""

    def __init__(
        self,
        tile_size: int = 512,
        overlap: int = 64,
        target_magnification: float = 20.0,
        background_threshold: float = 0.8,
        min_tissue_ratio: float = 0.1,
        output_format: str = "png",
        jpeg_quality: int = 95,
        num_workers: int = 4,
        background_method: str = "threshold"
    ):
        """
        初始化切片器

        Args:
            tile_size: 切片大小（正方形）
            overlap: 重叠像素数
            target_magnification: 目标放大倍数
            background_threshold: 背景判定阈值（灰度值归一化后）
            min_tissue_ratio: 最小组织占比，低于此值的切片将被丢弃
            output_format: 输出格式 ("png" 或 "jpg")
            jpeg_quality: JPEG 质量 (1-100)
            num_workers: 并行保存的工作线程数
            background_method: 背景检测方法 ("threshold" 或 "otsu")
        """
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        self.target_magnification = target_magnification
        self.background_threshold = background_threshold
        self.min_tissue_ratio = min_tissue_ratio
        self.output_format = OutputFormat(output_format.lower())
        self.jpeg_quality = jpeg_quality
        self.num_workers = num_workers
        self.background_method = BackgroundMethod(background_method.lower())

        # Otsu 阈值缓存
        self._otsu_threshold: Optional[float] = None

    def get_level_for_magnification(self, slide: "OpenSlide") -> Tuple[int, float]:
        """根据目标放大倍数获取最接近的金字塔层级"""
        try:
            if 'openslide.objective-power' in slide.properties:
                base_mag = float(slide.properties['openslide.objective-power'])
            elif 'aperio.AppMag' in slide.properties:
                base_mag = float(slide.properties['aperio.AppMag'])
            else:
                print("Warning: Cannot determine base magnification, assuming 40x")
                base_mag = 40.0
        except:
            base_mag = 40.0

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

    def _compute_otsu_threshold(self, slide: "OpenSlide", level: int,
                                 sample_size: int = 10) -> float:
        """
        使用缩略图计算 Otsu 阈值

        Args:
            slide: OpenSlide 对象
            level: 目标层级
            sample_size: 采样缩略图的最大尺寸（百分比）

        Returns:
            Otsu 阈值（归一化到 0-1）
        """
        # 获取缩略图
        level_dims = slide.level_dimensions[level]
        thumb_size = (min(1000, level_dims[0] // 10), min(1000, level_dims[1] // 10))

        try:
            thumbnail = slide.get_thumbnail(thumb_size)
            thumb_array = np.array(thumbnail.convert('L'))

            # 计算直方图
            hist, bin_edges = np.histogram(thumb_array.flatten(), bins=256, range=(0, 256))
            hist = hist.astype(float)

            # Otsu 算法
            total = hist.sum()
            if total == 0:
                return 0.8

            current_max = 0
            threshold = 0
            sum_total = np.dot(np.arange(256), hist)
            sum_bg = 0
            weight_bg = 0

            for i in range(256):
                weight_bg += hist[i]
                if weight_bg == 0:
                    continue

                weight_fg = total - weight_bg
                if weight_fg == 0:
                    break

                sum_bg += i * hist[i]
                mean_bg = sum_bg / weight_bg
                mean_fg = (sum_total - sum_bg) / weight_fg

                variance_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

                if variance_between > current_max:
                    current_max = variance_between
                    threshold = i

            # 归一化到 0-1
            return threshold / 255.0

        except Exception as e:
            print(f"Warning: Otsu calculation failed, using default threshold. Error: {e}")
            return 0.8

    def is_background(self, tile: np.ndarray, otsu_threshold: Optional[float] = None) -> bool:
        """
        判断切片是否为背景

        Args:
            tile: 图像切片 (H, W, C)
            otsu_threshold: Otsu 阈值（可选，用于 Otsu 方法）

        Returns:
            True if 背景, False if 组织
        """
        # 转换为灰度
        if len(tile.shape) == 3:
            # 使用加权灰度转换（更准确）
            gray = 0.299 * tile[:, :, 0] + 0.587 * tile[:, :, 1] + 0.114 * tile[:, :, 2]
        else:
            gray = tile

        # 归一化到 [0, 1]
        gray_norm = gray / 255.0

        # 选择阈值
        if self.background_method == BackgroundMethod.OTSU and otsu_threshold is not None:
            threshold = otsu_threshold
        else:
            threshold = self.background_threshold

        # 计算背景像素占比
        background_pixels = np.sum(gray_norm > threshold)
        total_pixels = gray_norm.size
        background_ratio = background_pixels / total_pixels

        tissue_ratio = 1 - background_ratio
        return tissue_ratio < self.min_tissue_ratio

    def _save_tile(self, tile_rgb: Image.Image, tile_path: Path) -> bool:
        """保存单个切片"""
        try:
            if self.output_format == OutputFormat.JPEG:
                tile_rgb.save(tile_path, 'JPEG', quality=self.jpeg_quality, optimize=True)
            else:
                tile_rgb.save(tile_path, 'PNG', optimize=True)
            return True
        except Exception as e:
            print(f"Error saving tile {tile_path}: {e}")
            return False

    def generate_tiles(
        self,
        image_path: str,
        output_dir: str,
        save_metadata: bool = True,
        progress_callback: Optional[Callable[[int, int, int, int], None]] = None,
        stop_check: Optional[Callable[[], bool]] = None
    ) -> dict:
        """
        生成图像切片

        Args:
            image_path: 输入图像路径
            output_dir: 输出目录
            save_metadata: 是否保存元数据
            progress_callback: 进度回调函数 (current, total, saved, skipped)
            stop_check: 停止检查函数，返回 True 时停止处理

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

        # 如果使用 Otsu 方法，计算阈值
        otsu_threshold = None
        if self.background_method == BackgroundMethod.OTSU:
            print("Computing Otsu threshold...")
            otsu_threshold = self._compute_otsu_threshold(slide, level)
            print(f"Otsu threshold: {otsu_threshold:.3f}")

        # 获取层级尺寸
        level_width, level_height = slide.level_dimensions[level]
        downsample = slide.level_downsamples[level]

        # 计算切片位置
        tiles_info: List[Dict[str, Any]] = []
        total_tiles = 0
        saved_tiles = 0
        skipped_background = 0

        # 计算总的切片数量
        n_tiles_x = (level_width + self.stride - 1) // self.stride
        n_tiles_y = (level_height + self.stride - 1) // self.stride
        total_expected = n_tiles_x * n_tiles_y

        # 文件扩展名
        ext = ".jpg" if self.output_format == OutputFormat.JPEG else ".png"

        print(f"Tile size: {self.tile_size}x{self.tile_size}")
        print(f"Overlap: {self.overlap}px, Stride: {self.stride}px")
        print(f"Output format: {self.output_format.value.upper()}")
        print(f"Expected tiles: {total_expected} ({n_tiles_x}x{n_tiles_y})")
        print(f"Using {self.num_workers} workers for parallel saving")
        print(f"Processing tiles...")

        # 使用线程池并行保存
        pending_saves = []

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            row = 0
            for y in tqdm(range(0, level_height, self.stride), desc="Processing rows"):
                # 检查是否需要停止
                if stop_check and stop_check():
                    print("\nProcessing stopped by user")
                    executor.shutdown(wait=False)
                    slide.close()
                    return {
                        'source_image': os.path.basename(image_path),
                        'status': 'stopped',
                        'total_tiles_processed': total_tiles,
                        'saved_tiles': saved_tiles,
                        'skipped_background': skipped_background
                    }

                row += 1
                col = 0

                for x in range(0, level_width, self.stride):
                    col += 1
                    total_tiles += 1

                    # 进度回调
                    if progress_callback:
                        progress_callback(total_tiles, total_expected, saved_tiles, skipped_background)

                    # 计算实际可读取的尺寸
                    actual_w = min(self.tile_size, level_width - x)
                    actual_h = min(self.tile_size, level_height - y)

                    # 在 level 0 坐标系中的位置
                    x_level0 = int(x * downsample)
                    y_level0 = int(y * downsample)

                    # 读取切片
                    tile = slide.read_region(
                        (x_level0, y_level0),
                        level,
                        (actual_w, actual_h)
                    )

                    # 转换为 RGB
                    tile_rgb = tile.convert('RGB')

                    # 填充边界
                    if actual_w < self.tile_size or actual_h < self.tile_size:
                        padded = Image.new('RGB', (self.tile_size, self.tile_size), (255, 255, 255))
                        padded.paste(tile_rgb, (0, 0))
                        tile_rgb = padded

                    tile_array = np.array(tile_rgb)

                    # 检查是否为背景
                    if self.is_background(tile_array, otsu_threshold):
                        skipped_background += 1
                        continue

                    # 保存切片（异步）
                    tile_filename = f"tile_{row}_{col}_{y}_{x}{ext}"
                    tile_path = output_path / tile_filename

                    # 提交保存任务
                    future = executor.submit(self._save_tile, tile_rgb.copy(), tile_path)
                    pending_saves.append((future, {
                        'filename': tile_filename,
                        'row': row,
                        'col': col,
                        'x': x,
                        'y': y,
                        'x_level0': x_level0,
                        'y_level0': y_level0,
                        'level': level,
                        'size': self.tile_size
                    }))

                    saved_tiles += 1

                # 定期清理已完成的任务和垃圾回收
                if row % 10 == 0:
                    # 收集已完成的保存任务
                    still_pending = []
                    for future, info in pending_saves:
                        if future.done():
                            if future.result():
                                tiles_info.append(info)
                        else:
                            still_pending.append((future, info))
                    pending_saves = still_pending
                    gc.collect()

            # 等待所有保存任务完成
            for future, info in pending_saves:
                if future.result():
                    tiles_info.append(info)

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
            'output_format': self.output_format.value,
            'background_method': self.background_method.value,
            'background_threshold': otsu_threshold if otsu_threshold else self.background_threshold,
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
        if total_tiles > 0:
            print(f"Tissue ratio: {saved_tiles/total_tiles*100:.2f}%")
        print(f"Output directory: {output_path}")
        print(f"{'='*50}")

        slide.close()
        return metadata


def main():
    """主函数示例"""
    from pathlib import Path

    config = {
        'tile_size': 512,
        'overlap': 64,
        'target_magnification': 20.0,
        'background_threshold': 0.8,
        'min_tissue_ratio': 0.1,
        'output_format': 'png',
        'num_workers': 4,
        'background_method': 'otsu'
    }

    slicer = ImageSlicer(**config)

    data_dir = Path(__file__).parent.parent / 'data'
    input_image = data_dir / "DC2200025 A4 CD34.ndpi"
    output_base = Path(__file__).parent.parent / 'output' / 'tiles'
    output_dir = output_base / input_image.stem

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
                if f.suffix in ['.ndpi', '.svs', '.tiff', '.tif', '.mrxs']:
                    print(f"  - {f.name}")
        else:
            print(f"  Data directory not found: {data_dir}")


if __name__ == "__main__":
    main()
