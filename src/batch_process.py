"""
批量图像切片处理脚本
支持处理多个NDPI/SVS等全扫描切片图像
"""

import os
import sys
import yaml
from pathlib import Path

# 添加父目录到路径以便导入模块
sys.path.insert(0, str(Path(__file__).parent))

from image_slicer import ImageSlicer
from datetime import datetime


def load_config(config_path: str = "../config.yaml") -> dict:
    """加载配置文件"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    else:
        # 返回默认配置
        return {
            'tile_size': 512,
            'overlap': 64,
            'target_magnification': 20.0,
            'background_threshold': 0.8,
            'min_tissue_ratio': 0.1,
            'output_base_dir': '../output/tiles',
            'save_metadata': True,
            'input_dir': '../data',
            'file_extensions': ['.ndpi', '.svs', '.tiff', '.tif']
        }


def find_images(input_dir: str, extensions: list) -> list:
    """查找所有支持的图像文件"""
    images = []
    for ext in extensions:
        images.extend(Path(input_dir).glob(f"*{ext}"))
    return sorted(images)


def process_batch(config_path: str = "config.yaml"):
    """批量处理图像"""

    # 加载配置
    print("Loading configuration...")
    config = load_config(config_path)

    # 创建切片器
    slicer_config = {
        'tile_size': config.get('tile_size', 512),
        'overlap': config.get('overlap', 64),
        'target_magnification': config.get('target_magnification', 20.0),
        'background_threshold': config.get('background_threshold', 0.8),
        'min_tissue_ratio': config.get('min_tissue_ratio', 0.1)
    }

    slicer = ImageSlicer(**slicer_config)

    # 查找图像
    input_dir = config.get('input_dir', '.')
    extensions = config.get('file_extensions', ['.ndpi', '.svs', '.tiff', '.tif'])
    images = find_images(input_dir, extensions)

    if not images:
        print(f"No images found in {input_dir} with extensions {extensions}")
        return

    print(f"\nFound {len(images)} image(s) to process:")
    for img in images:
        print(f"  - {img.name}")

    # 创建输出基础目录
    output_base = Path(config.get('output_base_dir', 'output_tiles'))
    output_base.mkdir(parents=True, exist_ok=True)

    # 处理每个图像
    results = []
    start_time = datetime.now()

    for idx, image_path in enumerate(images, 1):
        print(f"\n{'='*60}")
        print(f"Processing [{idx}/{len(images)}]: {image_path.name}")
        print(f"{'='*60}")

        # 为每个图像创建单独的输出目录
        image_name = image_path.stem  # 文件名不含扩展名
        output_dir = output_base / image_name

        try:
            metadata = slicer.generate_tiles(
                image_path=str(image_path),
                output_dir=str(output_dir),
                save_metadata=config.get('save_metadata', True)
            )

            results.append({
                'image': image_path.name,
                'status': 'success',
                'saved_tiles': metadata['saved_tiles'],
                'output_dir': str(output_dir)
            })

        except Exception as e:
            print(f"Error processing {image_path.name}: {str(e)}")
            results.append({
                'image': image_path.name,
                'status': 'failed',
                'error': str(e)
            })

    # 打印总结
    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\n{'='*60}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total images: {len(images)}")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}")
    print(f"Total time: {duration}")
    print(f"\nResults:")

    for result in results:
        if result['status'] == 'success':
            print(f"  ✓ {result['image']}: {result['saved_tiles']} tiles -> {result['output_dir']}")
        else:
            print(f"  ✗ {result['image']}: {result['error']}")

    print(f"{'='*60}\n")


def process_single(image_path: str, config_path: str = "../config.yaml"):
    """处理单个图像"""

    # 加载配置
    config = load_config(config_path)

    # 创建切片器
    slicer_config = {
        'tile_size': config.get('tile_size', 512),
        'overlap': config.get('overlap', 64),
        'target_magnification': config.get('target_magnification', 20.0),
        'background_threshold': config.get('background_threshold', 0.8),
        'min_tissue_ratio': config.get('min_tissue_ratio', 0.1)
    }

    slicer = ImageSlicer(**slicer_config)

    # 设置输出目录
    output_base = Path(config.get('output_base_dir', '../output/tiles'))
    image_name = Path(image_path).stem
    output_dir = output_base / image_name

    # 处理图像
    metadata = slicer.generate_tiles(
        image_path=image_path,
        output_dir=str(output_dir),
        save_metadata=config.get('save_metadata', True)
    )

    return metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Batch image slicing tool')
    parser.add_argument('--config', type=str, default='../config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--image', type=str, default=None,
                        help='Process single image (optional)')
    parser.add_argument('--batch', action='store_true',
                        help='Process all images in batch mode')

    args = parser.parse_args()

    if args.image:
        # 处理单个图像
        print(f"Processing single image: {args.image}")
        process_single(args.image, args.config)
    elif args.batch:
        # 批量处理
        print("Starting batch processing...")
        process_batch(args.config)
    else:
        # 默认批量处理
        print("No specific mode selected, running batch processing...")
        process_batch(args.config)
