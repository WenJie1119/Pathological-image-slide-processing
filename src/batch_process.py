"""
批量图像切片处理脚本
支持处理多个NDPI/SVS等全扫描切片图像

优化版本：
- 支持新参数（输出格式、背景检测方法等）
- 多文件并行处理（可选）
"""

import os
import sys
import yaml
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

# 添加父目录到路径以便导入模块
sys.path.insert(0, str(Path(__file__).parent))

from image_slicer import ImageSlicer


def load_config(config_path: str = "../config.yaml") -> dict:
    """加载配置文件"""
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = Path(__file__).parent / config_path

    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
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
            'output_format': 'png',
            'jpeg_quality': 95,
            'num_workers': 4,
            'background_method': 'otsu',
            'output_base_dir': '../output/tiles',
            'save_metadata': True,
            'input_dir': '../data',
            'file_extensions': ['.ndpi', '.svs', '.tiff', '.tif', '.mrxs']
        }


def find_images(input_dir: str, extensions: list) -> list:
    """查找所有支持的图像文件"""
    input_path = Path(input_dir)
    if not input_path.is_absolute():
        input_path = Path(__file__).parent / input_dir

    images = []
    for ext in extensions:
        images.extend(input_path.glob(f"*{ext}"))
        images.extend(input_path.glob(f"*{ext.upper()}"))
    return sorted(set(images))


def process_single_image(args: tuple) -> Dict[str, Any]:
    """处理单个图像（用于多进程）"""
    image_path, output_dir, slicer_config, save_metadata = args

    try:
        slicer = ImageSlicer(**slicer_config)
        metadata = slicer.generate_tiles(
            image_path=str(image_path),
            output_dir=str(output_dir),
            save_metadata=save_metadata
        )

        return {
            'image': image_path.name,
            'status': 'success',
            'saved_tiles': metadata['saved_tiles'],
            'total_tiles': metadata['total_tiles_processed'],
            'skipped': metadata['skipped_background'],
            'output_dir': str(output_dir)
        }

    except Exception as e:
        return {
            'image': image_path.name,
            'status': 'failed',
            'error': str(e)
        }


def process_batch(config_path: str = "../config.yaml", parallel_files: int = 1):
    """
    批量处理图像

    Args:
        config_path: 配置文件路径
        parallel_files: 并行处理的文件数（1表示顺序处理）
    """
    # 加载配置
    print("Loading configuration...")
    config = load_config(config_path)

    # 切片器配置
    slicer_config = {
        'tile_size': config.get('tile_size', 512),
        'overlap': config.get('overlap', 64),
        'target_magnification': config.get('target_magnification', 20.0),
        'background_threshold': config.get('background_threshold', 0.8),
        'min_tissue_ratio': config.get('min_tissue_ratio', 0.1),
        'output_format': config.get('output_format', 'png'),
        'jpeg_quality': config.get('jpeg_quality', 95),
        'num_workers': config.get('num_workers', 4),
        'background_method': config.get('background_method', 'otsu')
    }

    # 查找图像
    input_dir = config.get('input_dir', '../data')
    extensions = config.get('file_extensions', ['.ndpi', '.svs', '.tiff', '.tif', '.mrxs'])
    images = find_images(input_dir, extensions)

    if not images:
        print(f"No images found in {input_dir} with extensions {extensions}")
        return

    print(f"\nFound {len(images)} image(s) to process:")
    for img in images:
        print(f"  - {img.name}")

    # 创建输出基础目录
    output_base = Path(config.get('output_base_dir', '../output/tiles'))
    if not output_base.is_absolute():
        output_base = Path(__file__).parent / output_base
    output_base.mkdir(parents=True, exist_ok=True)

    save_metadata = config.get('save_metadata', True)

    # 打印配置信息
    print(f"\nConfiguration:")
    print(f"  Tile size: {slicer_config['tile_size']}x{slicer_config['tile_size']}")
    print(f"  Output format: {slicer_config['output_format'].upper()}")
    print(f"  Background method: {slicer_config['background_method']}")
    print(f"  Parallel workers per file: {slicer_config['num_workers']}")
    print(f"  Parallel files: {parallel_files}")

    # 处理每个图像
    results = []
    start_time = datetime.now()

    if parallel_files > 1 and len(images) > 1:
        # 多进程处理多个文件
        print(f"\nProcessing {len(images)} files in parallel (max {parallel_files} concurrent)...")

        # 准备任务参数
        tasks = []
        for image_path in images:
            output_dir = output_base / image_path.stem
            tasks.append((image_path, output_dir, slicer_config, save_metadata))

        with ProcessPoolExecutor(max_workers=parallel_files) as executor:
            futures = {executor.submit(process_single_image, task): task[0] for task in tasks}

            for future in as_completed(futures):
                image_path = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    if result['status'] == 'success':
                        print(f"  ✓ {result['image']}: {result['saved_tiles']} tiles saved")
                    else:
                        print(f"  ✗ {result['image']}: {result['error']}")
                except Exception as e:
                    results.append({
                        'image': image_path.name,
                        'status': 'failed',
                        'error': str(e)
                    })
                    print(f"  ✗ {image_path.name}: {e}")
    else:
        # 顺序处理
        slicer = ImageSlicer(**slicer_config)

        for idx, image_path in enumerate(images, 1):
            print(f"\n{'='*60}")
            print(f"Processing [{idx}/{len(images)}]: {image_path.name}")
            print(f"{'='*60}")

            output_dir = output_base / image_path.stem

            try:
                metadata = slicer.generate_tiles(
                    image_path=str(image_path),
                    output_dir=str(output_dir),
                    save_metadata=save_metadata
                )

                results.append({
                    'image': image_path.name,
                    'status': 'success',
                    'saved_tiles': metadata['saved_tiles'],
                    'total_tiles': metadata['total_tiles_processed'],
                    'skipped': metadata['skipped_background'],
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

    total_tiles = sum(r.get('saved_tiles', 0) for r in results if r['status'] == 'success')
    print(f"Total tiles saved: {total_tiles}")

    print(f"\nResults:")
    for result in results:
        if result['status'] == 'success':
            print(f"  ✓ {result['image']}: {result['saved_tiles']} tiles -> {result['output_dir']}")
        else:
            print(f"  ✗ {result['image']}: {result['error']}")

    print(f"{'='*60}\n")


def process_single(image_path: str, config_path: str = "../config.yaml"):
    """处理单个图像"""
    config = load_config(config_path)

    slicer_config = {
        'tile_size': config.get('tile_size', 512),
        'overlap': config.get('overlap', 64),
        'target_magnification': config.get('target_magnification', 20.0),
        'background_threshold': config.get('background_threshold', 0.8),
        'min_tissue_ratio': config.get('min_tissue_ratio', 0.1),
        'output_format': config.get('output_format', 'png'),
        'jpeg_quality': config.get('jpeg_quality', 95),
        'num_workers': config.get('num_workers', 4),
        'background_method': config.get('background_method', 'otsu')
    }

    slicer = ImageSlicer(**slicer_config)

    output_base = Path(config.get('output_base_dir', '../output/tiles'))
    if not output_base.is_absolute():
        output_base = Path(__file__).parent / output_base

    image_name = Path(image_path).stem
    output_dir = output_base / image_name

    metadata = slicer.generate_tiles(
        image_path=image_path,
        output_dir=str(output_dir),
        save_metadata=config.get('save_metadata', True)
    )

    return metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Batch image slicing tool (optimized)')
    parser.add_argument('--config', type=str, default='../config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--image', type=str, default=None,
                        help='Process single image (optional)')
    parser.add_argument('--batch', action='store_true',
                        help='Process all images in batch mode')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of files to process in parallel (default: 1)')

    args = parser.parse_args()

    if args.image:
        print(f"Processing single image: {args.image}")
        process_single(args.image, args.config)
    elif args.batch:
        print("Starting batch processing...")
        process_batch(args.config, parallel_files=args.parallel)
    else:
        print("No specific mode selected, running batch processing...")
        process_batch(args.config, parallel_files=args.parallel)
