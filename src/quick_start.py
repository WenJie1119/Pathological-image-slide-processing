"""
快速开始示例
演示基本用法
"""

import sys
from pathlib import Path
from datetime import datetime

# 添加父目录到路径以便导入模块
sys.path.insert(0, str(Path(__file__).parent))

from image_slicer import ImageSlicer
import os


def quick_start():
    """快速开始示例 - 批量处理多种配置"""

    print("=" * 60)
    print("图像切片预处理 - 批量配置处理")
    print("=" * 60)

    # 步骤1: 定义所有配置组合
    print("\n[步骤1] 配置参数...")
    magnifications = [20.0, 40.0]  # 20倍和40倍
    tile_sizes = [512, 1024, 2048, 4096]  # 4种切片大小

    # 基础配置
    base_config = {
        'overlap': 0,                 # 重叠像素
        'background_threshold': 0.8,   # 背景阈值
        'min_tissue_ratio': 0.1        # 最小组织占比10%
    }

    print(f"  - 倍数配置: {magnifications}")
    print(f"  - 切片大小: {tile_sizes}")
    print(f"  - 总共配置数: {len(magnifications) * len(tile_sizes)}")

    # 步骤2: 查找可用图像
    print("\n[步骤2] 查找可用图像...")
    data_dir = Path(__file__).parent.parent / 'data'
    available_images = []
    for ext in ['.ndpi', '.svs', '.tiff', '.tif']:
        for f in data_dir.glob(f'*{ext}'):
            available_images.append(str(f))

    if not available_images:
        print(f"  [错误] 未找到可用的图像文件在 {data_dir}")
        print("\n支持的格式: .ndpi, .svs, .tiff, .tif")
        print(f"请将图像文件放在 {data_dir} 目录下")
        return

    print(f"  [OK] 找到 {len(available_images)} 个图像:")
    for img in available_images:
        file_size = os.path.getsize(img) / (1024**3)  # GB
        print(f"    - {Path(img).name} ({file_size:.2f} GB)")

    # 步骤3: 自动选择第一张图像
    print("\n[步骤3] 自动选择图像...")
    selected_image = available_images[0]
    print(f"  已选择: {Path(selected_image).name}")

    # 步骤4: 批量处理所有配置
    print("\n[步骤4] 开始批量处理...")
    print("=" * 60)

    output_base = Path(__file__).parent.parent / 'output' / 'tiles'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []

    total_configs = len(magnifications) * len(tile_sizes)
    current_config = 0

    for mag in magnifications:
        for tile_size in tile_sizes:
            current_config += 1
            print(f"\n{'='*60}")
            print(f"配置 [{current_config}/{total_configs}]: {int(mag)}倍 × {tile_size}px")
            print(f"{'='*60}")

            # 创建当前配置
            config = {
                **base_config,
                'tile_size': tile_size,
                'target_magnification': mag
            }

            print(f"  - 切片大小: {tile_size}x{tile_size}")
            print(f"  - 目标倍数: {mag}x")
            print(f"  - 重叠区域: {config['overlap']}px")

            # 创建切片器
            print("  - 创建切片器...")
            slicer = ImageSlicer(**config)

            # 设置输出目录
            param_info = f"{int(mag)}x_{tile_size}"
            output_dir = output_base / Path(selected_image).stem / f"{param_info}_{timestamp}"
            print(f"  - 输出目录: {output_dir}")

            # 开始处理
            print("  - 开始切片处理...")
            try:
                metadata = slicer.generate_tiles(
                    image_path=selected_image,
                    output_dir=str(output_dir),
                    save_metadata=True
                )

                print(f"  - 保存的切片: {metadata['saved_tiles']}")
                print(f"  - 跳过的背景: {metadata['skipped_background']}")
                print(f"  - 组织占比: {metadata['saved_tiles']/metadata['total_tiles_processed']*100:.2f}%")

                # 自动生成可视化
                print("  - 生成可视化图...")
                try:
                    from visualize_tiles import visualize_tiles
                    visualize_tiles(f"{output_dir}/metadata.json", max_tiles_preview=16)
                    print("  - [OK] 可视化图已保存")
                except ImportError:
                    print("  - [警告] 未安装matplotlib，跳过可视化")
                except Exception as e:
                    print(f"  - [警告] 可视化失败: {e}")

                results.append({
                    'magnification': mag,
                    'tile_size': tile_size,
                    'output_dir': str(output_dir),
                    'saved_tiles': metadata['saved_tiles'],
                    'status': 'success'
                })

            except Exception as e:
                print(f"  - [错误] 处理失败: {e}")
                results.append({
                    'magnification': mag,
                    'tile_size': tile_size,
                    'status': 'failed',
                    'error': str(e)
                })

    # 步骤5: 显示总结
    print("\n" + "=" * 60)
    print("处理总结")
    print("=" * 60)
    print(f"图像: {Path(selected_image).name}")
    print(f"总配置数: {total_configs}")
    print(f"成功: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"失败: {sum(1 for r in results if r['status'] == 'failed')}")

    print("\n详细结果:")
    for r in results:
        if r['status'] == 'success':
            print(f"  [OK] {int(r['magnification'])}倍 × {r['tile_size']}px - {r['saved_tiles']} 切片")
            print(f"       输出: {r['output_dir']}")
        else:
            print(f"  [失败] {int(r['magnification'])}倍 × {r['tile_size']}px - {r['error']}")

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    quick_start()
