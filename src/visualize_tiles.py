"""
切片可视化工具
用于查看切片位置分布和统计信息
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np


def visualize_tiles(metadata_path: str, max_tiles_preview: int = 20, output_dir: str = None):
    """
    可视化切片位置和分布

    Args:
        metadata_path: metadata.json文件路径
        max_tiles_preview: 最大预览切片数量
        output_dir: 输出目录，如果为None则使用metadata所在目录
    """
    # 读取元数据
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # 设置输出目录
    if output_dir is None:
        output_dir = str(Path(metadata_path).parent / 'visualizations')
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 提取信息
    level_width, level_height = metadata['level_dimensions']
    tile_size = metadata['tile_size']
    stride = metadata['stride']
    tiles = metadata['tiles']

    print(f"{'='*60}")
    print(f"Slide: {metadata['source_image']}")
    print(f"{'='*60}")
    print(f"Level dimensions: {level_width} x {level_height}")
    print(f"Magnification: {metadata['actual_magnification']:.2f}x (target: {metadata['target_magnification']}x)")
    print(f"Tile size: {tile_size}x{tile_size}")
    print(f"Overlap: {metadata['overlap']}px, Stride: {stride}px")
    print(f"Total processed: {metadata['total_tiles_processed']}")
    print(f"Saved tiles: {metadata['saved_tiles']}")
    print(f"Skipped (background): {metadata['skipped_background']}")
    print(f"Tissue ratio: {metadata['saved_tiles']/metadata['total_tiles_processed']*100:.2f}%")
    print(f"{'='*60}\n")

    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Tile Analysis: {metadata['source_image']}", fontsize=16)

    # 1. 切片位置分布图
    ax1 = axes[0, 0]
    ax1.set_title('Tile Position Distribution')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    ax1.set_xlim(0, level_width)
    ax1.set_ylim(level_height, 0)  # Y轴反转

    # 绘制所有切片位置
    for tile in tiles:
        rect = patches.Rectangle(
            (tile['x'], tile['y']),
            tile_size, tile_size,
            linewidth=0.5,
            edgecolor='blue',
            facecolor='lightblue',
            alpha=0.5
        )
        ax1.add_patch(rect)

    # 2. 切片密度热图
    ax2 = axes[0, 1]
    ax2.set_title('Tile Density Heatmap')

    # 创建密度网格
    grid_size = 50
    x_bins = np.linspace(0, level_width, grid_size)
    y_bins = np.linspace(0, level_height, grid_size)

    tile_x = [t['x'] for t in tiles]
    tile_y = [t['y'] for t in tiles]

    heatmap, xedges, yedges = np.histogram2d(tile_x, tile_y, bins=[x_bins, y_bins])
    extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]]

    im = ax2.imshow(heatmap.T, extent=extent, origin='upper', cmap='hot', aspect='auto')
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    plt.colorbar(im, ax=ax2, label='Tile count')

    # 3. 切片预览
    ax3 = axes[1, 0]
    ax3.set_title(f'Sample Tiles (first {min(max_tiles_preview, len(tiles))})')
    ax3.axis('off')

    # 加载并显示部分切片
    tile_dir = Path(metadata_path).parent
    sample_tiles = tiles[:max_tiles_preview]

    n_rows = int(np.sqrt(len(sample_tiles)))
    n_cols = int(np.ceil(len(sample_tiles) / n_rows))

    preview_fig, preview_axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))
    preview_fig.suptitle(f'Sample Tiles from {metadata["source_image"]}', fontsize=14)

    if n_rows == 1 and n_cols == 1:
        preview_axes = [[preview_axes]]
    elif n_rows == 1:
        preview_axes = [preview_axes]
    elif n_cols == 1:
        preview_axes = [[ax] for ax in preview_axes]

    for idx, tile in enumerate(sample_tiles):
        row = idx // n_cols
        col = idx % n_cols

        tile_path = tile_dir / tile['filename']
        if tile_path.exists():
            img = Image.open(tile_path)
            preview_axes[row][col].imshow(img)
            preview_axes[row][col].set_title(f"({tile['y']}, {tile['x']})", fontsize=8)
            preview_axes[row][col].axis('off')

    # 隐藏空白子图
    for idx in range(len(sample_tiles), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        preview_axes[row][col].axis('off')

    # 4. 统计信息
    ax4 = axes[1, 1]
    ax4.set_title('Statistics')
    ax4.axis('off')

    stats_text = f"""
    Source Image: {metadata['source_image']}

    Image Properties:
    - Level: {metadata['level']}
    - Dimensions: {level_width} x {level_height}
    - Downsample: {metadata['downsample']:.2f}
    - Magnification: {metadata['actual_magnification']:.2f}x

    Tiling Parameters:
    - Tile size: {tile_size} x {tile_size}
    - Overlap: {metadata['overlap']} px
    - Stride: {stride} px

    Processing Results:
    - Total tiles processed: {metadata['total_tiles_processed']}
    - Saved tiles: {metadata['saved_tiles']}
    - Skipped (background): {metadata['skipped_background']}
    - Tissue ratio: {metadata['saved_tiles']/metadata['total_tiles_processed']*100:.2f}%

    Storage:
    - Average tile size: ~{estimate_avg_file_size(tile_dir, sample_tiles[:10])} KB
    - Total estimated size: ~{estimate_total_size(tile_dir, metadata['saved_tiles'], sample_tiles[:10])} MB
    """

    ax4.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 保存主分析图
    analysis_output = Path(output_dir) / f"{Path(metadata['source_image']).stem}_analysis.png"
    plt.tight_layout()
    plt.savefig(analysis_output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Analysis visualization saved to: {analysis_output}")

    # 保存切片预览图
    preview_output = Path(output_dir) / f"{Path(metadata['source_image']).stem}_preview.png"
    preview_fig.tight_layout()
    preview_fig.savefig(preview_output, dpi=150, bbox_inches='tight')
    plt.close(preview_fig)
    print(f"Preview visualization saved to: {preview_output}")


def estimate_avg_file_size(tile_dir: Path, sample_tiles: list) -> float:
    """估算平均文件大小（KB）"""
    sizes = []
    for tile in sample_tiles:
        tile_path = tile_dir / tile['filename']
        if tile_path.exists():
            sizes.append(os.path.getsize(tile_path) / 1024)  # KB
    return np.mean(sizes) if sizes else 0


def estimate_total_size(tile_dir: Path, total_tiles: int, sample_tiles: list) -> float:
    """估算总文件大小（MB）"""
    avg_size_kb = estimate_avg_file_size(tile_dir, sample_tiles)
    total_size_mb = (avg_size_kb * total_tiles) / 1024  # MB
    return total_size_mb


def compare_multiple_slides(metadata_paths: list, output_dir: str = 'visualizations'):
    """
    比较多个切片的统计信息

    Args:
        metadata_paths: metadata.json文件路径列表
        output_dir: 输出目录
    """
    data = []

    for path in metadata_paths:
        with open(path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            data.append({
                'name': metadata['source_image'],
                'saved_tiles': metadata['saved_tiles'],
                'total_tiles': metadata['total_tiles_processed'],
                'tissue_ratio': metadata['saved_tiles'] / metadata['total_tiles_processed'] * 100
            })

    # 创建对比图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    names = [d['name'] for d in data]
    saved = [d['saved_tiles'] for d in data]
    total = [d['total_tiles'] for d in data]
    ratio = [d['tissue_ratio'] for d in data]

    # 1. 保存的切片数量
    axes[0].bar(names, saved, color='steelblue')
    axes[0].set_title('Saved Tiles')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)

    # 2. 总处理切片数量
    axes[1].bar(names, total, color='coral')
    axes[1].set_title('Total Processed Tiles')
    axes[1].set_ylabel('Count')
    axes[1].tick_params(axis='x', rotation=45)

    # 3. 组织占比
    axes[2].bar(names, ratio, color='seagreen')
    axes[2].set_title('Tissue Ratio')
    axes[2].set_ylabel('Percentage (%)')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].axhline(y=50, color='r', linestyle='--', label='50% threshold')
    axes[2].legend()

    # 保存对比图
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    comparison_output = Path(output_dir) / 'slides_comparison.png'
    plt.tight_layout()
    plt.savefig(comparison_output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Comparison visualization saved to: {comparison_output}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Visualize tile distribution and statistics')
    parser.add_argument('--metadata', type=str, required=True,
                        help='Path to metadata.json file')
    parser.add_argument('--max-preview', type=int, default=20,
                        help='Maximum number of tiles to preview')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for visualizations (default: metadata dir/visualizations)')
    parser.add_argument('--compare', type=str, nargs='+',
                        help='Compare multiple metadata files')

    args = parser.parse_args()

    if args.compare:
        output_dir = args.output_dir if args.output_dir else 'visualizations'
        compare_multiple_slides(args.compare, output_dir)
    else:
        visualize_tiles(args.metadata, args.max_preview, args.output_dir)
