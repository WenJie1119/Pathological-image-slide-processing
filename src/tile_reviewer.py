"""
切片审核工具 - 缩略图网格预览（优化版）
采用虚拟滚动 + 多线程加载 + 缩略图缓存 + 颜色直方图筛选
"""

import os
import sys
import json
import hashlib
import threading
from pathlib import Path
from typing import List, Set, Optional, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np


class ImageAnalyzer:
    """图像特征分析器"""

    @staticmethod
    def get_color_ratio(image_path: Path, target_colors: List[Tuple[Tuple[int,int,int], int]],
                        thumbnail_size: int = 256) -> Dict[int, float]:
        """
        计算图像中特定颜色的占比

        Args:
            image_path: 图像路径
            target_colors: [(RGB颜色, 容差), ...] 列表
            thumbnail_size: 缩略图大小（加速计算）

        Returns:
            {颜色索引: 占比} 字典
        """
        img = None
        try:
            img = Image.open(image_path)
            img.thumbnail((thumbnail_size, thumbnail_size))
            img = img.convert('RGB')
            arr = np.array(img, dtype=np.float32)

            total_pixels = arr.shape[0] * arr.shape[1]
            results = {}

            for idx, (color, tolerance) in enumerate(target_colors):
                # 计算每个像素与目标颜色的距离
                diff = arr - np.array(color, dtype=np.float32)
                distance = np.sqrt(np.sum(diff ** 2, axis=2))

                # 统计在容差范围内的像素比例
                match_count = np.sum(distance <= tolerance)
                results[idx] = match_count / total_pixels

            return results
        except Exception as e:
            return {}
        finally:
            if img is not None:
                img.close()

    @staticmethod
    def analyze(image_path: Path) -> Dict:
        """分析图像特征"""
        try:
            img = Image.open(image_path).convert('RGB')
            arr = np.array(img)

            # 1. 背景占比（白色/浅色区域）
            gray = np.mean(arr, axis=2)
            background_ratio = np.sum(gray > 200) / gray.size

            # 2. 组织占比
            tissue_ratio = 1 - background_ratio

            # 3. 模糊度检测（拉普拉斯方差）
            gray_img = img.convert('L')
            gray_arr = np.array(gray_img, dtype=np.float64)
            laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
            from scipy import ndimage
            lap = ndimage.convolve(gray_arr, laplacian)
            blur_score = lap.var()

            # 4. 颜色多样性（标准差）
            color_std = np.std(arr)

            # 5. 平均亮度
            brightness = np.mean(gray)

            # 6. 红色通道占比（组织通常偏红/粉）
            red_ratio = np.mean(arr[:, :, 0]) / (np.mean(arr) + 1e-6)

            return {
                'background_ratio': background_ratio,
                'tissue_ratio': tissue_ratio,
                'blur_score': blur_score,
                'color_std': color_std,
                'brightness': brightness,
                'red_ratio': red_ratio
            }
        except Exception as e:
            return None

    @staticmethod
    def analyze_simple(image_path: Path) -> Dict:
        """简化版分析（不需要scipy），包含直方图特征"""
        img = None
        try:
            img = Image.open(image_path)
            # 缩小图像以节省内存
            img.thumbnail((256, 256))
            img = img.convert('RGB')
            arr = np.array(img)

            # 1. 背景占比
            gray = np.mean(arr, axis=2)
            background_ratio = np.sum(gray > 200) / gray.size

            # 2. 组织占比
            tissue_ratio = 1 - background_ratio

            # 3. 颜色多样性
            color_std = np.std(arr)

            # 4. 平均亮度
            brightness = np.mean(gray)

            # 5. 简单模糊度（梯度方差）
            gx = np.diff(gray, axis=1)
            gy = np.diff(gray, axis=0)
            blur_score = np.var(gx) + np.var(gy)

            # 6. 直方图特征
            gray_flat = gray.flatten().astype(np.uint8)
            hist, _ = np.histogram(gray_flat, bins=256, range=(0, 256))
            hist = hist.astype(np.float64)
            hist_sum = hist.sum()
            if hist_sum > 0:
                hist_norm = hist / hist_sum  # 归一化直方图

                # 6.1 直方图峰值位置（主要亮度）
                hist_peak = np.argmax(hist)

                # 6.2 直方图重心（加权平均亮度）
                bins = np.arange(256)
                hist_centroid = np.sum(bins * hist_norm)

                # 6.3 直方图分布宽度（标准差，反映对比度）
                hist_spread = np.sqrt(np.sum(((bins - hist_centroid) ** 2) * hist_norm))

                # 6.4 直方图熵（复杂度/信息量）
                hist_nonzero = hist_norm[hist_norm > 0]
                hist_entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))

                # 6.5 高亮区域占比（亮度>200的像素比例）
                bright_ratio = np.sum(hist[200:]) / hist_sum

                # 6.6 暗区域占比（亮度<50的像素比例）
                dark_ratio = np.sum(hist[:50]) / hist_sum

                # 6.7 中间调占比
                mid_ratio = np.sum(hist[50:200]) / hist_sum
            else:
                hist_peak = 0
                hist_centroid = 128
                hist_spread = 0
                hist_entropy = 0
                bright_ratio = 0
                dark_ratio = 0
                mid_ratio = 0

            return {
                'background_ratio': background_ratio,
                'tissue_ratio': tissue_ratio,
                'blur_score': blur_score,
                'color_std': color_std,
                'brightness': brightness,
                # 直方图特征
                'hist_peak': hist_peak,
                'hist_centroid': hist_centroid,
                'hist_spread': hist_spread,
                'hist_entropy': hist_entropy,
                'bright_ratio': bright_ratio,
                'dark_ratio': dark_ratio,
                'mid_ratio': mid_ratio
            }
        except Exception as e:
            return None
        finally:
            if img is not None:
                img.close()

    @staticmethod
    def get_color_histogram(image_path: Path, bins: int = 32) -> np.ndarray:
        """
        获取图像的颜色直方图（RGB三通道）

        Args:
            image_path: 图像路径
            bins: 每个通道的bin数量

        Returns:
            归一化的颜色直方图向量
        """
        img = None
        try:
            img = Image.open(image_path)
            img.thumbnail((128, 128))  # 缩小加速
            img = img.convert('RGB')
            arr = np.array(img)

            # 分别计算RGB三个通道的直方图
            hist_r, _ = np.histogram(arr[:, :, 0], bins=bins, range=(0, 256))
            hist_g, _ = np.histogram(arr[:, :, 1], bins=bins, range=(0, 256))
            hist_b, _ = np.histogram(arr[:, :, 2], bins=bins, range=(0, 256))

            # 合并成一个向量
            hist = np.concatenate([hist_r, hist_g, hist_b]).astype(np.float64)

            # 归一化
            hist_sum = hist.sum()
            if hist_sum > 0:
                hist = hist / hist_sum

            return hist
        except Exception:
            return None
        finally:
            if img is not None:
                img.close()

    @staticmethod
    def histogram_similarity(hist1: np.ndarray, hist2: np.ndarray) -> float:
        """
        计算两个直方图的相似度（使用相关系数）

        Returns:
            相似度，范围 [-1, 1]，1表示完全相同
        """
        if hist1 is None or hist2 is None:
            return 0.0

        # 使用相关系数
        mean1 = np.mean(hist1)
        mean2 = np.mean(hist2)

        num = np.sum((hist1 - mean1) * (hist2 - mean2))
        den = np.sqrt(np.sum((hist1 - mean1) ** 2) * np.sum((hist2 - mean2) ** 2))

        if den == 0:
            return 0.0

        return num / den


class ThumbnailCache:
    """缩略图缓存管理"""

    CACHE_FILE = ".tile_cache.pkl"

    def __init__(self, tiles_dir: Path, thumb_size: int = 80):
        self.tiles_dir = tiles_dir
        self.thumb_size = thumb_size
        self.cache_path = tiles_dir / self.CACHE_FILE
        self.cache: Dict[str, bytes] = {}  # {filename: jpeg_bytes}
        self._load_cache()

    def _load_cache(self):
        """加载缓存"""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'rb') as f:
                    data = pickle.load(f)
                    if data.get('thumb_size') == self.thumb_size:
                        self.cache = data.get('thumbnails', {})
            except:
                self.cache = {}

    def save_cache(self):
        """保存缓存"""
        try:
            data = {
                'thumb_size': self.thumb_size,
                'thumbnails': self.cache
            }
            with open(self.cache_path, 'wb') as f:
                pickle.dump(data, f)
        except:
            pass

    def get_thumbnail(self, tile_path: Path) -> Optional[Image.Image]:
        """获取缩略图（优先从缓存）"""
        filename = tile_path.name

        # 检查缓存
        if filename in self.cache:
            try:
                from io import BytesIO
                return Image.open(BytesIO(self.cache[filename]))
            except:
                pass

        # 生成缩略图
        try:
            img = Image.open(tile_path)
            img.thumbnail((self.thumb_size, self.thumb_size))

            # 转换为RGB并缓存为JPEG bytes
            if img.mode != 'RGB':
                img = img.convert('RGB')

            from io import BytesIO
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=70)
            self.cache[filename] = buffer.getvalue()

            return img
        except:
            return None

    def remove(self, filename: str):
        """从缓存移除"""
        self.cache.pop(filename, None)


class TileReviewer:
    """切片审核工具 - 优化版（虚拟滚动）"""

    THUMB_SIZE = 80  # 缩略图大小（稍小以提高性能）
    COLS = 10  # 每行列数
    VISIBLE_ROWS_BUFFER = 3  # 可见区域外额外加载的行数
    MAX_PHOTO_CACHE = 200  # 最大缓存的PhotoImage数量（防止GDI资源耗尽）

    def __init__(self, tiles_dir: Optional[str] = None):
        self.root = tk.Tk()
        self.root.title("切片审核工具")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)

        # 数据
        self.tiles_dir: Optional[Path] = Path(tiles_dir) if tiles_dir else None
        self.tile_files: List[Path] = []
        self.selected: Set[int] = set()  # 选中的索引
        self.thumbnails: Dict[int, ImageTk.PhotoImage] = {}  # {index: PhotoImage}

        # 缓存
        self.cache: Optional[ThumbnailCache] = None

        # 虚拟滚动
        self.visible_start = 0
        self.visible_end = 0
        self.tile_labels: Dict[int, tk.Label] = {}  # 当前显示的标签

        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.loading_indices: Set[int] = set()

        # 框选
        self.drag_start: Optional[Tuple[int, int]] = None
        self.drag_rect: Optional[int] = None

        # 智能筛选
        self.keep_indices: Set[int] = set()  # 标记为保留的索引
        self.features_cache: Dict[int, Dict] = {}  # 特征缓存
        self.last_learn_result: Optional[Dict] = None  # 上次学习的规则（用于延迟保存）
        self.last_clicked_idx: Optional[int] = None  # 上次点击的图像索引

        # 状态
        self.loading = False

        # 创建界面
        self._create_widgets()

        # 如果提供了目录，直接加载
        if self.tiles_dir:
            self.root.after(100, self._load_tiles)

    def _create_widgets(self):
        """创建界面组件"""
        # 顶部工具栏
        toolbar = ttk.Frame(self.root, padding="5")
        toolbar.pack(fill=tk.X)

        ttk.Button(toolbar, text="选择文件夹", command=self._select_folder).pack(side=tk.LEFT, padx=5)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        ttk.Button(toolbar, text="全选", command=self._select_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="取消全选", command=self._deselect_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="反选", command=self._invert_selection).pack(side=tk.LEFT, padx=5)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        self.delete_btn = ttk.Button(toolbar, text="删除选中 (0)", command=self._delete_selected)
        self.delete_btn.pack(side=tk.LEFT, padx=5)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # 智能筛选按钮
        ttk.Button(toolbar, text="标记为保留", command=self._mark_as_keep).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="从样本学习", command=self._learn_from_samples).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="颜色筛选", command=self._color_filter).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="保存规则", command=self._save_last_rules).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="加载规则", command=self._load_rules).pack(side=tk.LEFT, padx=5)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        ttk.Button(toolbar, text="生成缓存", command=self._generate_cache).pack(side=tk.LEFT, padx=5)

        # 状态栏
        self.status_var = tk.StringVar(value="请选择切片文件夹")
        ttk.Label(toolbar, textvariable=self.status_var).pack(side=tk.RIGHT, padx=10)

        # 主内容区域
        container = ttk.Frame(self.root)
        container.pack(fill=tk.BOTH, expand=True)

        # Canvas和滚动条
        self.canvas = tk.Canvas(container, bg='#1e1e1e', highlightthickness=0)
        self.v_scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=self._on_scroll)

        self.canvas.configure(yscrollcommand=self._on_scroll_change)

        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 绑定事件
        self.canvas.bind('<Configure>', self._on_canvas_configure)
        # 跨平台鼠标滚轮绑定
        self.canvas.bind_all('<MouseWheel>', self._on_mousewheel)  # Windows/macOS
        self.canvas.bind_all('<Button-4>', self._on_mousewheel)    # Linux 向上
        self.canvas.bind_all('<Button-5>', self._on_mousewheel)    # Linux 向下

        # 框选
        self.canvas.bind('<ButtonPress-1>', self._on_drag_start)
        self.canvas.bind('<B1-Motion>', self._on_drag_motion)
        self.canvas.bind('<ButtonRelease-1>', self._on_drag_end)

        # 底部信息栏
        info_bar = ttk.Frame(self.root, padding="5")
        info_bar.pack(fill=tk.X)

        self.info_var = tk.StringVar(value="")
        ttk.Label(info_bar, textvariable=self.info_var).pack(side=tk.LEFT)

        self.progress = ttk.Progressbar(info_bar, length=200, mode='determinate')
        self.progress.pack(side=tk.RIGHT, padx=10)

        # 快捷键
        self.root.bind('<Delete>', lambda e: self._delete_selected())
        self.root.bind('<Control-a>', lambda e: self._select_all())
        self.root.bind('<Escape>', lambda e: self._deselect_all())

    def _on_scroll(self, *args):
        """滚动条事件"""
        self.canvas.yview(*args)
        self._update_visible_tiles()

    def _on_scroll_change(self, first, last):
        """滚动位置变化"""
        self.v_scrollbar.set(first, last)
        self._update_visible_tiles()

    def _on_canvas_configure(self, event):
        """Canvas大小变化"""
        new_cols = max(1, event.width // (self.THUMB_SIZE + 6))
        if new_cols != self.COLS:
            self.COLS = new_cols
            self._refresh_layout()
        else:
            self._update_visible_tiles()

    def _on_mousewheel(self, event):
        """鼠标滚轮（跨平台兼容）"""
        # Windows 和 macOS 使用 event.delta
        # Linux 使用 Button-4 (向上) 和 Button-5 (向下)
        if event.delta:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        elif event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")
        self._update_visible_tiles()

    def _select_folder(self):
        """选择文件夹"""
        folder = filedialog.askdirectory(title="选择切片文件夹")
        if folder:
            self.tiles_dir = Path(folder)
            self._load_tiles()

    def _load_tiles(self):
        """加载切片列表"""
        if not self.tiles_dir or not self.tiles_dir.exists():
            return

        self.status_var.set("正在扫描文件...")
        self.root.update()

        # 清空
        self.tile_files.clear()
        self.selected.clear()
        self.thumbnails.clear()
        self._clear_canvas()

        # 查找切片文件（支持 PNG 和 JPEG）
        png_files = list(self.tiles_dir.glob("tile_*.png"))
        jpg_files = list(self.tiles_dir.glob("tile_*.jpg"))
        self.tile_files = sorted(png_files + jpg_files, key=lambda x: x.name)

        if not self.tile_files:
            self.status_var.set("未找到切片文件")
            return

        # 初始化缓存
        self.cache = ThumbnailCache(self.tiles_dir, self.THUMB_SIZE)

        # 设置滚动区域
        self._refresh_layout()
        self._update_info()

        cached_count = len(self.cache.cache)
        total = len(self.tile_files)
        self.status_var.set(f"共 {total} 个切片 (缓存: {cached_count})")

    def _refresh_layout(self):
        """刷新布局"""
        if not self.tile_files:
            return

        total = len(self.tile_files)
        rows = (total + self.COLS - 1) // self.COLS
        cell_height = self.THUMB_SIZE + 20

        # 设置滚动区域
        total_height = rows * cell_height
        self.canvas.configure(scrollregion=(0, 0, self.canvas.winfo_width(), total_height))

        self._clear_canvas()
        self._update_visible_tiles()

    def _clear_canvas(self):
        """清空画布"""
        self.canvas.delete('all')
        self.tile_labels.clear()
        # 同时清理PhotoImage释放GDI资源
        self.thumbnails.clear()

    def _update_visible_tiles(self):
        """更新可见区域的切片"""
        if not self.tile_files:
            return

        # 计算可见区域
        canvas_height = self.canvas.winfo_height()
        cell_height = self.THUMB_SIZE + 20

        # 获取当前滚动位置
        try:
            y_top = self.canvas.canvasy(0)
            y_bottom = self.canvas.canvasy(canvas_height)
        except:
            return

        # 计算可见行范围
        start_row = max(0, int(y_top / cell_height) - self.VISIBLE_ROWS_BUFFER)
        end_row = int(y_bottom / cell_height) + self.VISIBLE_ROWS_BUFFER + 1

        # 计算索引范围
        start_idx = start_row * self.COLS
        end_idx = min(end_row * self.COLS, len(self.tile_files))

        # 移除不在视野内的（删除canvas元素和PhotoImage，彻底释放GDI资源）
        to_remove = []
        for idx in self.tile_labels:
            if idx < start_idx or idx >= end_idx:
                to_remove.append(idx)

        for idx in to_remove:
            self.canvas.delete(f"tile_{idx}")
            self.canvas.delete(f"text_{idx}")
            self.canvas.delete(f"frame_{idx}")
            self.canvas.delete(f"select_{idx}")
            self.canvas.delete(f"keep_{idx}")
            del self.tile_labels[idx]
            # 关键：删除PhotoImage释放GDI资源，下次滚动回来时重新加载
            if idx in self.thumbnails:
                del self.thumbnails[idx]

        # 添加新进入视野的
        for idx in range(start_idx, end_idx):
            if idx not in self.tile_labels and idx < len(self.tile_files):
                self._create_tile_placeholder(idx)
                # 如果已有缓存的PhotoImage，直接显示
                if idx in self.thumbnails:
                    self._apply_cached_thumbnail(idx)
                else:
                    self._load_thumbnail_async(idx)

    def _apply_cached_thumbnail(self, idx: int):
        """直接应用已缓存的缩略图（无需重新加载）"""
        photo = self.thumbnails.get(idx)
        if photo is None or idx not in self.tile_labels:
            return

        row = idx // self.COLS
        col = idx % self.COLS
        cell_width = self.THUMB_SIZE + 6
        cell_height = self.THUMB_SIZE + 20

        x = col * cell_width + 3 + self.THUMB_SIZE // 2
        y = row * cell_height + 3 + self.THUMB_SIZE // 2

        self.canvas.create_image(x, y, image=photo, tags=(f"tile_{idx}", "tile"))
        self.canvas.tag_bind(f"tile_{idx}", '<Button-1>',
                             lambda e, i=idx: self._toggle_selection(i))
        self.tile_labels[idx] = photo

    def _create_tile_placeholder(self, idx: int):
        """创建占位符"""
        row = idx // self.COLS
        col = idx % self.COLS
        cell_width = self.THUMB_SIZE + 6
        cell_height = self.THUMB_SIZE + 20

        x = col * cell_width + 3
        y = row * cell_height + 3

        # 创建背景框
        self.canvas.create_rectangle(
            x, y, x + self.THUMB_SIZE, y + self.THUMB_SIZE,
            fill='#2d2d2d', outline='#3d3d3d', tags=(f"frame_{idx}", "frame")
        )

        # 绑定点击事件
        self.canvas.tag_bind(f"frame_{idx}", '<Button-1>',
                             lambda e, i=idx: self._toggle_selection(i))

        # 文件名
        filename = self.tile_files[idx].name[:12]
        self.canvas.create_text(
            x + self.THUMB_SIZE // 2, y + self.THUMB_SIZE + 8,
            text=filename, fill='#888888', font=('', 7),
            tags=(f"text_{idx}",)
        )

        # 如果已选中，显示选中效果
        if idx in self.selected:
            self._show_selection_overlay(idx)

        self.tile_labels[idx] = None  # 占位

    def _show_selection_overlay(self, idx: int):
        """显示选中遮罩效果"""
        row = idx // self.COLS
        col = idx % self.COLS
        cell_width = self.THUMB_SIZE + 6
        cell_height = self.THUMB_SIZE + 20

        x = col * cell_width + 3
        y = row * cell_height + 3

        # 红色粗边框
        self.canvas.create_rectangle(
            x, y, x + self.THUMB_SIZE, y + self.THUMB_SIZE,
            fill='', outline='#ff0000', width=4,
            tags=(f"select_{idx}", "selection")
        )

        # 右上角的勾选标记背景
        self.canvas.create_oval(
            x + self.THUMB_SIZE - 22, y + 2,
            x + self.THUMB_SIZE - 2, y + 22,
            fill='#ff0000', outline='',
            tags=(f"select_{idx}", "selection")
        )

        # 勾选标记 ✓
        self.canvas.create_text(
            x + self.THUMB_SIZE - 12, y + 12,
            text='✓', fill='white', font=('', 12, 'bold'),
            tags=(f"select_{idx}", "selection")
        )

        # 绑定点击
        self.canvas.tag_bind(f"select_{idx}", '<Button-1>',
                             lambda e, i=idx: self._toggle_selection(i))

    def _hide_selection_overlay(self, idx: int):
        """隐藏选中遮罩"""
        self.canvas.delete(f"select_{idx}")

    def _load_thumbnail_async(self, idx: int):
        """异步加载缩略图"""
        if idx in self.loading_indices or idx in self.thumbnails:
            return

        self.loading_indices.add(idx)

        def load():
            if idx >= len(self.tile_files):
                return None
            tile_path = self.tile_files[idx]
            img = self.cache.get_thumbnail(tile_path)
            return (idx, img)

        def callback(future):
            try:
                result = future.result()
                if result:
                    self.root.after(0, lambda: self._apply_thumbnail(*result))
            except:
                pass
            finally:
                self.loading_indices.discard(idx)

        future = self.executor.submit(load)
        future.add_done_callback(callback)

    def _apply_thumbnail(self, idx: int, img: Optional[Image.Image]):
        """应用缩略图到画布"""
        if img is None or idx not in self.tile_labels:
            return

        try:
            photo = ImageTk.PhotoImage(img)
            self.thumbnails[idx] = photo

            row = idx // self.COLS
            col = idx % self.COLS
            cell_width = self.THUMB_SIZE + 6
            cell_height = self.THUMB_SIZE + 20

            x = col * cell_width + 3 + self.THUMB_SIZE // 2
            y = row * cell_height + 3 + self.THUMB_SIZE // 2

            self.canvas.create_image(x, y, image=photo, tags=(f"tile_{idx}", "tile"))
            self.canvas.tag_bind(f"tile_{idx}", '<Button-1>',
                                 lambda e, i=idx: self._toggle_selection(i))

            self.tile_labels[idx] = photo
        except:
            pass

    def _toggle_selection(self, idx: int):
        """切换选中状态"""
        self.last_clicked_idx = idx  # 记录最后点击的图像
        if idx in self.selected:
            self.selected.discard(idx)
            self._hide_selection_overlay(idx)
        else:
            self.selected.add(idx)
            self._show_selection_overlay(idx)

        self._update_info()

    def _select_all(self):
        """全选"""
        self.selected = set(range(len(self.tile_files)))
        for idx in self.tile_labels:
            self._show_selection_overlay(idx)
        self._update_info()

    def _deselect_all(self):
        """取消全选"""
        for idx in list(self.selected):
            if idx in self.tile_labels:
                self._hide_selection_overlay(idx)
        self.selected.clear()
        self._update_info()

    def _invert_selection(self):
        """反选"""
        all_indices = set(range(len(self.tile_files)))
        new_selected = all_indices - self.selected

        # 更新显示
        for idx in self.tile_labels:
            if idx in self.selected and idx not in new_selected:
                self._hide_selection_overlay(idx)
            elif idx not in self.selected and idx in new_selected:
                self._show_selection_overlay(idx)

        self.selected = new_selected
        self._update_info()

    def _on_drag_start(self, event):
        """开始框选"""
        self.drag_start = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))

    def _on_drag_motion(self, event):
        """框选拖动"""
        if self.drag_start is None:
            return

        if self.drag_rect:
            self.canvas.delete(self.drag_rect)

        x1, y1 = self.drag_start
        x2, y2 = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        self.drag_rect = self.canvas.create_rectangle(
            x1, y1, x2, y2, outline='#00ff00', width=2, dash=(4, 4)
        )

    def _on_drag_end(self, event):
        """框选结束"""
        if self.drag_start is None:
            return

        if self.drag_rect:
            self.canvas.delete(self.drag_rect)
            self.drag_rect = None

        x1, y1 = self.drag_start
        x2, y2 = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1

        # 只有拖动了一定距离才算框选
        if abs(x2 - x1) < 10 and abs(y2 - y1) < 10:
            self.drag_start = None
            return

        # 计算框选区域内的切片
        cell_width = self.THUMB_SIZE + 6
        cell_height = self.THUMB_SIZE + 20

        start_col = max(0, int(x1 / cell_width))
        end_col = min(self.COLS, int(x2 / cell_width) + 1)
        start_row = max(0, int(y1 / cell_height))
        end_row = int(y2 / cell_height) + 1

        for row in range(start_row, end_row):
            for col in range(start_col, end_col):
                idx = row * self.COLS + col
                if idx < len(self.tile_files) and idx not in self.selected:
                    self.selected.add(idx)
                    if idx in self.tile_labels:
                        self._show_selection_overlay(idx)

        self.drag_start = None
        self._update_info()

    def _update_info(self):
        """更新信息"""
        total = len(self.tile_files)
        selected = len(self.selected)
        self.info_var.set(f"总计: {total} | 已选中: {selected}")
        self.delete_btn.configure(text=f"删除选中 ({selected})")

    def _delete_selected(self):
        """删除选中"""
        if not self.selected:
            messagebox.showinfo("提示", "没有选中任何切片")
            return

        count = len(self.selected)
        if not messagebox.askyesno("确认删除", f"确定删除 {count} 个切片？\n此操作不可撤销！"):
            return

        # 按索引降序删除
        deleted = 0
        for idx in sorted(self.selected, reverse=True):
            try:
                tile_path = self.tile_files[idx]
                tile_path.unlink()
                self.cache.remove(tile_path.name)
                del self.tile_files[idx]
                deleted += 1
            except Exception as e:
                print(f"删除失败: {e}")

        self.selected.clear()
        self.cache.save_cache()
        self._update_metadata()

        messagebox.showinfo("完成", f"已删除 {deleted} 个切片")
        self._refresh_layout()
        self._update_info()
        self.status_var.set(f"共 {len(self.tile_files)} 个切片")

    def _update_metadata(self):
        """更新元数据"""
        if not self.tiles_dir:
            return

        metadata_path = self.tiles_dir / 'metadata.json'
        if not metadata_path.exists():
            return

        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            remaining = {p.name for p in self.tile_files}
            if 'tiles' in metadata:
                metadata['tiles'] = [t for t in metadata['tiles'] if t['filename'] in remaining]
                metadata['saved_tiles'] = len(metadata['tiles'])

            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except:
            pass

    def _generate_cache(self):
        """预生成所有缩略图缓存"""
        if not self.cache or not self.tile_files:
            return

        total = len(self.tile_files)
        cached = len(self.cache.cache)

        if cached >= total:
            messagebox.showinfo("提示", "缓存已是最新")
            return

        if not messagebox.askyesno("生成缓存", f"将为 {total - cached} 个切片生成缓存\n这可能需要一些时间，是否继续？"):
            return

        self.progress['maximum'] = total
        self.progress['value'] = 0
        self.status_var.set("正在生成缓存...")

        def generate():
            for i, tile_path in enumerate(self.tile_files):
                if tile_path.name not in self.cache.cache:
                    self.cache.get_thumbnail(tile_path)

                self.root.after(0, lambda v=i+1: self._update_cache_progress(v, total))

            self.cache.save_cache()
            self.root.after(0, self._cache_done)

        threading.Thread(target=generate, daemon=True).start()

    def _update_cache_progress(self, current, total):
        """更新缓存进度"""
        self.progress['value'] = current
        self.status_var.set(f"生成缓存中... {current}/{total}")

    def _cache_done(self):
        """缓存完成"""
        self.status_var.set(f"缓存完成！共 {len(self.tile_files)} 个切片")
        self.progress['value'] = 0
        messagebox.showinfo("完成", "缩略图缓存已生成，下次打开将秒加载")

    def _mark_as_keep(self):
        """将当前选中的标记为保留（好样本）"""
        if not self.selected:
            messagebox.showinfo("提示", "请先选中要保留的切片（好样本）")
            return

        # 添加到保留列表
        self.keep_indices.update(self.selected)

        # 更新显示（用绿色边框标记保留的）
        for idx in self.selected:
            if idx in self.tile_labels:
                self._show_keep_overlay(idx)

        count = len(self.keep_indices)
        self.status_var.set(f"已标记 {count} 个保留样本")
        messagebox.showinfo("完成", f"已将 {len(self.selected)} 个切片标记为保留\n共 {count} 个保留样本")

        # 清除选中
        self._deselect_all()

    def _save_last_rules(self):
        """保存上次学习的规则"""
        if self.last_learn_result is None:
            messagebox.showinfo("提示", "没有可保存的规则\n请先使用'从样本学习'功能")
            return

        file_path = filedialog.asksaveasfilename(
            title="保存筛选规则",
            defaultextension=".json",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")],
            initialfile="filter_rules.json"
        )
        if not file_path:
            return

        # 处理直方图类型的规则（将numpy数组转为列表）
        save_data = {
            'version': '2.0',
            'description': '切片筛选规则 - 基于颜色直方图',
        }

        if self.last_learn_result.get('type') == 'histogram':
            save_data['type'] = 'histogram'
            save_data['threshold'] = self.last_learn_result['threshold']
            # 将numpy数组转为列表以便JSON序列化
            save_data['bad_histograms'] = [h.tolist() for h in self.last_learn_result['bad_histograms']]
        else:
            # 兼容旧格式
            save_data.update(self.last_learn_result)

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            messagebox.showinfo("成功", f"规则已保存到:\n{file_path}")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {str(e)}")

    def _color_filter(self):
        """颜色筛选 - 点击图像吸取颜色，范围内保留，范围外删除"""
        if not self.tile_files:
            messagebox.showinfo("提示", "请先加载切片文件夹")
            return

        # 创建颜色筛选对话框
        dialog = tk.Toplevel(self.root)
        dialog.title("颜色筛选")
        dialog.geometry("900x750")
        dialog.transient(self.root)
        dialog.grab_set()

        # 颜色条件列表: [(rgb, tolerance, min_ratio, max_ratio), ...]
        color_rules = []

        # === 左侧：图像预览区 ===
        left_frame = ttk.Frame(dialog)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(left_frame, text="点击图像吸取颜色", font=('', 11, 'bold')).pack(pady=5)

        # 图像选择
        select_frame = ttk.Frame(left_frame)
        select_frame.pack(fill=tk.X, pady=5)

        current_img_var = tk.StringVar(value="")
        ttk.Label(select_frame, text="选择图像:").pack(side=tk.LEFT)
        img_combo = ttk.Combobox(select_frame, textvariable=current_img_var, width=30, state='readonly')

        # 构建图像列表：优先显示最后点击的图像
        img_names = [f.name for f in self.tile_files[:100]]
        default_idx = 0
        clicked_name = None

        if self.last_clicked_idx is not None and self.last_clicked_idx < len(self.tile_files):
            clicked_name = self.tile_files[self.last_clicked_idx].name
            if self.last_clicked_idx < 100:
                default_idx = self.last_clicked_idx
            else:
                # 如果点击的图像不在前100个，把它加到列表开头
                img_names = [clicked_name] + img_names
                default_idx = 0

        img_combo['values'] = img_names
        img_combo.pack(side=tk.LEFT, padx=5)

        # 提示当前选中的图像
        if clicked_name:
            ttk.Label(select_frame, text=f"← 当前: {clicked_name[:20]}...", foreground='blue').pack(side=tk.LEFT, padx=5)

        # 图像显示Canvas
        img_canvas = tk.Canvas(left_frame, bg='#2d2d2d', width=400, height=400)
        img_canvas.pack(fill=tk.BOTH, expand=True, pady=10)

        current_photo = [None]
        current_pil_img = [None]
        img_scale = [1.0]

        # 选中的颜色显示
        picked_color_frame = ttk.Frame(left_frame)
        picked_color_frame.pack(fill=tk.X, pady=5)

        ttk.Label(picked_color_frame, text="吸取的颜色:").pack(side=tk.LEFT)
        color_preview = tk.Label(picked_color_frame, width=4, height=2, bg='#808080', relief='solid')
        color_preview.pack(side=tk.LEFT, padx=5)
        color_rgb_var = tk.StringVar(value="RGB: -")
        ttk.Label(picked_color_frame, textvariable=color_rgb_var).pack(side=tk.LEFT, padx=5)

        picked_rgb = [None]

        def load_image(event=None):
            filename = current_img_var.get()
            if not filename:
                return

            tile_path = self.tiles_dir / filename
            if not tile_path.exists():
                return

            try:
                img = Image.open(tile_path)
                current_pil_img[0] = img.copy()

                canvas_w = img_canvas.winfo_width() or 400
                canvas_h = img_canvas.winfo_height() or 400
                scale = min(canvas_w / img.width, canvas_h / img.height, 1.0)
                img_scale[0] = scale

                new_w = int(img.width * scale)
                new_h = int(img.height * scale)
                img_resized = img.resize((new_w, new_h), Image.LANCZOS)

                photo = ImageTk.PhotoImage(img_resized)
                current_photo[0] = photo

                img_canvas.delete('all')
                img_canvas.create_image(canvas_w // 2, canvas_h // 2, image=photo, anchor=tk.CENTER)
                img.close()
            except Exception as e:
                print(f"加载图像失败: {e}")

        def pick_color(event):
            if current_pil_img[0] is None:
                return

            canvas_w = img_canvas.winfo_width()
            canvas_h = img_canvas.winfo_height()
            img_w, img_h = current_pil_img[0].size
            scale = img_scale[0]

            display_w = int(img_w * scale)
            display_h = int(img_h * scale)
            offset_x = (canvas_w - display_w) // 2
            offset_y = (canvas_h - display_h) // 2

            img_x = int((event.x - offset_x) / scale)
            img_y = int((event.y - offset_y) / scale)

            if 0 <= img_x < img_w and 0 <= img_y < img_h:
                rgb = current_pil_img[0].convert('RGB').getpixel((img_x, img_y))
                picked_rgb[0] = rgb
                hex_color = f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'
                color_preview.configure(bg=hex_color)
                color_rgb_var.set(f"RGB: {rgb[0]}, {rgb[1]}, {rgb[2]}")

        img_combo.bind('<<ComboboxSelected>>', load_image)
        img_canvas.bind('<Button-1>', pick_color)

        # 默认选择并加载图像
        if self.tile_files:
            img_combo.current(default_idx)
            dialog.after(100, load_image)

        # === 右侧：规则设置区 ===
        right_frame = ttk.Frame(dialog, width=380)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        right_frame.pack_propagate(False)

        ttk.Label(right_frame, text="颜色筛选规则", font=('', 11, 'bold')).pack(pady=5)
        ttk.Label(right_frame, text="设置颜色占比范围，范围内保留，范围外删除", foreground='gray').pack(pady=2)

        # 参数设置
        param_frame = ttk.LabelFrame(right_frame, text="添加规则", padding=10)
        param_frame.pack(fill=tk.X, pady=10)

        # 容差
        ttk.Label(param_frame, text="颜色容差 (越大匹配越宽松):").pack(anchor=tk.W)
        tolerance_var = tk.IntVar(value=30)
        tol_frame = ttk.Frame(param_frame)
        tol_frame.pack(fill=tk.X)
        ttk.Scale(tol_frame, from_=5, to=100, variable=tolerance_var, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tolerance_label = ttk.Label(tol_frame, text="30", width=4)
        tolerance_label.pack(side=tk.RIGHT)
        tolerance_var.trace('w', lambda *a: tolerance_label.config(text=str(tolerance_var.get())))

        # 占比范围 - 最小值
        ttk.Label(param_frame, text="占比范围 - 最小 (%):").pack(anchor=tk.W, pady=(10, 0))
        min_ratio_var = tk.DoubleVar(value=5.0)
        min_frame = ttk.Frame(param_frame)
        min_frame.pack(fill=tk.X)
        ttk.Scale(min_frame, from_=0, to=100, variable=min_ratio_var, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
        min_label = ttk.Label(min_frame, text="5%", width=5)
        min_label.pack(side=tk.RIGHT)
        min_ratio_var.trace('w', lambda *a: min_label.config(text=f"{min_ratio_var.get():.0f}%"))

        # 占比范围 - 最大值
        ttk.Label(param_frame, text="占比范围 - 最大 (%):").pack(anchor=tk.W, pady=(5, 0))
        max_ratio_var = tk.DoubleVar(value=50.0)
        max_frame = ttk.Frame(param_frame)
        max_frame.pack(fill=tk.X)
        ttk.Scale(max_frame, from_=0, to=100, variable=max_ratio_var, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
        max_label = ttk.Label(max_frame, text="50%", width=5)
        max_label.pack(side=tk.RIGHT)
        max_ratio_var.trace('w', lambda *a: max_label.config(text=f"{max_ratio_var.get():.0f}%"))

        # 说明
        ttk.Label(param_frame, text="→ 占比在此范围内的图像会被保留", foreground='green').pack(anchor=tk.W, pady=(10, 0))
        ttk.Label(param_frame, text="→ 占比超出此范围的图像会被选中删除", foreground='red').pack(anchor=tk.W)

        def add_rule():
            if picked_rgb[0] is None:
                messagebox.showinfo("提示", "请先点击图像吸取颜色")
                return

            min_r = min_ratio_var.get() / 100.0
            max_r = max_ratio_var.get() / 100.0
            if min_r >= max_r:
                messagebox.showinfo("提示", "最小值必须小于最大值")
                return

            rgb = picked_rgb[0]
            tolerance = tolerance_var.get()
            color_rules.append((rgb, tolerance, min_r, max_r))
            update_rules_list()

        ttk.Button(param_frame, text="添加此颜色规则", command=add_rule).pack(fill=tk.X, pady=(15, 0))

        # 规则列表
        rules_frame = ttk.LabelFrame(right_frame, text="已添加的规则", padding=10)
        rules_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        rules_count_var = tk.StringVar(value="(0条规则)")
        ttk.Label(rules_frame, textvariable=rules_count_var).pack(anchor=tk.W)

        rules_listbox = tk.Listbox(rules_frame, height=6, bg='white', fg='black',
                                   selectbackground='#0078d7', selectforeground='white')
        rules_listbox.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        def update_rules_list():
            rules_listbox.delete(0, tk.END)
            for i, (rgb, tol, min_r, max_r) in enumerate(color_rules):
                text = f"{i+1}. RGB({rgb[0]},{rgb[1]},{rgb[2]}) 容差{tol} 保留{min_r*100:.0f}%-{max_r*100:.0f}%"
                rules_listbox.insert(tk.END, text)
            rules_count_var.set(f"({len(color_rules)}条规则)")
            rules_listbox.update_idletasks()

        def remove_selected_rule():
            sel = rules_listbox.curselection()
            if sel:
                color_rules.pop(sel[0])
                update_rules_list()

        ttk.Button(rules_frame, text="删除选中规则", command=remove_selected_rule).pack(fill=tk.X, pady=(5, 0))

        # 规则逻辑选择
        logic_frame = ttk.LabelFrame(right_frame, text="多规则逻辑", padding=5)
        logic_frame.pack(fill=tk.X, pady=5)

        logic_var = tk.StringVar(value="any")
        ttk.Radiobutton(logic_frame, text="任一规则不满足即删除", variable=logic_var, value="any").pack(anchor=tk.W)
        ttk.Radiobutton(logic_frame, text="全部规则都不满足才删除", variable=logic_var, value="all").pack(anchor=tk.W)

        # 进度和结果
        progress_var = tk.DoubleVar(value=0)
        progress = ttk.Progressbar(right_frame, variable=progress_var, maximum=100)
        progress.pack(fill=tk.X, pady=10)

        status_var = tk.StringVar(value="")
        ttk.Label(right_frame, textvariable=status_var).pack()

        result_var = tk.StringVar(value="")
        ttk.Label(right_frame, textvariable=result_var).pack(pady=5)

        filter_matched = []

        def run_filter():
            if not color_rules:
                messagebox.showinfo("提示", "请先添加至少一个颜色规则")
                return

            filter_matched.clear()
            total = len(self.tile_files)
            status_var.set("正在筛选...")
            use_any_logic = (logic_var.get() == "any")

            target_colors = [(rgb, tol) for rgb, tol, _, _ in color_rules]

            def do_filter():
                import gc
                for i, tile_path in enumerate(self.tile_files):
                    if i in self.keep_indices:
                        continue

                    ratios = ImageAnalyzer.get_color_ratio(tile_path, target_colors)

                    # 检查规则
                    out_of_range_count = 0
                    reasons = []
                    for j, (rgb, tol, min_r, max_r) in enumerate(color_rules):
                        ratio = ratios.get(j, 0)
                        if ratio < min_r or ratio > max_r:
                            out_of_range_count += 1
                            if ratio < min_r:
                                reasons.append(f"颜色{j+1}:{ratio*100:.1f}%<{min_r*100:.0f}%")
                            else:
                                reasons.append(f"颜色{j+1}:{ratio*100:.1f}%>{max_r*100:.0f}%")

                    # 判断是否删除
                    should_delete = False
                    if use_any_logic:
                        # 任一规则不满足即删除
                        should_delete = (out_of_range_count > 0)
                    else:
                        # 全部规则都不满足才删除
                        should_delete = (out_of_range_count == len(color_rules))

                    if should_delete:
                        filter_matched.append((i, tile_path.name, ', '.join(reasons)))

                    if i % 5 == 0:
                        dialog.after(0, lambda v=(i+1)/total*100: progress_var.set(v))

                    if i % 50 == 0:
                        gc.collect()

                dialog.after(0, on_filter_done)

            def on_filter_done():
                progress_var.set(100)
                status_var.set("筛选完成")
                result_var.set(f"找到 {len(filter_matched)} 个待删除项")
                apply_btn.config(state=tk.NORMAL)

            threading.Thread(target=do_filter, daemon=True).start()

        def apply_result():
            if not filter_matched:
                messagebox.showinfo("提示", "没有匹配项")
                return

            self._deselect_all()
            for idx, name, reason in filter_matched:
                self.selected.add(idx)
                if idx in self.tile_labels:
                    self._show_selection_overlay(idx)
            self._update_info()
            dialog.destroy()
            messagebox.showinfo("完成",
                f"已选中 {len(filter_matched)} 个待删除项\n"
                "请检查后点击'删除选中'确认删除")

        btn_frame = ttk.Frame(right_frame)
        btn_frame.pack(fill=tk.X, pady=10)

        ttk.Button(btn_frame, text="开始筛选", command=run_filter).pack(side=tk.LEFT, padx=5)
        apply_btn = ttk.Button(btn_frame, text="选中待删除项", command=apply_result, state=tk.DISABLED)
        apply_btn.pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="关闭", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)

    def _learn_from_samples(self):
        """从选中的坏样本学习，基于颜色直方图筛选相似图片"""
        if not self.selected:
            messagebox.showinfo("提示", "请先选中要删除的切片作为样本")
            return

        if len(self.selected) < 1:
            messagebox.showinfo("提示", "请至少选择1个样本")
            return

        # 创建学习对话框
        dialog = tk.Toplevel(self.root)
        dialog.title("从样本学习")
        dialog.geometry("550x500")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="基于颜色直方图筛选相似图片", font=('', 12, 'bold')).pack(pady=10)

        # 样本统计
        info_frame = ttk.Frame(dialog)
        info_frame.pack(fill=tk.X, padx=20, pady=5)
        ttk.Label(info_frame, text=f"已选择 {len(self.selected)} 个样本", foreground='red').pack(anchor=tk.W)
        ttk.Label(info_frame, text="将筛选出颜色直方图相似的图片", foreground='gray').pack(anchor=tk.W)

        # 相似度阈值设置
        threshold_frame = ttk.LabelFrame(dialog, text="相似度阈值", padding=10)
        threshold_frame.pack(fill=tk.X, padx=20, pady=10)

        ttk.Label(threshold_frame, text="相似度越高，筛选越严格（匹配越少）").pack(anchor=tk.W)
        threshold_var = tk.DoubleVar(value=0.85)
        threshold_scale = ttk.Scale(threshold_frame, from_=0.5, to=0.99, variable=threshold_var,
                                     orient=tk.HORIZONTAL, length=300)
        threshold_scale.pack(fill=tk.X, pady=5)
        threshold_label = ttk.Label(threshold_frame, text="0.85")
        threshold_label.pack()
        threshold_var.trace('w', lambda *a: threshold_label.config(text=f"{threshold_var.get():.2f}"))

        # 进度条
        progress = ttk.Progressbar(dialog, length=400, mode='determinate')
        progress.pack(pady=10)
        status_label = ttk.Label(dialog, text="点击开始分析")
        status_label.pack(pady=5)

        # 匹配结果
        result_frame = ttk.LabelFrame(dialog, text="筛选结果", padding=10)
        result_frame.pack(fill=tk.X, padx=20, pady=10)

        match_var = tk.StringVar(value="待分析...")
        ttk.Label(result_frame, textvariable=match_var).pack()

        # 按钮
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill=tk.X, padx=20, pady=10)

        learn_result = {'matched': [], 'bad_histograms': [], 'original_samples': list(self.selected)}
        dialog_closed = [False]

        def on_dialog_close():
            dialog_closed[0] = True
            dialog.destroy()

        dialog.protocol("WM_DELETE_WINDOW", on_dialog_close)

        def safe_after(func):
            if not dialog_closed[0]:
                try:
                    dialog.after(0, func)
                except:
                    pass

        def analyze():
            try:
                threshold = threshold_var.get()

                # 1. 计算所有坏样本的颜色直方图
                bad_list = list(self.selected)
                bad_histograms = []

                safe_after(lambda: status_label.config(text="正在分析样本..."))
                safe_after(lambda: progress.configure(maximum=len(bad_list) + len(self.tile_files)))

                for i, idx in enumerate(bad_list):
                    if dialog_closed[0]:
                        return
                    tile_path = self.tile_files[idx]
                    hist = ImageAnalyzer.get_color_histogram(tile_path)
                    if hist is not None:
                        bad_histograms.append(hist)
                    safe_after(lambda v=i+1: progress.configure(value=v))

                if not bad_histograms:
                    safe_after(lambda: status_label.config(text="无法读取样本图片"))
                    return

                learn_result['bad_histograms'] = bad_histograms

                # 2. 计算平均直方图作为参考
                avg_histogram = np.mean(bad_histograms, axis=0)

                # 3. 筛选所有图片
                safe_after(lambda: status_label.config(text="正在筛选相似图片..."))
                matched = []
                base_progress = len(bad_list)

                import gc
                for i, tile_path in enumerate(self.tile_files):
                    if dialog_closed[0]:
                        return

                    # 跳过已选中的样本
                    if i in self.selected:
                        continue

                    hist = ImageAnalyzer.get_color_histogram(tile_path)
                    if hist is None:
                        continue

                    # 计算与平均直方图的相似度
                    similarity = ImageAnalyzer.histogram_similarity(hist, avg_histogram)

                    # 也计算与每个坏样本的最大相似度
                    max_similarity = similarity
                    for bad_hist in bad_histograms:
                        sim = ImageAnalyzer.histogram_similarity(hist, bad_hist)
                        if sim > max_similarity:
                            max_similarity = sim

                    if max_similarity >= threshold:
                        matched.append((i, max_similarity))

                    if i % 10 == 0:
                        safe_after(lambda v=base_progress+i: progress.configure(value=v))

                    if i % 100 == 0:
                        gc.collect()

                # 按相似度排序
                matched.sort(key=lambda x: x[1], reverse=True)
                learn_result['matched'] = [idx for idx, _ in matched]

                # 5. 显示结果
                original_count = len(learn_result['original_samples'])
                result_msg = f"找到 {len(matched)} 个相似图片\n"
                result_msg += f"（加上原样本共 {original_count + len(matched)} 个）"
                if matched:
                    result_msg += f"\n最高相似度: {matched[0][1]:.2f}"
                    if len(matched) > 1:
                        result_msg += f"，最低: {matched[-1][1]:.2f}"

                safe_after(lambda: match_var.set(result_msg))
                safe_after(lambda: status_label.config(text="分析完成！"))
                safe_after(lambda: apply_btn.config(state=tk.NORMAL))
                safe_after(lambda: progress.configure(value=0))

            except Exception as e:
                safe_after(lambda: status_label.config(text=f"分析出错: {str(e)}"))
                import traceback
                traceback.print_exc()

        def start_analyze():
            start_btn.config(state=tk.DISABLED)
            threading.Thread(target=analyze, daemon=True).start()

        def apply_matches():
            matched = learn_result['matched']
            original_samples = learn_result['original_samples']

            self._deselect_all()

            # 先加入原始样本
            for idx in original_samples:
                self.selected.add(idx)
                if idx in self.tile_labels:
                    self._show_selection_overlay(idx)

            # 再加入匹配项
            for idx in matched:
                self.selected.add(idx)
                if idx in self.tile_labels:
                    self._show_selection_overlay(idx)
            self._update_info()

            # 保存直方图用于后续保存规则
            self.last_learn_result = {
                'type': 'histogram',
                'bad_histograms': learn_result['bad_histograms'],
                'threshold': threshold_var.get()
            }

            total_selected = len(original_samples) + len(matched)
            dialog.destroy()
            messagebox.showinfo("完成",
                f"已选中 {total_selected} 个图片\n"
                f"（原样本 {len(original_samples)} 个 + 匹配 {len(matched)} 个）\n"
                "请检查后点击'删除选中'确认删除")

        start_btn = ttk.Button(btn_frame, text="开始分析", command=start_analyze)
        start_btn.pack(side=tk.LEFT, padx=5)
        apply_btn = ttk.Button(btn_frame, text="选中所有匹配项", state=tk.DISABLED, command=apply_matches)
        apply_btn.pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="关闭", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)

    def _load_rules(self):
        """加载已保存的筛选规则"""
        if not self.tile_files:
            messagebox.showinfo("提示", "请先加载切片文件夹")
            return

        # 选择规则文件
        file_path = filedialog.askopenfilename(
            title="加载筛选规则",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
        )
        if not file_path:
            return

        # 读取规则
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                rules_data = json.load(f)
        except Exception as e:
            messagebox.showerror("错误", f"读取规则失败: {str(e)}")
            return

        # 检查规则类型
        rule_type = rules_data.get('type', 'legacy')

        if rule_type == 'histogram':
            self._apply_histogram_rules(rules_data)
        else:
            messagebox.showerror("错误", "不支持的规则格式")

    def _apply_histogram_rules(self, rules_data):
        """应用直方图类型的规则"""
        # 创建应用规则对话框
        dialog = tk.Toplevel(self.root)
        dialog.title("应用筛选规则")
        dialog.geometry("500x400")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="应用颜色直方图筛选规则", font=('', 12, 'bold')).pack(pady=10)

        # 显示规则信息
        info_frame = ttk.LabelFrame(dialog, text="规则信息", padding=10)
        info_frame.pack(fill=tk.X, padx=20, pady=10)

        threshold = rules_data.get('threshold', 0.85)
        bad_histograms = [np.array(h) for h in rules_data['bad_histograms']]

        info_text = f"样本直方图数量: {len(bad_histograms)}\n"
        info_text += f"相似度阈值: {threshold:.2f}\n"
        if 'description' in rules_data:
            info_text += f"描述: {rules_data['description']}\n"

        ttk.Label(info_frame, text=info_text).pack(anchor=tk.W)

        # 阈值调整
        threshold_frame = ttk.Frame(dialog)
        threshold_frame.pack(fill=tk.X, padx=20, pady=5)
        ttk.Label(threshold_frame, text="调整阈值:").pack(side=tk.LEFT)
        threshold_var = tk.DoubleVar(value=threshold)
        ttk.Scale(threshold_frame, from_=0.5, to=0.99, variable=threshold_var,
                  orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT, padx=5)
        threshold_label = ttk.Label(threshold_frame, text=f"{threshold:.2f}")
        threshold_label.pack(side=tk.LEFT)
        threshold_var.trace('w', lambda *a: threshold_label.config(text=f"{threshold_var.get():.2f}"))

        # 进度
        progress = ttk.Progressbar(dialog, length=400, mode='determinate')
        progress.pack(pady=10)
        status_label = ttk.Label(dialog, text="点击'开始筛选'应用规则")
        status_label.pack(pady=5)

        # 结果
        result_var = tk.StringVar(value="")
        ttk.Label(dialog, textvariable=result_var).pack(pady=10)

        # 存储匹配结果和对话框状态
        matched_indices = []
        dialog_closed = [False]

        def on_dialog_close():
            dialog_closed[0] = True
            dialog.destroy()

        dialog.protocol("WM_DELETE_WINDOW", on_dialog_close)

        def safe_after(func):
            if not dialog_closed[0]:
                try:
                    dialog.after(0, func)
                except:
                    pass

        def apply_rules():
            nonlocal matched_indices
            matched_indices = []

            current_threshold = threshold_var.get()
            avg_histogram = np.mean(bad_histograms, axis=0)

            total = len(self.tile_files)
            progress['maximum'] = total
            status_label.config(text="正在筛选...")

            def do_filter():
                nonlocal matched_indices
                import gc

                for i, tile_path in enumerate(self.tile_files):
                    if dialog_closed[0]:
                        return

                    # 跳过已标记保留的
                    if i in self.keep_indices:
                        continue

                    hist = ImageAnalyzer.get_color_histogram(tile_path)
                    if hist is None:
                        continue

                    # 计算与平均直方图的相似度
                    similarity = ImageAnalyzer.histogram_similarity(hist, avg_histogram)

                    # 也计算与每个样本的最大相似度
                    max_similarity = similarity
                    for bad_hist in bad_histograms:
                        sim = ImageAnalyzer.histogram_similarity(hist, bad_hist)
                        if sim > max_similarity:
                            max_similarity = sim

                    if max_similarity >= current_threshold:
                        matched_indices.append(i)

                    if i % 10 == 0:
                        safe_after(lambda v=i: progress.configure(value=v))

                    if i % 100 == 0:
                        gc.collect()

                safe_after(lambda: on_filter_done())

            def on_filter_done():
                progress['value'] = 0
                result_var.set(f"找到 {len(matched_indices)} 个匹配项")
                status_label.config(text="筛选完成！点击'选中匹配项'应用")
                select_btn.config(state=tk.NORMAL)

            threading.Thread(target=do_filter, daemon=True).start()

        def select_matched():
            if not matched_indices:
                messagebox.showinfo("提示", "没有匹配项")
                return

            self._deselect_all()
            for idx in matched_indices:
                self.selected.add(idx)
                if idx in self.tile_labels:
                    self._show_selection_overlay(idx)
            self._update_info()
            dialog.destroy()
            messagebox.showinfo("完成",
                f"已选中 {len(matched_indices)} 个匹配项\n"
                "请检查后点击'删除选中'确认删除")

        # 按钮
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill=tk.X, padx=20, pady=10)

        ttk.Button(btn_frame, text="开始筛选", command=apply_rules).pack(side=tk.LEFT, padx=5)
        select_btn = ttk.Button(btn_frame, text="选中匹配项", command=select_matched, state=tk.DISABLED)
        select_btn.pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="关闭", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)

    def _show_keep_overlay(self, idx: int):
        """显示保留标记（绿色）"""
        row = idx // self.COLS
        col = idx % self.COLS
        cell_width = self.THUMB_SIZE + 6
        cell_height = self.THUMB_SIZE + 20

        x = col * cell_width + 3
        y = row * cell_height + 3

        # 绿色边框
        self.canvas.create_rectangle(
            x, y, x + self.THUMB_SIZE, y + self.THUMB_SIZE,
            fill='', outline='#00ff00', width=3,
            tags=(f"keep_{idx}", "keep")
        )

        # 左上角绿色标记
        self.canvas.create_oval(
            x + 2, y + 2, x + 18, y + 18,
            fill='#00aa00', outline='',
            tags=(f"keep_{idx}", "keep")
        )
        self.canvas.create_text(
            x + 10, y + 10,
            text='★', fill='white', font=('', 8),
            tags=(f"keep_{idx}", "keep")
        )

    def run(self):
        """运行"""
        self.root.mainloop()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='切片审核工具')
    parser.add_argument('--dir', type=str, default=None, help='切片文件夹路径')
    args = parser.parse_args()

    app = TileReviewer(tiles_dir=args.dir)
    app.run()


if __name__ == "__main__":
    main()
