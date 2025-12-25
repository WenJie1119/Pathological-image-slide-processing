"""
病理图像切片处理工具 - 可视化界面
支持单文件和批量处理，提供参数配置和进度显示

优化版本：
- 复用核心 ImageSlicer 逻辑
- 跨平台兼容（Windows/Linux/macOS）
- 支持新参数（输出格式、背景检测方法等）
"""

import os
import sys
import platform
import subprocess
import threading
import queue
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from image_slicer import ImageSlicer


def open_folder(path: str):
    """跨平台打开文件夹"""
    if not os.path.exists(path):
        return False

    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(path)
        elif system == "Darwin":  # macOS
            subprocess.run(["open", path], check=True)
        else:  # Linux
            subprocess.run(["xdg-open", path], check=True)
        return True
    except Exception as e:
        print(f"Error opening folder: {e}")
        return False


class ProcessingThread(threading.Thread):
    """后台处理线程（使用核心 ImageSlicer）"""

    def __init__(self, slicer: ImageSlicer, files: List[Path], output_base: Path,
                 progress_queue: queue.Queue, save_metadata: bool = True):
        super().__init__(daemon=True)
        self.slicer = slicer
        self.files = files
        self.output_base = output_base
        self.progress_queue = progress_queue
        self.save_metadata = save_metadata
        self._stop_event = threading.Event()

    def stop(self):
        """停止处理"""
        self._stop_event.set()

    def is_stopped(self):
        """检查是否已停止"""
        return self._stop_event.is_set()

    def run(self):
        """执行处理"""
        total_files = len(self.files)

        for idx, file_path in enumerate(self.files):
            if self.is_stopped():
                self.progress_queue.put(('stopped', None, None))
                return

            self.progress_queue.put(('file_start', idx, file_path.name))

            try:
                output_dir = self.output_base / file_path.stem

                # 进度回调
                def progress_callback(current, total, saved, skipped):
                    percent = current / total * 100 if total > 0 else 0
                    self.progress_queue.put(('tile_progress', idx, {
                        'current': current,
                        'total': total,
                        'percent': percent,
                        'file_idx': idx,
                        'total_files': total_files
                    }))

                # 使用核心 ImageSlicer 处理
                metadata = self.slicer.generate_tiles(
                    image_path=str(file_path),
                    output_dir=str(output_dir),
                    save_metadata=self.save_metadata,
                    progress_callback=progress_callback,
                    stop_check=self.is_stopped
                )

                if metadata.get('status') == 'stopped':
                    self.progress_queue.put(('stopped', None, None))
                    return

                self.progress_queue.put(('file_done', idx, {
                    'name': file_path.name,
                    'saved_tiles': metadata['saved_tiles'],
                    'total_tiles': metadata['total_tiles_processed'],
                    'skipped': metadata['skipped_background']
                }))

            except Exception as e:
                self.progress_queue.put(('file_error', idx, {
                    'name': file_path.name,
                    'error': str(e)
                }))

        self.progress_queue.put(('all_done', None, None))


class ImageSlicerGUI:
    """病理图像切片处理工具 GUI"""

    SUPPORTED_EXTENSIONS = ['.ndpi', '.svs', '.tiff', '.tif', '.mrxs']

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("病理图像切片处理工具")
        self.root.geometry("950x750")
        self.root.minsize(850, 650)

        # 文件列表
        self.files: List[Path] = []

        # 处理线程
        self.processing_thread: Optional[ProcessingThread] = None
        self.progress_queue = queue.Queue()

        # 创建界面
        self._create_widgets()

        # 启动进度更新
        self._check_progress_queue()

    def _create_widgets(self):
        """创建界面组件"""
        # 创建主Canvas和滚动条
        self.canvas = tk.Canvas(self.root)
        self.scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas, padding="10")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # 跨平台鼠标滚轮绑定
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)  # Windows/macOS
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)    # Linux 向上
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)    # Linux 向下

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.canvas.bind('<Configure>', self._on_canvas_configure)

        main_frame = self.scrollable_frame

        self._create_file_section(main_frame)

        middle_frame = ttk.Frame(main_frame)
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self._create_file_list(middle_frame)
        self._create_params_section(middle_frame)

        self._create_progress_section(main_frame)
        self._create_control_section(main_frame)

    def _on_mousewheel(self, event):
        """鼠标滚轮滚动（跨平台兼容）"""
        # Windows 和 macOS 使用 event.delta
        # Linux 使用 Button-4 (向上) 和 Button-5 (向下)
        if event.delta:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        elif event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")

    def _on_canvas_configure(self, event):
        """Canvas大小变化时调整内部框架宽度"""
        self.canvas.itemconfig(self.canvas.find_withtag("all")[0], width=event.width)

    def _create_file_section(self, parent):
        """创建文件选择区域"""
        frame = ttk.LabelFrame(parent, text="文件选择", padding="10")
        frame.pack(fill=tk.X, pady=(0, 10))

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X)

        ttk.Button(btn_frame, text="选择文件", command=self._select_files).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(btn_frame, text="选择文件夹", command=self._select_folder).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(btn_frame, text="清空列表", command=self._clear_files).pack(side=tk.LEFT)

        output_frame = ttk.Frame(frame)
        output_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Label(output_frame, text="输出目录:").pack(side=tk.LEFT)
        self.output_var = tk.StringVar(value=str(Path(__file__).parent.parent / "output" / "tiles"))
        self.output_entry = ttk.Entry(output_frame, textvariable=self.output_var, width=60)
        self.output_entry.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        ttk.Button(output_frame, text="浏览", command=self._select_output_dir).pack(side=tk.LEFT)

    def _create_file_list(self, parent):
        """创建文件列表区域"""
        frame = ttk.LabelFrame(parent, text="待处理文件", padding="10")
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        list_frame = ttk.Frame(frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        columns = ('文件名', '大小', '状态')
        self.file_tree = ttk.Treeview(list_frame, columns=columns, show='headings',
                                       yscrollcommand=scrollbar.set)

        self.file_tree.heading('文件名', text='文件名')
        self.file_tree.heading('大小', text='大小')
        self.file_tree.heading('状态', text='状态')

        self.file_tree.column('文件名', width=200)
        self.file_tree.column('大小', width=80)
        self.file_tree.column('状态', width=100)

        self.file_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.file_tree.yview)

        self.file_count_var = tk.StringVar(value="共 0 个文件")
        ttk.Label(frame, textvariable=self.file_count_var).pack(pady=(5, 0))

    def _create_params_section(self, parent):
        """创建参数配置区域"""
        frame = ttk.LabelFrame(parent, text="处理参数", padding="10")
        frame.pack(side=tk.RIGHT, fill=tk.Y)

        row = 0
        # 切片大小
        ttk.Label(frame, text="切片大小 (像素):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.tile_size_var = tk.IntVar(value=512)
        ttk.Combobox(frame, textvariable=self.tile_size_var,
                     values=[256, 512, 1024], width=10, state='readonly').grid(row=row, column=1, sticky=tk.W, pady=5)

        row += 1
        # 重叠像素
        ttk.Label(frame, text="重叠像素:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.overlap_var = tk.IntVar(value=64)
        ttk.Combobox(frame, textvariable=self.overlap_var,
                     values=[0, 32, 64, 128, 256], width=10, state='readonly').grid(row=row, column=1, sticky=tk.W, pady=5)

        row += 1
        # 目标放大倍数
        ttk.Label(frame, text="目标放大倍数:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.magnification_var = tk.DoubleVar(value=20.0)
        ttk.Combobox(frame, textvariable=self.magnification_var,
                     values=[5.0, 10.0, 20.0, 40.0], width=10, state='readonly').grid(row=row, column=1, sticky=tk.W, pady=5)

        row += 1
        # 输出格式
        ttk.Label(frame, text="输出格式:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.format_var = tk.StringVar(value="png")
        format_frame = ttk.Frame(frame)
        format_frame.grid(row=row, column=1, columnspan=2, sticky=tk.W, pady=5)
        ttk.Radiobutton(format_frame, text="PNG", variable=self.format_var, value="png").pack(side=tk.LEFT)
        ttk.Radiobutton(format_frame, text="JPEG", variable=self.format_var, value="jpg").pack(side=tk.LEFT, padx=10)

        row += 1
        # JPEG 质量
        ttk.Label(frame, text="JPEG质量:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.jpeg_quality_var = tk.IntVar(value=95)
        quality_frame = ttk.Frame(frame)
        quality_frame.grid(row=row, column=1, columnspan=2, sticky=tk.W, pady=5)
        ttk.Scale(quality_frame, from_=50, to=100, variable=self.jpeg_quality_var,
                  orient=tk.HORIZONTAL, length=80).pack(side=tk.LEFT)
        self.quality_label = ttk.Label(quality_frame, text="95")
        self.quality_label.pack(side=tk.LEFT, padx=5)
        self.jpeg_quality_var.trace('w', lambda *a: self.quality_label.config(text=str(self.jpeg_quality_var.get())))

        row += 1
        # 背景检测方法
        ttk.Label(frame, text="背景检测:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.bg_method_var = tk.StringVar(value="otsu")
        bg_frame = ttk.Frame(frame)
        bg_frame.grid(row=row, column=1, columnspan=2, sticky=tk.W, pady=5)
        ttk.Radiobutton(bg_frame, text="Otsu自动", variable=self.bg_method_var, value="otsu").pack(side=tk.LEFT)
        ttk.Radiobutton(bg_frame, text="固定阈值", variable=self.bg_method_var, value="threshold").pack(side=tk.LEFT, padx=10)

        row += 1
        # 背景阈值
        ttk.Label(frame, text="背景阈值:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.bg_threshold_var = tk.DoubleVar(value=0.8)
        threshold_frame = ttk.Frame(frame)
        threshold_frame.grid(row=row, column=1, columnspan=2, sticky=tk.W, pady=5)
        ttk.Scale(threshold_frame, from_=0.5, to=1.0, variable=self.bg_threshold_var,
                  orient=tk.HORIZONTAL, length=80).pack(side=tk.LEFT)
        self.bg_label = ttk.Label(threshold_frame, text="0.80")
        self.bg_label.pack(side=tk.LEFT, padx=5)
        self.bg_threshold_var.trace('w', lambda *a: self.bg_label.config(text=f"{self.bg_threshold_var.get():.2f}"))

        row += 1
        # 最小组织占比
        ttk.Label(frame, text="最小组织占比:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.tissue_ratio_var = tk.DoubleVar(value=0.1)
        tissue_frame = ttk.Frame(frame)
        tissue_frame.grid(row=row, column=1, columnspan=2, sticky=tk.W, pady=5)
        ttk.Scale(tissue_frame, from_=0.0, to=0.5, variable=self.tissue_ratio_var,
                  orient=tk.HORIZONTAL, length=80).pack(side=tk.LEFT)
        self.tissue_label = ttk.Label(tissue_frame, text="0.10")
        self.tissue_label.pack(side=tk.LEFT, padx=5)
        self.tissue_ratio_var.trace('w', lambda *a: self.tissue_label.config(text=f"{self.tissue_ratio_var.get():.2f}"))

        row += 1
        # 工作线程数
        ttk.Label(frame, text="并行线程:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.workers_var = tk.IntVar(value=4)
        ttk.Combobox(frame, textvariable=self.workers_var,
                     values=[1, 2, 4, 8], width=10, state='readonly').grid(row=row, column=1, sticky=tk.W, pady=5)

        row += 1
        # 保存元数据
        self.save_metadata_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="保存元数据JSON", variable=self.save_metadata_var).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=10)

        row += 1
        # 参数说明
        info_text = """参数说明:
• Otsu自动: 自动计算最佳阈值
• JPEG格式: 文件更小，速度更快
• 并行线程: 值越大处理越快"""

        ttk.Label(frame, text=info_text, justify=tk.LEFT,
                  foreground='gray', font=('', 9)).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=10)

    def _create_progress_section(self, parent):
        """创建进度显示区域"""
        frame = ttk.LabelFrame(parent, text="处理进度", padding="10")
        frame.pack(fill=tk.X, pady=(0, 10))

        file_frame = ttk.Frame(frame)
        file_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(file_frame, text="当前文件:").pack(side=tk.LEFT)
        self.current_file_var = tk.StringVar(value="-")
        ttk.Label(file_frame, textvariable=self.current_file_var, foreground='blue').pack(side=tk.LEFT, padx=10)

        overall_frame = ttk.Frame(frame)
        overall_frame.pack(fill=tk.X, pady=5)
        ttk.Label(overall_frame, text="总体进度:").pack(side=tk.LEFT)
        self.overall_progress = ttk.Progressbar(overall_frame, length=400, mode='determinate')
        self.overall_progress.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        self.overall_label = ttk.Label(overall_frame, text="0/0")
        self.overall_label.pack(side=tk.LEFT)

        current_frame = ttk.Frame(frame)
        current_frame.pack(fill=tk.X, pady=5)
        ttk.Label(current_frame, text="文件进度:").pack(side=tk.LEFT)
        self.file_progress = ttk.Progressbar(current_frame, length=400, mode='determinate')
        self.file_progress.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        self.file_progress_label = ttk.Label(current_frame, text="0%")
        self.file_progress_label.pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(frame, textvariable=self.status_var, foreground='green').pack(pady=5)

        log_frame = ttk.Frame(frame)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        log_scroll = ttk.Scrollbar(log_frame)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.log_text = tk.Text(log_frame, height=6, state=tk.DISABLED, yscrollcommand=log_scroll.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.config(command=self.log_text.yview)

    def _create_control_section(self, parent):
        """创建控制按钮区域"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X)

        self.start_btn = ttk.Button(frame, text="开始处理", command=self._start_processing)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.stop_btn = ttk.Button(frame, text="停止处理", command=self._stop_processing, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(frame, text="打开输出目录", command=self._open_output_dir).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(frame, text="审核切片", command=self._open_reviewer).pack(side=tk.LEFT)
        ttk.Button(frame, text="退出", command=self.root.quit).pack(side=tk.RIGHT)

    def _select_files(self):
        """选择文件"""
        filetypes = [
            ('病理图像文件', ' '.join(f'*{ext}' for ext in self.SUPPORTED_EXTENSIONS)),
            ('所有文件', '*.*')
        ]

        files = filedialog.askopenfilenames(title="选择病理图像文件", filetypes=filetypes)

        if files:
            for f in files:
                self._add_file(Path(f))

    def _select_folder(self):
        """选择文件夹"""
        folder = filedialog.askdirectory(title="选择包含病理图像的文件夹")

        if folder:
            folder_path = Path(folder)
            count = 0
            for ext in self.SUPPORTED_EXTENSIONS:
                for f in folder_path.glob(f'*{ext}'):
                    self._add_file(f)
                    count += 1

            if count == 0:
                messagebox.showinfo("提示", f"文件夹中没有找到支持的图像文件\n支持格式: {', '.join(self.SUPPORTED_EXTENSIONS)}")

    def _add_file(self, file_path: Path):
        """添加文件到列表"""
        if file_path not in self.files and file_path.exists():
            self.files.append(file_path)

            size = file_path.stat().st_size
            if size > 1024 * 1024 * 1024:
                size_str = f"{size / (1024*1024*1024):.2f} GB"
            elif size > 1024 * 1024:
                size_str = f"{size / (1024*1024):.2f} MB"
            else:
                size_str = f"{size / 1024:.2f} KB"

            self.file_tree.insert('', tk.END, values=(file_path.name, size_str, "等待"))
            self._update_file_count()

    def _clear_files(self):
        """清空文件列表"""
        self.files.clear()
        for item in self.file_tree.get_children():
            self.file_tree.delete(item)
        self._update_file_count()

    def _update_file_count(self):
        """更新文件计数"""
        self.file_count_var.set(f"共 {len(self.files)} 个文件")

    def _select_output_dir(self):
        """选择输出目录"""
        folder = filedialog.askdirectory(title="选择输出目录")
        if folder:
            self.output_var.set(folder)

    def _log(self, message: str):
        """添加日志"""
        self.log_text.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _start_processing(self):
        """开始处理"""
        if not self.files:
            messagebox.showwarning("警告", "请先选择要处理的文件")
            return

        output_dir = Path(self.output_var.get())
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True)
            except Exception as e:
                messagebox.showerror("错误", f"无法创建输出目录: {e}")
                return

        # 创建切片器（使用新参数）
        slicer = ImageSlicer(
            tile_size=self.tile_size_var.get(),
            overlap=self.overlap_var.get(),
            target_magnification=self.magnification_var.get(),
            background_threshold=self.bg_threshold_var.get(),
            min_tissue_ratio=self.tissue_ratio_var.get(),
            output_format=self.format_var.get(),
            jpeg_quality=self.jpeg_quality_var.get(),
            num_workers=self.workers_var.get(),
            background_method=self.bg_method_var.get()
        )

        self.overall_progress['value'] = 0
        self.file_progress['value'] = 0
        self.overall_progress['maximum'] = len(self.files)

        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set("处理中...")

        for item in self.file_tree.get_children():
            values = self.file_tree.item(item, 'values')
            self.file_tree.item(item, values=(values[0], values[1], "等待"))

        self._log(f"开始处理 {len(self.files)} 个文件")
        self._log(f"参数: 切片={self.tile_size_var.get()}, 格式={self.format_var.get().upper()}, "
                  f"背景检测={self.bg_method_var.get()}")

        self.processing_thread = ProcessingThread(
            slicer=slicer,
            files=self.files.copy(),
            output_base=output_dir,
            progress_queue=self.progress_queue,
            save_metadata=self.save_metadata_var.get()
        )
        self.processing_thread.start()

    def _stop_processing(self):
        """停止处理"""
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.stop()
            self.status_var.set("正在停止...")
            self._log("用户请求停止处理")

    def _check_progress_queue(self):
        """检查进度队列"""
        try:
            while True:
                msg_type, idx, data = self.progress_queue.get_nowait()

                if msg_type == 'file_start':
                    self.current_file_var.set(data)
                    self._update_tree_status(idx, "处理中")
                    self._log(f"开始处理: {data}")

                elif msg_type == 'tile_progress':
                    percent = data['percent']
                    self.file_progress['value'] = percent
                    self.file_progress_label.config(text=f"{percent:.1f}%")

                elif msg_type == 'file_done':
                    self.overall_progress['value'] = idx + 1
                    self.overall_label.config(text=f"{idx + 1}/{len(self.files)}")
                    self._update_tree_status(idx, f"完成 ({data['saved_tiles']})")
                    self._log(f"完成: {data['name']} - 保存 {data['saved_tiles']} 个切片, "
                              f"跳过 {data['skipped']} 个背景")
                    self.file_progress['value'] = 0

                elif msg_type == 'file_error':
                    self._update_tree_status(idx, "错误")
                    self._log(f"错误: {data['name']} - {data['error']}")
                    self.overall_progress['value'] = idx + 1
                    self.overall_label.config(text=f"{idx + 1}/{len(self.files)}")

                elif msg_type == 'all_done':
                    self._processing_complete()

                elif msg_type == 'stopped':
                    self._processing_stopped()

        except queue.Empty:
            pass

        self.root.after(100, self._check_progress_queue)

    def _update_tree_status(self, idx: int, status: str):
        """更新文件列表状态"""
        items = self.file_tree.get_children()
        if 0 <= idx < len(items):
            item = items[idx]
            values = self.file_tree.item(item, 'values')
            self.file_tree.item(item, values=(values[0], values[1], status))

    def _processing_complete(self):
        """处理完成"""
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("处理完成")
        self.current_file_var.set("-")
        self._log("所有文件处理完成!")
        messagebox.showinfo("完成", "所有文件处理完成!")

    def _processing_stopped(self):
        """处理被停止"""
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("已停止")
        self.current_file_var.set("-")
        self._log("处理已停止")

    def _open_output_dir(self):
        """打开输出目录"""
        output_dir = self.output_var.get()
        if not open_folder(output_dir):
            messagebox.showwarning("警告", "输出目录不存在或无法打开")

    def _open_reviewer(self):
        """打开切片审核工具"""
        output_base = self.output_var.get()
        initial_dir = output_base if os.path.exists(output_base) else None

        folder = filedialog.askdirectory(
            title="选择要审核的切片文件夹",
            initialdir=initial_dir
        )

        if folder:
            script_path = Path(__file__).parent / "tile_reviewer.py"
            subprocess.Popen([sys.executable, str(script_path), "--dir", folder])

    def run(self):
        """运行GUI"""
        self.root.mainloop()


def main():
    """主函数"""
    app = ImageSlicerGUI()
    app.run()


if __name__ == "__main__":
    main()
