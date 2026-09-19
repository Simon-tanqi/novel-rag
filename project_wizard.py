"""
project_wizard.py — 新建小说RAG项目向导窗口
提供创建新项目的图形化配置界面

流程: 配置 → 清洗 → 切片 → 向量化 → 完成
"""
import os
import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox
from typing import Optional, List, Callable

from project_manager import ProjectManager
from step1_clean import clean_file, get_available_rules
from step2_split_embed import build_vector_index_from_file
from utils import get_project_dir, get_vector_db_path


class ProjectWizard(ctk.CTkToplevel):
    """新建项目向导窗口"""

    def __init__(self, master, project_manager: ProjectManager, config_manager=None, on_complete: Optional[Callable] = None):
        """
        初始化项目向导

        Args:
            master: 主窗口
            project_manager: 项目管理器实例
            on_complete: 完成回调
        """
        super().__init__(master)
        self.project_manager = project_manager
        self.config_manager = config_manager
        self.on_complete = on_complete

        self.title("新建小说RAG项目")
        self.geometry("650x600")
        self.minsize(600, 550)
        self.transient(master)
        self.grab_set()

        # 状态变量
        self.project_name = ctk.StringVar()
        self.source_path = ctk.StringVar()
        self.chunk_size = ctk.IntVar(value=672)
        self.overlap = ctk.IntVar(value=50)
        self.is_processing = False

        # 清洗规则复选框变量
        self.rule_vars = {}
        self.custom_words_text = ctk.StringVar()

        # 进度相关
        self.progress = ctk.DoubleVar(value=0)
        self.status_text = ctk.StringVar(value="")

        self._create_widgets()

    def _create_widgets(self):
        """创建界面组件"""
        # 主容器
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # 标题
        ctk.CTkLabel(
            main_frame,
            text="➕ 新建小说RAG项目",
            font=("Microsoft YaHei UI", 18, "bold")
        ).pack(anchor="w", pady=(0, 20))

        # 基本信息区
        self._create_basic_info_section(main_frame)

        # 清洗规则区
        self._create_clean_rules_section(main_frame)

        # 切片参数区
        self._create_chunk_params_section(main_frame)

        # 进度显示区
        self._create_progress_section(main_frame)

        # 按钮区
        self._create_button_section(main_frame)

    def _create_basic_info_section(self, parent):
        """创建基本信息区"""
        section = ctk.CTkFrame(parent, fg_color="#2C2C2C", corner_radius=10)
        section.pack(fill="x", pady=(0, 15))

        inner = ctk.CTkFrame(section, fg_color="transparent")
        inner.pack(fill="x", padx=15, pady=15)

        # 小说名称
        ctk.CTkLabel(
            inner,
            text="📚 小说名称 *:",
            font=("Microsoft YaHei UI", 12, "bold")
        ).grid(row=0, column=0, sticky="w", padx=(0, 10), pady=5)

        name_entry = ctk.CTkEntry(
            inner,
            textvariable=self.project_name,
            width=400,
            placeholder_text="请输入小说名称，如：三体"
        )
        name_entry.grid(row=0, column=1, columnspan=3, sticky="w", pady=5)

        # 源文件选择
        ctk.CTkLabel(
            inner,
            text="📄 源文件 *:",
            font=("Microsoft YaHei UI", 12, "bold")
        ).grid(row=1, column=0, sticky="w", padx=(0, 10), pady=5)

        path_entry = ctk.CTkEntry(
            inner,
            textvariable=self.source_path,
            width=350,
            placeholder_text="请选择 .txt 文件"
        )
        path_entry.grid(row=1, column=1, sticky="w", pady=5)

        browse_btn = ctk.CTkButton(
            inner,
            text="浏览...",
            width=80,
            command=self._browse_source_file
        )
        browse_btn.grid(row=1, column=2, padx=(5, 0), pady=5)

    def _create_clean_rules_section(self, parent):
        """创建清洗规则区"""
        section = ctk.CTkFrame(parent, fg_color="#2C2C2C", corner_radius=10)
        section.pack(fill="x", pady=(0, 15))

        inner = ctk.CTkFrame(section, fg_color="transparent")
        inner.pack(fill="x", padx=15, pady=15)

        ctk.CTkLabel(
            inner,
            text="🧹 清洗规则:",
            font=("Microsoft YaHei UI", 12, "bold")
        ).grid(row=0, column=0, columnspan=6, sticky="w", pady=(0, 10))

        # 创建清洗规则复选框
        rules = get_available_rules()
        for i, rule in enumerate(rules):
            var = ctk.BooleanVar(value=True)
            self.rule_vars[rule] = var
            cb = ctk.CTkCheckBox(
                inner,
                text=rule,
                variable=var,
                font=("Microsoft YaHei UI", 11)
            )
            cb.grid(row=i // 3 + 1, column=i % 3, sticky="w", padx=(0, 20), pady=5)

        # 自定义关键词
        custom_row = len(rules) // 3 + 2
        ctk.CTkLabel(
            inner,
            text="📝 自定义脏数据关键词:",
            font=("Microsoft YaHei UI", 11)
        ).grid(row=custom_row, column=0, columnspan=2, sticky="w", pady=(10, 5))

        custom_entry = ctk.CTkEntry(
            inner,
            textvariable=self.custom_words_text,
            width=480,
            placeholder_text="多个关键词用逗号分隔，如：手打,支持正版,最新章节"
        )
        custom_entry.grid(row=custom_row, column=2, columnspan=4, sticky="w", pady=(10, 5))

    def _create_chunk_params_section(self, parent):
        """创建切片参数区"""
        section = ctk.CTkFrame(parent, fg_color="#2C2C2C", corner_radius=10)
        section.pack(fill="x", pady=(0, 15))

        inner = ctk.CTkFrame(section, fg_color="transparent")
        inner.pack(fill="x", padx=15, pady=15)

        ctk.CTkLabel(
            inner,
            text="⚙ 切片参数:",
            font=("Microsoft YaHei UI", 12, "bold")
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))

        # 切片大小
        ctk.CTkLabel(inner, text="切片大小:", font=("Microsoft YaHei UI", 11)).grid(
            row=1, column=0, sticky="w", padx=(0, 10), pady=5
        )
        chunk_size_entry = ctk.CTkEntry(
            inner,
            textvariable=self.chunk_size,
            width=80
        )
        chunk_size_entry.grid(row=1, column=1, sticky="w", pady=5)
        ctk.CTkLabel(
            inner,
            text="字符 (建议 300-800)",
            font=("Microsoft YaHei UI", 10),
            text_color="gray"
        ).grid(row=1, column=2, sticky="w", padx=(5, 0), pady=5)

        # 重叠长度
        ctk.CTkLabel(inner, text="重叠长度:", font=("Microsoft YaHei UI", 11)).grid(
            row=1, column=3, sticky="w", padx=(20, 10), pady=5
        )
        overlap_entry = ctk.CTkEntry(
            inner,
            textvariable=self.overlap,
            width=80
        )
        overlap_entry.grid(row=1, column=4, sticky="w", pady=5)
        ctk.CTkLabel(
            inner,
            text="字符 (建议 30-100)",
            font=("Microsoft YaHei UI", 10),
            text_color="gray"
        ).grid(row=1, column=5, sticky="w", padx=(5, 0), pady=5)

    def _create_progress_section(self, parent):
        """创建进度显示区"""
        self.progress_frame = ctk.CTkFrame(parent, fg_color="transparent")
        self.progress_frame.pack(fill="x", pady=(0, 15))

        # 进度条
        self.progress_bar = ctk.CTkProgressBar(
            self.progress_frame,
            height=12,
            mode="determinate"
        )
        self.progress_bar.pack(fill="x", pady=(0, 5))
        self.progress_bar.set(0)

        # 状态文字
        self.status_label = ctk.CTkLabel(
            self.progress_frame,
            textvariable=self.status_text,
            font=("Microsoft YaHei UI", 11),
            text_color="gray"
        )
        self.status_label.pack(anchor="w")

    def _create_button_section(self, parent):
        """创建按钮区"""
        btn_frame = ctk.CTkFrame(parent, fg_color="transparent")
        btn_frame.pack(fill="x", pady=(10, 0))

        ctk.CTkButton(
            btn_frame,
            text="取消",
            width=100,
            height=40,
            fg_color="#4A4A4A",
            hover_color="#3A3A3A",
            command=self._on_cancel
        ).pack(side="right", padx=(10, 0))

        self.create_btn = ctk.CTkButton(
            btn_frame,
            text="🚀 创建项目",
            width=150,
            height=40,
            font=("Microsoft YaHei UI", 12, "bold"),
            fg_color="#2E8B57",
            hover_color="#1E6B3E",
            command=self._on_create
        )
        self.create_btn.pack(side="right")

    def _browse_source_file(self):
        """浏览选择源文件"""
        filetypes = [
            ("文本文件", "*.txt"),
            ("所有文件", "*.*")
        ]
        file_path = filedialog.askopenfilename(
            parent=self,
            title="选择小说源文件",
            filetypes=filetypes
        )
        if file_path:
            self.source_path.set(file_path)

    def _on_create(self):
        """点击创建按钮"""
        # 验证输入
        name = self.project_name.get().strip()
        source = self.source_path.get().strip()

        if not name:
            messagebox.showwarning("提示", "请输入小说名称！", parent=self)
            return

        if not source:
            messagebox.showwarning("提示", "请选择源文件！", parent=self)
            return

        if not os.path.exists(source):
            messagebox.showerror("错误", "源文件不存在！", parent=self)
            return

        # 检查项目名是否已存在
        if self.project_manager.project_name_exists(name):
            if not messagebox.askyesno(
                "确认",
                f"项目「{name}」已存在，是否覆盖？",
                parent=self
            ):
                return

        # 获取参数
        chunk_size = self.chunk_size.get()
        overlap = self.overlap.get()

        # 获取选中的清洗规则
        selected_rules = [
            rule for rule, var in self.rule_vars.items()
            if var.get()
        ]

        # 获取自定义关键词
        custom_words_text = self.custom_words_text.get().strip()
        custom_words = []
        if custom_words_text:
            custom_words = [w.strip() for w in custom_words_text.split(",") if w.strip()]

        # 禁用按钮
        self.is_processing = True
        self.create_btn.configure(state="disabled", text="⏳ 处理中...")

        # 在后台线程执行创建流程
        thread = threading.Thread(
            target=self._run_create_pipeline,
            args=(name, source, chunk_size, overlap, selected_rules, custom_words)
        )
        thread.daemon = True
        thread.start()

    def _run_create_pipeline(
        self,
        name: str,
        source_path: str,
        chunk_size: int,
        overlap: int,
        rules: List[str],
        custom_words: List[str]
    ):
        """
        运行创建流程（后台线程）

        Args:
            name: 项目名称
            source_path: 源文件路径
            chunk_size: 切片大小
            overlap: 重叠长度
            rules: 清洗规则
            custom_words: 自定义关键词
        """
        try:
            # 步骤1: 创建项目配置
            self._update_progress(0, "正在创建项目配置...")
            project_config = self.project_manager.create_project(
                name=name,
                source_path=source_path,
                chunk_size=chunk_size,
                overlap=overlap,
                clean_rules=rules,
                custom_dirty_words=custom_words
            )

            project_id = project_config["id"]
            project_dir = get_project_dir(name)

            # 复制源文件到项目目录
            import shutil
            source_dir = project_dir / "source"
            source_dir.mkdir(exist_ok=True)
            dest_path = source_dir / os.path.basename(source_path)
            shutil.copy2(source_path, str(dest_path))

            # 更新配置中的源文件路径
            self.project_manager.update_project(
                project_id,
                {"source_path": str(dest_path)}
            )

            # 步骤2: 清洗
            self._update_progress(10, "正在清洗文本...")
            cleaned_path = str(project_dir / "cleaned" / "cleaned.txt")

            def clean_progress(step, total, msg, pct=0):
                overall = 10 + (pct / 100) * 40  # 10% - 50%
                self._update_progress(int(overall), f"清洗: {msg}")

            success = clean_file(
                input_path=str(dest_path),
                output_path=cleaned_path,
                rules=rules,
                custom_words=custom_words,
                progress_callback=clean_progress
            )

            if not success:
                raise Exception("文本清洗失败")

            self.project_manager.update_project(
                project_id,
                {
                    "cleaned_path": cleaned_path,
                    "status": "cleaned"
                }
            )

            # 步骤3: 切片与向量化
            self._update_progress(50, "正在切片与向量化...")
            vector_db_path = get_vector_db_path(name)

            # 获取嵌入模型配置：本地模型目录 或 HF 模型名（如 BAAI/bge-small-zh-v1.5）
            # 注意：显式指定嵌入模型却加载不到时，step2 直接返回失败（不产出半成品索引）；
            # 仅当配置置空、用户主动选择纯关键词模式时才只落 metadata.json。因此此处
            # 不能预设 status=ready，必须按落盘结果（embeddings.npy 是否存在）判定。
            embedding_model_path = ""
            if self.config_manager:
                embedding_model_path = self.config_manager.get("embedding_model_path", "")

            def embed_progress(step, total, msg, pct=0):
                overall = 50 + (pct / 100) * 45  # 50% - 95%
                self._update_progress(int(overall), f"向量化: {msg}")

            success = build_vector_index_from_file(
                input_file=cleaned_path,
                chunk_size=chunk_size,
                overlap=overlap,
                output_dir=vector_db_path,
                embedding_model_path=embedding_model_path if embedding_model_path else None,
                progress_callback=embed_progress,
                book_id=project_id,
                book_title=name,
            )

            if not success:
                raise Exception("向量化失败")

            # 真实落盘校验：只有 embeddings.npy 落地才算 ready。
            # 否则（用户置空模型走纯关键词模式）落 keyword_only，
            # 避免「状态显示已就绪、检索却只走关键词」的静默降级。
            has_vectors = os.path.exists(os.path.join(vector_db_path, "embeddings.npy"))
            real_status = "ready" if has_vectors else "keyword_only"

            # 更新项目状态
            self.project_manager.update_project(
                project_id,
                {
                    "vector_db_path": vector_db_path,
                    "status": real_status
                }
            )

            # 完成
            if has_vectors:
                self._update_progress(100, "✅ 项目创建完成！")
            else:
                self._update_progress(
                    100,
                    "⚠ 未生成向量（嵌入模型不可用），已按关键词模式创建；"
                    "请用项目虚拟环境重新向量化",
                )
            self.after(2000, self._on_create_success, project_id)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._update_progress(-1, f"❌ 创建失败: {str(e)}")
            self.after(2000, self._on_create_failed)

    def _update_progress(self, percent: int, message: str):
        """
        更新进度（线程安全）

        Args:
            percent: 进度百分比 (-1表示失败)
            message: 状态消息
        """
        self.after(0, self._update_progress_ui, percent, message)

    def _update_progress_ui(self, percent: int, message: str):
        """更新进度UI"""
        if percent < 0:
            self.progress_bar.set(0)
            self.status_label.configure(text=message, text_color="#E81123")
        else:
            self.progress_bar.set(percent / 100.0)
            self.status_label.configure(text=message, text_color="gray")
        self.status_text.set(message)

    def _on_create_success(self, project_id: str):
        """创建成功回调"""
        self.is_processing = False
        self.create_btn.configure(state="normal", text="🚀 创建项目")

        project = self.project_manager.get_project(project_id)
        if project:
            # 设置为当前项目
            self.project_manager.set_current_project(project_id)

            if self.on_complete:
                self.on_complete(project)

        self.destroy()

    def _on_create_failed(self):
        """创建失败回调"""
        self.is_processing = False
        self.create_btn.configure(state="normal", text="🚀 创建项目")
        messagebox.showerror("错误", "项目创建失败，请查看日志！", parent=self)

    def _on_cancel(self):
        """取消"""
        if not self.is_processing:
            self.destroy()
