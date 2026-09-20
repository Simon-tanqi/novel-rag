"""
main.py — 小说RAG系统主窗口
整合项目管理、对话界面、设置等功能

启动方式: python main.py
"""
import os
import re
import time
import threading
import traceback
from datetime import datetime

# 优先尝试导入 GUI 依赖。测试/CI 环境可能没装 customtkinter，
# 不影响纯逻辑测试（如 _api_timeout_check 标志位逻辑）。
try:
    import customtkinter as ctk
    import tkinter.messagebox
    import tkinter.filedialog
    _GUI_AVAILABLE = True
except ImportError:
    ctk = None  # type: ignore
    _GUI_AVAILABLE = False

from config_manager import ConfigManager
from project_manager import ProjectManager
from api_client import APIClient
from rag_retriever import RAGRetriever
from query_rewriter import has_meta_terms, rewrite_query
from chat_logger import ChatLogger
from message_bubble import MessageBubble
from settings_window import SettingsWindow
from project_wizard import ProjectWizard
from utils import get_root_dir, get_compute_device, resolve_rerank_policy

if _GUI_AVAILABLE:
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

# 最大对话轮数
MAX_TURNS = 15


def assemble_context(hits, top_k=None):
    """组装喂给 LLM 的检索上下文（分层召回的最后一步）。

    优先级：命中子块 > 父块 > 邻居；按章节分组，组内按优先级排序。
    - 无分层扩展字段（老向量库 / 未扩展）时，保持原始逐段输出：
      每命中一个块输出一段 ``[来源:章节]\\n正文``，段间以 ``\\n\\n---\\n\\n`` 连接。
    - 存在父块/邻居时，按章节聚合为块，章节内先命中、再父块、再邻居。
    - 同章节内按「文本包含关系」去重：父块由子块原文拼接而成，若父块已输出，
      其覆盖的子块不再重复输出；新项更长且包含已保留项时就地替换，避免
      父子块文本重复导致上下文膨胀。

    Args:
        hits: retrieve()/retrieve_multi() 返回的文档列表
        top_k: 兼容占位参数（未使用，保留以便调用方传参扩展）

    Returns:
        拼装好的上下文字符串
    """
    if not hits:
        return "未找到相关的原文片段。"

    def _kind(doc):
        if doc.get("is_parent"):
            return "parent"
        if doc.get("is_neighbor"):
            return "neighbor"
        return "hit"

    # 无分层扩展 → 原始逐段输出（不做章节聚合）
    if not any(_kind(h) != "hit" for h in hits):
        return "\n\n---\n\n".join(
            f"[来源:{h.get('chapter', '')}]\n{h.get('text', '')}" for h in hits
        )

    order = {"hit": 0, "parent": 1, "neighbor": 2}
    label = {"hit": "来源", "parent": "父块", "neighbor": "邻居"}

    buckets = {}
    chapter_seq = []
    for h in hits:
        chapter = h.get("chapter") or "未知"
        if chapter not in buckets:
            buckets[chapter] = []
            chapter_seq.append(chapter)
        buckets[chapter].append(h)

    blocks = []
    for chapter in chapter_seq:
        items = sorted(buckets[chapter], key=lambda d: order[_kind(d)])
        # 同章节内做「文本包含关系」去重：
        # - 父块由同章连续子块原文拼接而成，必然包含其子块文本 —— 若已保留
        #   更长的父块，则被包含的子块不再重复输出（消除命中/父块重复膨胀）；
        # - 反之若新项更长且包含已保留项，则就地替换（位次不变），以更全的
        #   父块文本为准，避免上下文被切碎重复。
        kept = []  # [(text, item), ...]
        for item in items:
            text = item.get("text", "")
            if not text:
                continue
            redundant = False
            for i, (kept_text, _) in enumerate(kept):
                if text == kept_text or text in kept_text:
                    redundant = True
                    break
                if kept_text in text:
                    kept[i] = (text, item)  # 就地替换为更完整的文本
                    redundant = True
                    break
            if redundant:
                continue
            kept.append((text, item))
        lines = [f"[{label[_kind(it)]}:{chapter}]\n{t}" for t, it in kept]
        if lines:
            blocks.append("\n\n".join(lines))
    return "\n\n---\n\n".join(blocks) if blocks else "未找到相关的原文片段。"


class NovelRAGApp(ctk.CTk):
    """小说RAG系统主应用"""

    def __init__(self):
        super().__init__()

        # 管理器
        self.config_manager = ConfigManager()
        self.project_manager = ProjectManager()

        # 当前状态
        self.current_project = None
        self.current_device = get_compute_device()  # GPU 加速可用时显示在 sidebar
        self.chat_logger = ChatLogger()
        self.messages = []
        self.turn_count = 0
        self.api_timeout = False
        self.api_start_time = 0
        # 向量检索提示：首次出现时插入提示气泡，避免反复刷屏（仅提示一次/会话）
        self._vector_mode_warned = False

        # 加载当前项目
        self._load_current_project()

        # 窗口设置
        project_name = self.current_project["name"] if self.current_project else "未选择"
        self.title(f"📚 小说RAG系统 - 当前项目: {project_name}")
        self.geometry("1000x750")
        self.minsize(700, 550)

        # 创建界面
        self._create_widgets()

        # 如果没有项目，显示欢迎提示
        if not self.current_project:
            self._show_welcome_message()

        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    # ===================== 项目管理 =====================

    def _detect_device(self) -> str:
        """探测当前计算设备（cuda/cpu）— 不跳慢，只查 torch.is_available"""
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except (ImportError, AttributeError, OSError):
            return "cpu"

    def _load_current_project(self):
        """加载当前项目"""

    def set_project(self, project_id: str):
        """切换当前项目"""
        project = self.project_manager.get_project(project_id)
        if not project:
            return

        self.current_project = project
        self.chat_logger.set_project(project["name"])
        self.turn_count = 0

        # 更新标题
        self.title(f"📚 小说RAG系统 - 当前项目: {project['name']}")

        # 更新侧边栏状态
        self._update_sidebar_project_info()

        # 清空聊天并显示欢迎
        self._clear_chat_area(show_welcome=True)

        self.status_label.configure(text=f"✓ 已切换到项目: {project['name']}")

    def _detect_device(self) -> str:
        """检测推理设备（cuda 可用返回 cuda，否则 cpu）"""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    def _update_sidebar_project_info(self):
        """更新侧边栏项目信息（所见即所得）"""
        if not self.current_project:
            self.project_info_label.configure(text="📚 当前项目: 未选择")
            return

        project = self.current_project
        pid = project.get("id", "")
        runtime = self.project_manager.get_project_runtime(pid)
        status = runtime.get("status", "unknown")
        chunk_count = runtime.get("chunk_count", 0)
        has_emb = runtime.get("has_embeddings", False)
        is_external = runtime.get("is_external", False)
        emb_mb = runtime.get("embeddings_size_mb", 0.0)
        vector_path = runtime.get("vector_db_path", "")

        # Embedding 模型 / 设备
        cfg = self.config_manager
        emb_model = (cfg.get("embedding_model_path") or "").strip() or "(未配置)"
        emb_name = emb_model.split("/")[-1] if "/" in emb_model else emb_model
        device = self.current_device or "cpu"

        # 路径摘要（只显示 data/ 后的部分）
        if vector_path:
            tail = vector_path.replace("\\", "/").split("data/")[-1]
            path_summary = f"data/{tail}" if "data/" in vector_path else vector_path[-40:]
        else:
            path_summary = "(无)"

        # 主行
        status_main = {
            "ready": "✅",
            "keyword_only": "⚠",
            "created": "⏳",
            "cleaned": "📝",
            "not_ready": "⏳",
            "error": "❌",
            "unknown": "❓",
        }.get(status, "❓")
        main_line = f"{status_main} {project['name']}"

        # 详情
        if status == "ready":
            if is_external:
                detail = f"⚠ 外部路径\n{chunk_count} 段 · {emb_mb}MB\n{emb_name} ({device})\n{path_summary}"
            else:
                detail = f"{chunk_count} 段 · {emb_mb}MB\n{emb_name} ({device})\n{path_summary}"
        elif status == "keyword_only":
            detail = f"⚠ 仅关键词模式\nmetadata 存在但 npy 缺失\n{path_summary}"
        elif status in ("cleaned", "created", "not_ready"):
            detail = f"⚠ 尚未向量化\n{path_summary}"
        else:
            detail = f"{status}\n{path_summary}"

        # Fallback 提示（current_project_id 指向与实际使用不一致）
        fallback_hint = ""
        current_id = self.project_manager.config.get("current_project_id")
        if current_id and current_id != pid:
            fallback_hint = f"\n\n⚠ 已自动切换\n原选: [{current_id[:20]}…]\n现在用: {project['name']}"

        self.project_info_label.configure(text=main_line + "\n" + detail + fallback_hint)

    # ===================== 界面创建 =====================

    def _create_widgets(self):
        """创建所有界面组件"""
        # 顶部标题栏
        self._create_title_bar()

        # 主内容区
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=(5, 10))

        # 聊天区
        self.chat_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.chat_frame.pack(side="left", fill="both", expand=True)

        self._create_chat_area()
        self._create_input_area()

        # 侧边栏
        self._create_sidebar()

    def _create_title_bar(self):
        """创建顶部标题栏"""
        self.title_bar = ctk.CTkFrame(self, height=45, fg_color="#1a1a1a")
        self.title_bar.pack(fill="x", side="top")
        self.title_bar.pack_propagate(False)

        project_name = self.current_project["name"] if self.current_project else "未选择"
        self.title_label = ctk.CTkLabel(
            self.title_bar,
            text=f"📚 小说RAG系统 | {project_name}",
            font=("Microsoft YaHei UI", 14, "bold")
        )
        self.title_label.pack(side="left", padx=15)

        # 右侧轮数显示
        self.turn_label = ctk.CTkLabel(
            self.title_bar,
            text=f"对话轮数: 0/{MAX_TURNS}",
            font=("Microsoft YaHei UI", 11),
            text_color="gray"
        )
        self.turn_label.pack(side="right", padx=15)

    def _create_chat_area(self):
        """创建聊天显示区"""
        self.chat_container = ctk.CTkScrollableFrame(
            self.chat_frame,
            fg_color="#1a1a1a"
        )
        self.chat_container.pack(fill="both", expand=True, pady=(0, 10))

    def _create_input_area(self):
        """创建输入区"""
        input_frame = ctk.CTkFrame(self.chat_frame, fg_color="#2C2C2C", corner_radius=20)
        input_frame.pack(fill="x")

        # 文本输入框
        self.message_entry = ctk.CTkTextbox(
            input_frame,
            height=70,
            font=("Microsoft YaHei UI", 12),
            fg_color="#3C3C3C",
            border_width=0,
            wrap="word"
        )
        self.message_entry.pack(fill="x", expand=True, padx=10, pady=(10, 5))
        self.message_entry.bind("<Return>", self._on_enter_pressed)
        self.message_entry.bind("<KeyRelease>", self._adjust_textbox_height)

        # 按钮行
        button_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        button_frame.pack(fill="x", padx=10, pady=(0, 10))

        # 轮数提示
        ctk.CTkLabel(
            button_frame,
            text=f"剩余轮数: {MAX_TURNS}",
            font=("Microsoft YaHei UI", 10),
            text_color="gray"
        ).pack(side="left", padx=(0, 15))

        # 模型选择
        models = self._get_models()
        model_names = [m["name"] for m in models]

        ctk.CTkLabel(button_frame, text="模型:", font=("Microsoft YaHei UI", 11)).pack(side="left", padx=(0, 5))

        if model_names:
            self.model_combobox = ctk.CTkComboBox(
                button_frame,
                values=model_names,
                width=150,
                height=35,
                font=("Microsoft YaHei UI", 11)
            )
            self.model_combobox.pack(side="left", padx=(0, 10))
            self.model_combobox.set(model_names[0])
        else:
            self.model_combobox = ctk.CTkButton(
                button_frame,
                text="⚠ 请先配置模型",
                width=150,
                height=35,
                font=("Microsoft YaHei UI", 11),
                fg_color="#E81123",
                hover_color="#C41E3A",
                command=self._open_settings
            )
            self.model_combobox.pack(side="left", padx=(0, 10))

        # 清空按钮
        clear_btn = ctk.CTkButton(
            button_frame,
            text="清空",
            width=60,
            height=35,
            font=("Microsoft YaHei UI", 11),
            fg_color="#4A4A4A",
            hover_color="#3A3A3A",
            command=self._clear_input
        )
        clear_btn.pack(side="right", padx=(10, 0))

        # 发送按钮
        self.send_btn = ctk.CTkButton(
            button_frame,
            text="发送",
            width=80,
            height=35,
            font=("Microsoft YaHei UI", 12, "bold"),
            command=self._send_message
        )
        self.send_btn.pack(side="right")

    def _create_sidebar(self):
        """创建侧边栏"""
        self.sidebar = ctk.CTkFrame(self.main_frame, width=260, fg_color="#2C2C2C")
        self.sidebar.pack(side="right", fill="y", padx=(10, 0))
        self.sidebar.pack_propagate(False)

        # 标题
        ctk.CTkLabel(
            self.sidebar,
            text="📚 小说项目",
            font=("Microsoft YaHei UI", 16, "bold")
        ).pack(pady=(20, 10))

        # 当前项目信息
        self.project_info_label = ctk.CTkLabel(
            self.sidebar,
            text="📚 当前项目: 未选择",
            font=("Microsoft YaHei UI", 11),
            justify="left",
            wraplength=220
        )
        self.project_info_label.pack(padx=15, pady=(0, 20), anchor="w")

        # 按钮区
        btn_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        btn_frame.pack(fill="x", padx=15)

        # 选择已有小说
        select_btn = ctk.CTkButton(
            btn_frame,
            text="📖 选择已有小说",
            height=40,
            font=("Microsoft YaHei UI", 12),
            fg_color="#1E90FF",
            hover_color="#104E8B",
            command=self._show_project_list
        )
        select_btn.pack(fill="x", pady=(0, 10))

        # 导入已有向量化文件
        import_btn = ctk.CTkButton(
            btn_frame,
            text="📥 导入向量文件",
            height=40,
            font=("Microsoft YaHei UI", 12),
            fg_color="#FF8C00",
            hover_color="#CC7000",
            command=self._show_import_vector_dialog
        )
        import_btn.pack(fill="x", pady=(0, 10))

        # 新建项目
        create_btn = ctk.CTkButton(
            btn_frame,
            text="➕ 新建RAG项目",
            height=40,
            font=("Microsoft YaHei UI", 12),
            fg_color="#2E8B57",
            hover_color="#1E6B3E",
            command=self._show_project_wizard
        )
        create_btn.pack(fill="x", pady=(0, 10))

        # 查看所有项目
        view_btn = ctk.CTkButton(
            btn_frame,
            text="📋 查看所有项目",
            height=40,
            font=("Microsoft YaHei UI", 12),
            fg_color="#4A4A4A",
            hover_color="#3A3A3A",
            command=self._show_all_projects
        )
        view_btn.pack(fill="x", pady=(0, 10))

        # 分隔线
        ctk.CTkFrame(self.sidebar, height=2, fg_color="#3D3D3D").pack(fill="x", padx=15, pady=15)

        # 功能按钮
        func_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        func_frame.pack(fill="x", padx=15)

        settings_btn = ctk.CTkButton(
            func_frame,
            text="⚙ 系统设置",
            height=40,
            font=("Microsoft YaHei UI", 12),
            command=self._open_settings
        )
        settings_btn.pack(fill="x", pady=(0, 10))

        clear_btn = ctk.CTkButton(
            func_frame,
            text="🗑 清空对话",
            height=40,
            font=("Microsoft YaHei UI", 12),
            fg_color="#4A4A4A",
            hover_color="#3A3A3A",
            command=self._clear_chat
        )
        clear_btn.pack(fill="x", pady=(0, 10))

        # 状态栏
        self.status_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.status_frame.pack(side="bottom", fill="x", padx=15, pady=20)

        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="状态: 就绪",
            font=("Microsoft YaHei UI", 10)
        )
        self.status_label.pack(anchor="w")

        self._update_sidebar_project_info()

    # ===================== 导入向量化文件 =====================

    def _show_import_vector_dialog(self):
        """显示导入向量化文件对话框"""
        dialog = ctk.CTkToplevel(self)
        dialog.title("📥 导入已有向量化文件")
        dialog.geometry("550x500")
        dialog.transient(self)
        dialog.grab_set()

        # 小说名称输入
        name_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        name_frame.pack(fill="x", padx=20, pady=(20, 10))

        ctk.CTkLabel(
            name_frame,
            text="小说名称:",
            font=("Microsoft YaHei UI", 12)
        ).pack(side="left", padx=(0, 10))

        novel_name_entry = ctk.CTkEntry(
            name_frame,
            placeholder_text="例如: 遮天",
            font=("Microsoft YaHei UI", 12),
            width=200
        )
        novel_name_entry.pack(side="left")

        # 向量化目录选择
        vector_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        vector_frame.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(
            vector_frame,
            text="向量文件目录:",
            font=("Microsoft YaHei UI", 12)
        ).pack(anchor="w")

        vector_path_entry = ctk.CTkEntry(
            vector_frame,
            placeholder_text="选择包含 .npy 和 .json 文件的目录",
            font=("Microsoft YaHei UI", 12)
        )
        vector_path_entry.pack(fill="x", pady=(5, 5))

        def browse_vector_dir():
            dir_path = tkinter.filedialog.askdirectory(
                title="选择向量化文件目录",
                initialdir=get_root_dir()
            )
            if dir_path:
                vector_path_entry.delete(0, "end")
                vector_path_entry.insert(0, dir_path)
                # 自动检测小说名称
                dir_name = os.path.basename(dir_path)
                if not novel_name_entry.get():
                    novel_name_entry.delete(0, "end")
                    novel_name_entry.insert(0, dir_name)

        browse_btn = ctk.CTkButton(
            vector_frame,
            text="浏览...",
            width=100,
            command=browse_vector_dir
        )
        browse_btn.pack(anchor="e")

        # 源文件选择（可选）
        source_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        source_frame.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(
            source_frame,
            text="原始小说文件 (可选):",
            font=("Microsoft YaHei UI", 12)
        ).pack(anchor="w")

        source_path_entry = ctk.CTkEntry(
            source_frame,
            placeholder_text="选择原始小说 .txt 文件",
            font=("Microsoft YaHei UI", 12)
        )
        source_path_entry.pack(fill="x", pady=(5, 5))

        def browse_source():
            file_path = tkinter.filedialog.askopenfilename(
                title="选择原始小说文件",
                filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
            )
            if file_path:
                source_path_entry.delete(0, "end")
                source_path_entry.insert(0, file_path)

        browse_source_btn = ctk.CTkButton(
            source_frame,
            text="浏览...",
            width=100,
            command=browse_source
        )
        browse_source_btn.pack(anchor="e")

        # 提示信息
        info_label = ctk.CTkLabel(
            dialog,
            text="💡 提示: 目录中需要包含配对的 .npy 和 .json 文件\n(例如: 遮天_20260610.npy 和 遮天_20260610.json)",
            font=("Microsoft YaHei UI", 10),
            text_color="gray",
            justify="left"
        )
        info_label.pack(padx=20, pady=10)

        # 导入按钮
        def do_import():
            name = novel_name_entry.get().strip()
            vector_dir = vector_path_entry.get().strip()
            source_file = source_path_entry.get().strip()

            if not name:
                tkinter.messagebox.showwarning("提示", "请输入小说名称！")
                return
            if not vector_dir:
                tkinter.messagebox.showwarning("提示", "请选择向量化文件目录！")
                return

            # 检查小说名是否已存在
            if self.project_manager.project_name_exists(name):
                if not tkinter.messagebox.askyesno(
                    "确认",
                    f"小说「{name}」已存在，是否覆盖？"
                ):
                    return

            # 执行导入
            project = self.project_manager.import_existing_vector(
                name=name,
                source_path=source_file,
                vector_db_path=vector_dir
            )

            if project:
                tkinter.messagebox.showinfo(
                    "成功",
                    f"✓ 小说「{name}」导入成功！\n现在可以选择该项目进行问答。"
                )
                dialog.destroy()
                self._show_project_list()
            else:
                tkinter.messagebox.showerror(
                    "错误",
                    "导入失败，请检查目录中是否包含完整的 .npy 和 .json 文件。"
                )

        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=20)

        ctk.CTkButton(
            btn_frame,
            text="取消",
            width=100,
            fg_color="#4A4A4A",
            hover_color="#3A3A3A",
            command=dialog.destroy
        ).pack(side="right", padx=(10, 0))

        ctk.CTkButton(
            btn_frame,
            text="导入",
            width=100,
            fg_color="#2E8B57",
            hover_color="#1E6B3E",
            command=do_import
        ).pack(side="right")

    # ===================== 项目列表对话框 =====================

    def _show_project_list(self):
        """显示项目选择对话框"""
        projects = self.project_manager.get_all_projects()

        if not projects:
            tkinter.messagebox.showinfo("提示", "暂无小说项目，请先创建一个新项目！")
            return

        # 创建选择对话框
        dialog = ctk.CTkToplevel(self)
        dialog.title("📖 选择小说项目")
        dialog.geometry("500x400")
        dialog.transient(self)
        dialog.grab_set()

        # 列表框架
        list_frame = ctk.CTkScrollableFrame(dialog, fg_color="#1a1a1a")
        list_frame.pack(fill="both", expand=True, padx=15, pady=15)

        # 项目列表
        current_pid = self.project_manager.config.get("current_project_id")
        active_pid = self.current_project.get("id") if self.current_project else None

        for project in projects:
            rt = self.project_manager.get_project_runtime(project["id"])
            status = rt["status"]
            chunk_count = rt["chunk_count"]
            has_emb = rt["has_embeddings"]
            is_external = rt["is_external"]
            emb_mb = rt["embeddings_size_mb"]

            # 路径摘要
            vpath = rt["vector_db_path"]
            path_tail = vpath.replace("\\", "/").split("data/")[-1] if "data/" in vpath else vpath[-35:]
            path_summary = f"data/{path_tail}" if "data/" in vpath else vpath

            # 状态图标
            status_icons = {
                "ready": "✅",
                "keyword_only": "⚠",
                "created": "⏳",
                "cleaned": "📝",
                "not_ready": "⏳",
                "error": "❌"
            }
            icon = status_icons.get(status, "❓")

            # 主行
            main_line = f"{icon} {project['name']}  ({status})"
            if status == "ready":
                if is_external:
                    main_line += f"\n   {chunk_count} 段 · {emb_mb}MB · ⚠外部路径"
                else:
                    main_line += f"\n   {chunk_count} 段 · {emb_mb}MB"
            elif status == "keyword_only":
                main_line += f"\n   仅关键词检索 · npy 缺失"
            elif status in ("cleaned", "created"):
                main_line += f"\n   尚未向量化"
            main_line += f"\n   路径: {path_summary}"

            # 当前在用 / 自动选中标记
            badges = []
            if project["id"] == active_pid:
                badges.append("🔵正在用")
            if current_pid and project["id"] != active_pid and status == "ready":
                badges.append("⚠自动选")
            if badges:
                main_line += f"\n   {' · '.join(badges)}"

            # 项目卡片
            card = ctk.CTkFrame(list_frame, fg_color="#2C2C2C", corner_radius=10)
            card.pack(fill="x", pady=5, padx=5)

            info_label = ctk.CTkLabel(
                card,
                text=main_line,
                font=("Microsoft YaHei UI", 11),
                anchor="w",
                justify="left"
            )
            info_label.pack(side="left", padx=15, pady=10)

            # 选择按钮
            select_btn = ctk.CTkButton(
                card,
                text="选择",
                width=60,
                height=30,
                fg_color="#1E90FF",
                hover_color="#104E8B",
                command=lambda pid=project["id"]: self._select_project(pid, dialog)
            )
            select_btn.pack(side="right", padx=10)

        # 关闭按钮
        close_btn = ctk.CTkButton(
            dialog,
            text="关闭",
            width=100,
            command=dialog.destroy
        )
        close_btn.pack(pady=10)

    def _select_project(self, project_id: str, dialog=None):
        """选择项目"""
        self.set_project(project_id)
        if dialog:
            dialog.destroy()

    def _show_all_projects(self):
        """显示所有项目的详细信息"""
        projects = self.project_manager.get_all_projects()

        dialog = ctk.CTkToplevel(self)
        dialog.title("📋 所有项目")
        dialog.geometry("600x500")
        dialog.transient(self)
        dialog.grab_set()

        # 头部统计
        header = ctk.CTkFrame(dialog, fg_color="transparent")
        header.pack(fill="x", padx=15, pady=15)

        ctk.CTkLabel(
            header,
            text=f"共 {len(projects)} 个项目",
            font=("Microsoft YaHei UI", 14, "bold")
        ).pack(side="left")

        # 项目列表
        list_frame = ctk.CTkScrollableFrame(dialog, fg_color="#1a1a1a")
        list_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        if not projects:
            ctk.CTkLabel(
                list_frame,
                text="暂无项目，请创建新项目",
                font=("Microsoft YaHei UI", 12)
            ).pack(pady=50)
        else:
            for project in projects:
                status = self.project_manager.get_project_status(project["id"])

                card = ctk.CTkFrame(list_frame, fg_color="#2C2C2C", corner_radius=10)
                card.pack(fill="x", pady=5, padx=5)

                # 项目信息
                info_frame = ctk.CTkFrame(card, fg_color="transparent")
                info_frame.pack(side="left", fill="both", expand=True, padx=15, pady=10)

                ctk.CTkLabel(
                    info_frame,
                    text=f"📚 {project['name']}",
                    font=("Microsoft YaHei UI", 12, "bold"),
                    anchor="w"
                ).pack(anchor="w")

                created = project.get("created_at", "未知")
                ctk.CTkLabel(
                    info_frame,
                    text=f"创建: {created} | 状态: {status}",
                    font=("Microsoft YaHei UI", 10),
                    text_color="gray",
                    anchor="w"
                ).pack(anchor="w")

                # 操作按钮
                btn_frame = ctk.CTkFrame(card, fg_color="transparent")
                btn_frame.pack(side="right", padx=10)

                ctk.CTkButton(
                    btn_frame,
                    text="选择",
                    width=60,
                    height=28,
                    fg_color="#1E90FF",
                    hover_color="#104E8B",
                    command=lambda pid=project["id"]: self._select_project(pid, dialog)
                ).pack(side="left", padx=(0, 5))

                ctk.CTkButton(
                    btn_frame,
                    text="删除",
                    width=60,
                    height=28,
                    fg_color="#E81123",
                    hover_color="#C41E3A",
                    command=lambda pid=project["id"], pname=project["name"]: self._delete_project(pid, pname, dialog)
                ).pack(side="left")

        # 关闭按钮
        close_btn = ctk.CTkButton(
            dialog,
            text="关闭",
            width=100,
            command=dialog.destroy
        )
        close_btn.pack(pady=10)

    def _delete_project(self, project_id: str, project_name: str, parent_dialog=None):
        """删除项目"""
        if not tkinter.messagebox.askyesno(
            "确认删除",
            f"确定要删除项目「{project_name}」吗？\n此操作不可恢复！"
        ):
            return

        success = self.project_manager.delete_project(project_id)
        if success:
            # 如果删除的是当前项目
            if self.current_project and self.current_project["id"] == project_id:
                self.current_project = None
                self.turn_count = 0
                self.title("📚 小说RAG系统 - 当前项目: 未选择")
                self._update_sidebar_project_info()
                self._clear_chat_area(show_welcome=True)

            # 刷新项目列表
            if parent_dialog:
                parent_dialog.destroy()
                self._show_all_projects()
        else:
            tkinter.messagebox.showerror("错误", "删除项目失败！")

    # ===================== 新建项目向导 =====================

    def _show_project_wizard(self):
        """显示新建项目向导"""
        if not self.config_manager.get_models():
            if not tkinter.messagebox.askyesno(
                "提示",
                "请先配置至少一个AI模型，是否现在去配置？"
            ):
                return
            self._open_settings()
            return

        wizard = ProjectWizard(self, self.project_manager, self.config_manager, on_complete=self._on_project_created)

    def _on_project_created(self, project):
        """项目创建完成回调"""
        if project:
            self.set_project(project["id"])
            self.status_label.configure(text=f"✓ 项目「{project['name']}」创建完成！")

    # ===================== 对话功能 =====================

    def _get_models(self):
        """获取模型列表"""
        return self.config_manager.get_models()

    def update_model_list(self):
        """更新模型列表"""
        models = self._get_models()
        model_names = [m["name"] for m in models]

        if hasattr(self, 'model_combobox'):
            try:
                self.model_combobox.configure(values=model_names)
                if model_names:
                    self.model_combobox.set(model_names[0])
            except Exception:
                self._recreate_model_combobox()

    def _recreate_model_combobox(self):
        """重建模型选择框"""
        if hasattr(self, 'model_combobox'):
            parent = self.model_combobox.master
            self.model_combobox.destroy()

        models = self._get_models()
        model_names = [m["name"] for m in models]

        self.model_combobox = ctk.CTkComboBox(
            parent,
            values=model_names,
            width=150,
            height=35,
            font=("Microsoft YaHei UI", 11)
        )
        self.model_combobox.pack(side="left", padx=(0, 10))
        if model_names:
            self.model_combobox.set(model_names[0])

    def _send_message(self):
        """发送消息"""
        # 检查项目
        if not self.current_project:
            tkinter.messagebox.showwarning("提示", "请先选择或创建一个小说项目！")
            self._show_project_list()
            return

        # 检查模型配置（避免进入后台线程后调用不存在的 .get()）
        if not self._get_models():
            tkinter.messagebox.showwarning(
                "提示",
                "尚未配置 AI 模型，请先在「系统设置」中填入 API Key。",
            )
            self._open_settings()
            return

        # 检查项目状态
        status = self.project_manager.get_project_status(self.current_project["id"])
        if status in ("created", "cleaned", "not_ready", "error"):
            tkinter.messagebox.showwarning(
                "提示",
                f"当前项目尚未就绪（状态: {status}），请先完成项目创建流程！"
            )
            return
        if status == "keyword_only":
            # 纯关键词模式可问答，但召回质量差；一次性提示后放行
            self.status_label.configure(
                text="⚠ 关键词模式(缺向量库)，召回质量较差，建议重建向量"
            )

        # 检查轮数
        if self.turn_count >= MAX_TURNS:
            self._show_turn_limit_warning()
            return

        # 获取消息
        message = self.message_entry.get("1.0", "end-1c").strip()
        if not message:
            return

        # 添加用户消息到界面
        user_bubble = MessageBubble(self.chat_container, message, is_user=True)
        user_bubble.pack(fill="x")

        self.messages.append({
            "role": "user",
            "content": message,
            "is_user": True
        })

        # 保存到日志
        self.chat_logger.save_message("user", message)

        # 清空输入
        self.message_entry.delete("1.0", "end")
        self.message_entry.configure(height=70)

        # 滚动到底部
        self._scroll_to_bottom()

        # 更新状态
        self.status_label.configure(text="状态: 准备中...")

        # 在后台线程中调用API
        thread = threading.Thread(target=self._call_api, args=(message,))
        thread.daemon = True
        thread.start()

    def _call_api(self, message: str):
        """调用AI API（后台线程）"""
        self.api_start_time = time.time()
        self.api_timeout = False
        self.api_completed = False  # API 调用完成（成功或失败）后置 True
        self.api_error = False       # 用于区分 "超时" 和 "错误"

        # 超时检测线程
        timeout_thread = threading.Thread(target=self._api_timeout_check)
        timeout_thread.daemon = True
        timeout_thread.start()

        def _mark_completed(success: bool = True):
            """标记 API 调用已结束（让超时线程能及时退出）"""
            self.api_completed = True
            self.api_timeout = not success  # 成功 → False（不超时）；失败 → 由调用方决定
            self.api_error = not success

        try:
            # 获取选中的模型（防御：未配置时 model_combobox 是占位按钮，无 .get()）
            if not hasattr(self.model_combobox, "get"):
                _mark_completed(success=True)
                self.after(0, lambda: self.status_label.configure(text="状态: 请先配置AI模型"))
                self.after(0, lambda: self._add_ai_message("尚未配置AI模型，请点击右下角「系统设置」填入API Key后重试。"))
                return
            selected_model_name = self.model_combobox.get()
            if not selected_model_name:
                _mark_completed(success=True)
                self.after(0, lambda: self.status_label.configure(text="状态: 请先配置AI模型"))
                return

            models = self._get_models()
            selected_model = None
            for model in models:
                if model["name"] == selected_model_name:
                    selected_model = model
                    break

            if not selected_model:
                _mark_completed(success=True)
                self.after(0, lambda: self.status_label.configure(text="状态: 模型配置不存在"))
                return

            if selected_model.get("is_local", False):
                _mark_completed(success=True)
                self.after(0, lambda: self.status_label.configure(text="状态: 本地模型暂未实现"))
                self.after(0, lambda: self._add_ai_message("本地模型功能正在开发中..."))
                return

            api_url = selected_model.get("api_url", "")
            api_key = selected_model.get("api_key", "")
            model_name = selected_model.get("model_id", selected_model.get("name", ""))

            if not api_url or not api_key:
                _mark_completed(success=True)
                self.after(0, lambda: self.status_label.configure(text="状态: 请先配置API设置"))
                return

            # 构建Prompt
            prompt_template = self.config_manager.get("prompt_template", "")
            embedding_path = self.config_manager.get("embedding_model_path", "")
            top_k = int(self.config_manager.get("top_k", "3"))
            enable_thinking = self.config_manager.get("enable_thinking", False)

            # RAG检索
            vector_path = self.current_project.get("vector_db_path", "")
            vector_file = self.current_project.get("vector_file", "")
            metadata_file = self.current_project.get("metadata_file", "")
            # 精排策略：GUI 没有命令行开关，与 CLI 共用同一套优先级
            # （命令行/GUI 显式 > 配置 enable_rerank > 探测到本地重排模型自动启用 > 降级纯向量检索）
            rerank_policy = resolve_rerank_policy(None, self.config_manager)
            reranker_path = rerank_policy.get("reranker_path", "")
            enable_rerank = rerank_policy["enable"]
            if rerank_policy.get("note"):
                print(rerank_policy["note"])

            if vector_path and os.path.exists(vector_path) and prompt_template:
                vector_status = ""  # 向量检索状态（空=向量正常参与）
                query_rewrite_note = ""  # 查询改写提示（空=未触发或失败降级）
                self.after(0, lambda: self.status_label.configure(text="状态: 正在加载检索引擎（首次需 5-15s）..."))

                # 启动进度心跳线程，让用户看到系统在干活（每秒更新状态栏）
                progress_stop = [False]
                def _progress_heartbeat():
                    t0 = time.time()
                    while not progress_stop[0]:
                        time.sleep(1)
                        if progress_stop[0]:
                            break
                        elapsed = int(time.time() - t0)
                        try:
                            self.after(0, lambda e=elapsed: self.status_label.configure(
                                text=f"状态: 正在加载检索引擎... {e}s"
                            ))
                        except Exception:
                            break

                heartbeat_thread = threading.Thread(target=_progress_heartbeat, daemon=True)
                heartbeat_thread.start()

                try:
                    retriever = RAGRetriever.get_or_create(
                        vector_path, reranker_path, embedding_path,
                        vector_file=vector_file,
                        metadata_file=metadata_file
                    )
                    # 检索前查询改写：仅当问题命中元词（主角/男主/女主…）时才多调一次
                    # LLM；失败自动回退原查询（extra_queries 为空 → 等同不做改写）
                    extra_queries, query_rewrite_note = self._maybe_rewrite_query(
                        message, api_url, api_key, model_name
                    )
                    hits = retriever.retrieve_multi(
                        message,
                        extra_queries=extra_queries,
                        top_k=top_k,
                        enable_rerank=enable_rerank
                    )
                    # 读取向量检索状态：若向量检索未生效，记录用于向用户提示
                    vector_status = getattr(retriever, "last_vector_status", "") or ""
                finally:
                    progress_stop[0] = True

                # 分层上下文组装：命中子块 > 父块 > 邻居，按章节分组
                context = assemble_context(hits)

                # 查询改写提示：状态栏 + 提示气泡（失败时 note 为空 → 静默不打扰）
                if query_rewrite_note:
                    def _notify_rewrite_note(n=query_rewrite_note):
                        self.status_label.configure(text=n)
                        try:
                            note_bubble = MessageBubble(self.chat_container, n, is_user=False)
                            note_bubble.pack(fill="x")
                            self._scroll_to_bottom()
                        except Exception:
                            pass
                    self.after(0, _notify_rewrite_note)

                # 拼接多轮对话历史
                history = self._get_conversation_history_for_prompt()
                final_prompt = self._build_prompt_with_history(
                    prompt_template, context, message, history
                )
            else:
                final_prompt = message
                vector_status = ""  # 未走检索路径时无向量提示

            # 调用API
            self.after(0, lambda: self.status_label.configure(text="状态: 正在调用AI..."))

            api_client = APIClient(api_url, api_key, model_name)
            reply = api_client.call_api(final_prompt, enable_thinking)
            _mark_completed(success=True)  # API 已成功返回 → 告诉 timeout 线程别报警

            if not self.api_timeout:
                self.after(0, lambda r=reply: self._add_ai_message(r))

            # 向量检索未生效时，提示用户（气泡仅提示一次，状态栏每次都会显示）
            if vector_status:
                note = f"⚠ {vector_status}。若需启用语义召回，请检查嵌入模型配置与依赖。"
                def _notify_vector_note(n=note):
                    self.status_label.configure(text=n)
                    if not self._vector_mode_warned:
                        self._vector_mode_warned = True
                        try:
                            note_bubble = MessageBubble(self.chat_container, n, is_user=False)
                            note_bubble.pack(fill="x")
                            self._scroll_to_bottom()
                        except Exception:
                            pass
                self.after(0, _notify_vector_note)

        except Exception as e:
            _mark_completed(success=False)  # API 失败 → 防止 timeout 线程重复报警
            if not self.api_timeout:
                error_msg = f"API调用异常: {str(e)}"
                print(f"API Error: {traceback.format_exc()}")
                self.after(0, lambda: self._show_error(error_msg))
                self.after(0, lambda: self.status_label.configure(text="状态: 错误"))

    def _maybe_rewrite_query(self, question, api_url, api_key, model_name):
        """检索前查询改写（元词路由触发 + 静默降级）。

        只在问题命中元词（主角/男主/女主/主人公/男一号/女一号/主角团…）时才
        发起一次短 LLM 调用，把元词改写为具体人物名；非元词问题直接返回，
        零额外开销。任何异常（网络/解析/配置）都回退原查询，不影响原链路。

        Args:
            question: 用户原始问题
            api_url / api_key / model_name: 当前选中模型的连接信息（复用现有配置）

        Returns:
            (extra_queries, note)
            extra_queries: 改写查询列表，供 retrieve_multi 做多路召回
            note: 给用户看的提示文案（如「查询改写：主角 → 陈曦」）；
                  未触发或失败时为空串 → 调用方不展示、不提示
        """
        try:
            if not self.config_manager.get_enable_query_rewrite():
                return [], ""  # 开关关闭 → 完全走原有路径
            if not has_meta_terms(question):
                return [], ""  # 路由触发：非元词问题零额外 LLM 调用
        except Exception:
            return [], ""

        self.after(0, lambda: self.status_label.configure(
            text="状态: 正在改写查询（元词 → 具体人物）..."
        ))
        novel_name = ""
        try:
            if self.current_project:
                novel_name = self.current_project.get("name", "") or ""
        except Exception:
            novel_name = ""

        try:
            rewrite_client = APIClient(api_url, api_key, model_name)
            result = rewrite_query(question, rewrite_client, novel_name=novel_name)
        except Exception as e:
            # rewrite_query 内部已兜底，这里再兜一层，确保永不打断问答链路
            print(f"⚠ 查询改写异常（{type(e).__name__}: {e}），本次回退原查询")
            return [], ""

        if not result.succeeded:
            return [], ""  # 静默降级：失败不提示，用户无感知
        return result.extra_queries, result.summary

    def _api_timeout_check(self):
        """API超时检测（根据 api_completed 标志提前退出）"""
        timeout_seconds = 90
        check_interval = 1

        while time.time() - self.api_start_time < timeout_seconds:
            time.sleep(check_interval)
            # 关键：API 成功后 api_completed=True，立即退出循环
            if self.api_completed:
                return
            elapsed = int(time.time() - self.api_start_time)
            status_text = f"状态: AI生成中... {elapsed}s"
            self.after(0, lambda t=status_text: self.status_label.configure(text=t))

        # 超时：只有在 API 仍未完成时才报警
        if not self.api_completed:
            self.api_timeout = True
            self.api_error = True
            self.after(0, lambda: self.status_label.configure(text="状态: 超时"))
            self.after(0, lambda: self._show_error("请求超时，服务器响应时间过长。请检查网络连接或稍后重试。"))

    def _get_conversation_history_for_prompt(self) -> str:
        """获取用于Prompt的对话历史"""
        if len(self.messages) <= 1:
            return ""

        # 获取最近的对话历史（限制在10轮以内）
        history_messages = self.messages[-20:]  # 最多20条消息

        history_text = "以下是与用户之前的对话历史（供参考）：\n"
        for msg in history_messages:
            role = "用户" if msg["role"] == "user" else "助手"
            history_text += f"{role}: {msg['content'][:200]}\n"
        history_text += "---以上是历史记录---\n\n"

        return history_text

    def _build_prompt_with_history(
        self,
        template: str,
        context: str,
        question: str,
        history: str
    ) -> str:
        """构建包含历史的Prompt"""
        # 获取当前小说名称
        novel_name = self.current_project["name"] if self.current_project else ""
        
        # 先填充模板
        if novel_name:
            base_prompt = template.format(
                novel_name=novel_name,
                context=context,
                question=question
            )
        else:
            base_prompt = template.format(context=context, question=question)

        # 如果有历史，插入到Prompt前面
        if history:
            return history + base_prompt
        return base_prompt

    def _add_ai_message(self, reply: str):
        """添加AI回复"""
        cleaned_reply = self._clean_markdown(reply)

        ai_bubble = MessageBubble(self.chat_container, cleaned_reply, is_user=False)
        ai_bubble.pack(fill="x")

        self.messages.append({
            "role": "assistant",
            "content": cleaned_reply,
            "is_user": False
        })

        # 保存到日志
        self.chat_logger.save_message("assistant", reply)

        # 增加轮数
        self.turn_count += 1
        self._update_turn_count_display()

        self.status_label.configure(text="状态: 就绪")

        # 检查是否达到轮数上限
        if self.turn_count >= MAX_TURNS:
            self.after(500, self._show_turn_limit_warning)

    def _update_turn_count_display(self):
        """更新轮数显示"""
        remaining = MAX_TURNS - self.turn_count
        self.turn_label.configure(text=f"对话轮数: {self.turn_count}/{MAX_TURNS}")

        # 更新输入区的提示
        for widget in self.message_entry.master.winfo_children():
            if isinstance(widget, ctk.CTkLabel) and "剩余轮数" in widget.cget("text"):
                widget.configure(text=f"剩余轮数: {max(0, remaining)}")

    def _show_turn_limit_warning(self):
        """显示轮数限制警告"""
        tkinter.messagebox.showwarning(
            "对话轮数已达上限",
            f"您已经连续对话了 {MAX_TURNS} 轮，建议选择其他小说项目继续对话，或清空当前对话重新开始。"
        )
        # 清空聊天区域
        self._clear_chat_area(show_welcome=True)

    def _show_welcome_message(self):
        """显示欢迎消息"""
        welcome_text = (
            "👋 欢迎使用小说RAG系统！\n\n"
            "请从左侧选择已有项目，或创建新的RAG项目开始对话。\n\n"
            "💡 使用提示：\n"
            "1. 点击「➕ 新建RAG项目」创建新项目\n"
            "2. 选择小说源文件 (.txt)\n"
            "3. 配置清洗规则和切片参数\n"
            "4. 等待处理完成后即可开始问答\n\n"
            "⚠ 注意：每个项目最多支持连续对话 15 轮"
        )

        welcome_msg = MessageBubble(self.chat_container, welcome_text, is_user=False)
        welcome_msg.pack(fill="x", padx=10, pady=20)

    def _clear_chat_area(self, show_welcome: bool = False):
        """清空聊天区域"""
        for widget in self.chat_container.winfo_children():
            widget.destroy()

        self.messages = []
        self.turn_count = 0
        self._update_turn_count_display()

        if show_welcome and self.current_project:
            welcome_msg = MessageBubble(
                self.chat_container,
                f"✅ 当前项目: {self.current_project['name']}\n\n请输入您的问题，我将基于小说内容为您解答。",
                is_user=False
            )
            welcome_msg.pack(fill="x", padx=10, pady=20)
        elif show_welcome:
            self._show_welcome_message()

        self.status_label.configure(text="状态: 就绪")

    def _clear_chat(self):
        """清空对话"""
        if self.current_project:
            self._clear_chat_area(show_welcome=True)
        else:
            self._clear_chat_area(show_welcome=True)

    def _clear_input(self):
        """清空输入框"""
        self.message_entry.delete("1.0", "end")
        self.message_entry.configure(height=70)

    def _on_enter_pressed(self, event):
        """回车键处理"""
        if event.state == 4:  # Shift+Enter
            self.message_entry.insert("insert", "\n")
            return "break"
        else:
            self._send_message()
            return "break"

    def _adjust_textbox_height(self, event):
        """调整输入框高度"""
        content = self.message_entry.get("1.0", "end-1c")
        lines = content.count("\n") + 1
        max_lines = 6
        min_height = 70
        line_height = 24

        if lines <= 1:
            new_height = min_height
        elif lines >= max_lines:
            new_height = min_height + (max_lines - 1) * line_height
        else:
            new_height = min_height + (lines - 1) * line_height

        new_height = min(new_height, 200)
        self.message_entry.configure(height=new_height)

    def _clean_markdown(self, content: str) -> str:
        """清理AI回复中的Markdown格式"""
        # 移除标题
        content = re.sub(r'^#{1,6}\s+', '', content, flags=re.MULTILINE)
        # 移除粗体/斜体
        content = content.replace('**', '')
        content = content.replace('*', '')
        content = content.replace('__', '')
        content = content.replace('_', '')
        # 移除行内代码
        content = content.replace('`', '')
        content = re.sub(r'```[\s\S]*?```', '', content)
        # 移除链接
        content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)
        # 移除图片
        content = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', content)
        # 移除水平线
        content = re.sub(r'^---\s*$', '', content, flags=re.MULTILINE)
        # 移除引用
        content = re.sub(r'^>\s*', '', content, flags=re.MULTILINE)

        # 处理列表
        lines = content.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('- ') or stripped.startswith('* ') or stripped.startswith('+ '):
                cleaned_lines.append(stripped[2:])
            elif re.match(r'^\d+\.\s+', stripped):
                cleaned_lines.append(re.sub(r'^\d+\.\s+', '', stripped))
            else:
                cleaned_lines.append(line)

        cleaned_content = '\n'.join(cleaned_lines)
        cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)

        return cleaned_content.strip()

    def _scroll_to_bottom(self):
        """滚动到底部"""
        try:
            self.chat_container._parent_canvas.yview_moveto(1.0)
        except Exception:
            pass

    def _show_error(self, error_msg: str):
        """显示错误消息"""
        error_bubble = MessageBubble(self.chat_container, error_msg, is_user=False)
        error_bubble.pack(fill="x")
        self.status_label.configure(text="状态: 错误")

    def _open_settings(self):
        """打开设置窗口"""
        SettingsWindow(self, self.config_manager)

    def _on_closing(self):
        """关闭窗口时保存"""
        self.config_manager.save()
        self.project_manager.save_config()
        self.destroy()


def main():
    """主函数"""
    try:
        # 确保必要目录存在
        from utils import (
            get_data_dir,
            get_chat_logs_dir,
            check_vector_dependencies,
            format_vector_dependency_hint,
        )
        get_data_dir()
        get_chat_logs_dir()

        # 运行环境自检：向量检索依赖 torch / sentence-transformers。
        # 缺失时不崩溃（仍可纯关键词模式运行），但必须显式告知用户，
        # 否则会静默降级：建库只落 metadata.json、检索拿不到 embeddings.npy。
        dep = check_vector_dependencies()
        if not dep["ok"]:
            hint = format_vector_dependency_hint(dep)
            print(hint)
            try:
                root = ctk.CTk()
                root.withdraw()
                tkinter.messagebox.showwarning(
                    "环境警告：向量检索不可用",
                    hint + "\n\n请改用项目虚拟环境启动后再建库/提问。",
                )
                root.destroy()
            except Exception:
                pass

        app = NovelRAGApp()
        app.mainloop()
    except Exception as e:
        error_msg = f"程序异常退出: {str(e)}\n\n详细信息:\n{traceback.format_exc()}"
        print(error_msg)

        # 写入崩溃日志
        crash_log_path = str(get_root_dir() / "crash_log.txt")
        with open(crash_log_path, "w", encoding="utf-8") as f:
            f.write(error_msg)

        # 显示错误对话框
        try:
            root = ctk.CTk()
            root.withdraw()
            tkinter.messagebox.showerror(
                "程序崩溃",
                f"程序异常退出，请查看 crash_log.txt 获取详细信息\n\n{str(e)}"
            )
            root.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    main()
