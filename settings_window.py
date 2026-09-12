"""
settings_window.py — 设置窗口
管理模型配置、向量化设置、Prompt模板等
"""
import json
import customtkinter as ctk
from tkinter import filedialog
from typing import Optional

from config_manager import ConfigManager
from rag_retriever import RAGRetriever


class SettingsWindow(ctk.CTkToplevel):
    """设置窗口"""

    def __init__(self, master, config_manager: ConfigManager, **kwargs):
        """
        初始化设置窗口

        Args:
            master: 主窗口
            config_manager: 配置管理器
        """
        super().__init__(master, **kwargs)
        self.config_manager = config_manager
        self.master = master

        self.title("⚙ 系统设置")
        master_width = self.master.winfo_width()
        master_height = self.master.winfo_height()

        default_width = min(900, max(750, master_width))
        default_height = min(700, max(600, master_height))

        self.geometry(f"{default_width}x{default_height}")
        self.minsize(750, 600)
        self.maxsize(master_width + 50, master_height)
        self.resizable(True, True)

        self.lift()
        self.attributes("-topmost", True)
        self.focus_force()

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self._create_widgets()

    def _create_widgets(self):
        """创建界面组件"""
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=(10, 5))

        self.tab_models = self.tabview.add("🤖 模型管理")
        self.tab_vector = self.tabview.add("📊 向量化设置")

        self._create_models_tab()
        self._create_vector_tab()

    # ===================== 模型管理 =====================

    def _create_models_tab(self):
        """创建模型管理标签页"""
        models_frame = ctk.CTkFrame(self.tab_models, fg_color="transparent")
        models_frame.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(
            models_frame,
            text="模型管理",
            font=("Microsoft YaHei UI", 16, "bold")
        ).pack(anchor="w", pady=(0, 20))

        # 模型选择器
        model_selector_frame = ctk.CTkFrame(models_frame, fg_color="transparent")
        model_selector_frame.pack(fill="x", pady=(0, 15))

        ctk.CTkLabel(model_selector_frame, text="选择模型:", font=("Microsoft YaHei UI", 12)).pack(side="left", padx=(0, 10))

        models = self._get_models()
        self.model_names = [m["name"] for m in models]
        if not self.model_names:
            self.model_names = ["暂无模型"]

        self.model_combobox = ctk.CTkComboBox(
            model_selector_frame,
            values=self.model_names,
            width=200,
            command=self._on_model_selected
        )
        self.model_combobox.pack(side="left")

        add_save_btn = ctk.CTkButton(
            model_selector_frame,
            text="➕ 添加/保存",
            width=120,
            height=28,
            fg_color="#1E90FF",
            hover_color="#104E8B",
            command=self._add_or_save_model
        )
        add_save_btn.pack(side="right")

        # 模型配置表单
        self.model_config_frame = ctk.CTkFrame(models_frame, fg_color="#2C2C2C", corner_radius=10)
        self.model_config_frame.pack(fill="both", expand=True, pady=(0, 15))

        config_inner_frame = ctk.CTkFrame(self.model_config_frame, fg_color="transparent")
        config_inner_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # 模型名称
        ctk.CTkLabel(config_inner_frame, text="模型显示名称:", font=("Microsoft YaHei UI", 12)).pack(anchor="w")
        self.model_name_entry = ctk.CTkEntry(config_inner_frame, width=480, height=35)
        self.model_name_entry.pack(anchor="w", pady=(5, 15))

        # 模型ID
        ctk.CTkLabel(config_inner_frame, text="模型ID (API使用):", font=("Microsoft YaHei UI", 12)).pack(anchor="w")
        self.model_id_entry = ctk.CTkEntry(config_inner_frame, width=480, height=35)
        self.model_id_entry.pack(anchor="w", pady=(5, 15))

        # API URL
        ctk.CTkLabel(config_inner_frame, text="API URL:", font=("Microsoft YaHei UI", 12)).pack(anchor="w")
        self.url_entry = ctk.CTkEntry(config_inner_frame, width=480, height=35)
        self.url_entry.pack(anchor="w", pady=(5, 15))

        # API Key
        ctk.CTkLabel(config_inner_frame, text="API Key:", font=("Microsoft YaHei UI", 12)).pack(anchor="w")
        self.api_key_entry = ctk.CTkEntry(config_inner_frame, width=480, height=35, show="*")
        self.api_key_entry.pack(anchor="w", pady=(5, 15))

        # 本地模型选项
        local_model_frame = ctk.CTkFrame(config_inner_frame, fg_color="transparent")
        local_model_frame.pack(fill="x", pady=(0, 15))

        self.is_local_var = ctk.BooleanVar()
        self.local_checkbox = ctk.CTkCheckBox(
            local_model_frame,
            text="本地大模型",
            variable=self.is_local_var,
            command=self._toggle_local_model
        )
        self.local_checkbox.pack(side="left")

        ctk.CTkLabel(local_model_frame, text="模型路径:", font=("Microsoft YaHei UI", 12)).pack(side="left", padx=(20, 5))
        self.local_path_entry = ctk.CTkEntry(local_model_frame, width=250, height=30)
        self.local_path_entry.pack(side="left", padx=(0, 10))
        self.local_path_entry.insert(0, self.config_manager.get("local_model_path", ""))

        browse_local_btn = ctk.CTkButton(
            local_model_frame,
            text="浏览",
            width=60,
            height=30,
            command=self._browse_local_model
        )
        browse_local_btn.pack(side="left")

        self.local_path_entry.configure(state="disabled")

        # 删除按钮
        delete_frame = ctk.CTkFrame(models_frame, fg_color="transparent")
        delete_frame.pack(fill="x")

        delete_model_btn = ctk.CTkButton(
            delete_frame,
            text="🗑 删除选中模型",
            width=150,
            height=35,
            fg_color="#E81123",
            hover_color="#C41E3A",
            command=self._delete_model
        )
        delete_model_btn.pack(side="left")

        # 加载第一个模型
        if models:
            self.model_combobox.set(models[0]["name"])
            self._load_model(models[0])

    def _get_models(self):
        return self.config_manager.get_models()

    def _on_model_selected(self, model_name):
        models = self._get_models()
        for model in models:
            if model["name"] == model_name:
                self._load_model(model)
                return

    def _load_model(self, model):
        self.model_name_entry.delete(0, "end")
        self.model_name_entry.insert(0, model["name"])
        self.model_id_entry.delete(0, "end")
        self.model_id_entry.insert(0, model.get("model_id", model.get("name", "")))
        self.url_entry.delete(0, "end")
        self.url_entry.insert(0, model.get("api_url", ""))
        self.api_key_entry.delete(0, "end")
        self.api_key_entry.insert(0, model.get("api_key", ""))
        self.is_local_var.set(model.get("is_local", False))
        self.local_path_entry.delete(0, "end")
        self.local_path_entry.insert(0, model.get("local_path", ""))
        self._toggle_local_model()

    def _toggle_local_model(self):
        if self.is_local_var.get():
            self.url_entry.configure(state="disabled")
            self.api_key_entry.configure(state="disabled")
            self.local_path_entry.configure(state="normal")
        else:
            self.url_entry.configure(state="normal")
            self.api_key_entry.configure(state="normal")
            self.local_path_entry.configure(state="disabled")

    def _browse_local_model(self):
        self.lift()
        self.attributes('-topmost', True)
        self.after_idle(lambda: self.attributes('-topmost', False))

        file_path = filedialog.askopenfilename(
            parent=self,
            filetypes=[("Model Files", "*.bin;*.pt;*.pth"), ("All Files", "*.*")]
        )
        if file_path:
            self.local_path_entry.delete(0, "end")
            self.local_path_entry.insert(0, file_path)

    def _add_or_save_model(self):
        model_name = self.model_name_entry.get().strip()
        if not model_name:
            self._show_message("请输入模型名称")
            return

        api_url = self.url_entry.get().strip()
        api_key = self.api_key_entry.get().strip()
        is_local = self.is_local_var.get()
        local_path = self.local_path_entry.get().strip()

        if not is_local and (not api_url or not api_key):
            self._show_message("云端模型需要填写API URL和API Key")
            return

        if is_local and not local_path:
            self._show_message("本地模型需要选择模型文件路径")
            return

        new_model = {
            "name": model_name,
            "model_id": self.model_id_entry.get().strip(),
            "api_url": api_url,
            "api_key": api_key,
            "is_local": is_local,
            "local_path": local_path
        }

        # 检查是否已存在
        models = self._get_models()
        existing_index = None
        for i, model in enumerate(models):
            if model["name"] == model_name:
                existing_index = i
                break

        if existing_index is not None:
            models[existing_index] = new_model
            action = "更新"
        else:
            models.append(new_model)
            action = "添加"

        self.config_manager.config["models"] = models

        try:
            with open(self.config_manager.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config_manager.config, f, ensure_ascii=False, indent=2)
            self._show_message(f"模型{action}成功")
            self._update_model_combobox()
            self.model_combobox.set(model_name)
            self.master.update_model_list()
        except Exception as e:
            self._show_message(f"保存失败: {str(e)}")

    def _delete_model(self):
        model_name = self.model_name_entry.get().strip()
        if not model_name:
            self._show_message("请选择要删除的模型")
            return

        models = self._get_models()
        models = [m for m in models if m["name"] != model_name]
        self.config_manager.config["models"] = models

        try:
            with open(self.config_manager.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config_manager.config, f, ensure_ascii=False, indent=2)
            self._add_new_model()
            self._update_model_combobox()
            self._show_message("模型删除成功")
            self.master.update_model_list()
        except Exception as e:
            self._show_message(f"删除失败: {str(e)}")

    def _update_model_combobox(self):
        models = self._get_models()
        self.model_names = [m["name"] for m in models]
        if not self.model_names:
            self.model_names = ["添加新模型"]
        self.model_combobox.configure(values=self.model_names)

    def _add_new_model(self):
        self.model_name_entry.delete(0, "end")
        self.url_entry.delete(0, "end")
        self.api_key_entry.delete(0, "end")
        self.is_local_var.set(False)
        self.local_path_entry.delete(0, "end")
        self.local_path_entry.configure(state="disabled")
        self.model_name_entry.focus()

    # ===================== 向量化设置 =====================

    def _create_vector_tab(self):
        """创建向量化设置标签页"""
        vector_frame = ctk.CTkFrame(self.tab_vector, fg_color="transparent")
        vector_frame.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(
            vector_frame,
            text="向量化数据设置",
            font=("Microsoft YaHei UI", 16, "bold")
        ).pack(anchor="w", pady=(0, 20))

        # Prompt模板
        ctk.CTkLabel(
            vector_frame,
            text="Prompt拼接规则:",
            font=("Microsoft YaHei UI", 12)
        ).pack(anchor="w", pady=(0, 5))

        prompt_container = ctk.CTkFrame(vector_frame, fg_color="transparent")
        prompt_container.pack(anchor="w", pady=(5, 5), fill="both", expand=True)

        self.prompt_template = ctk.CTkTextbox(
            prompt_container,
            width=500,
            height=250,
            font=("Microsoft YaHei UI", 11)
        )
        self.prompt_template.pack(side="left", fill="both", expand=True, padx=(0, 10))

        save_btn = ctk.CTkButton(
            prompt_container,
            text="💾 保存",
            command=self._save_vector_settings,
            width=100,
            height=40,
            font=("Microsoft YaHei UI", 12, "bold"),
            fg_color="#2E8B57"
        )
        save_btn.pack(side="left", padx=(10, 0))

        default_prompt = """你是一个专门根据提供的小说原文片段回答问题的助手。

重要规则：
1. 只使用提供的原文片段中的信息回答问题
2. 如果原文片段中没有相关信息，就说"根据提供的原文片段无法回答此问题"
3. 如果有相关信息，直接引用原文片段的内容
4. 不要编造原文中没有的信息
5. 不要回答与原文无关的内容

以下是提供的原文片段：
{context}

用户的问题：{question}

根据原文片段回答："""

        saved_prompt = self.config_manager.get("prompt_template", "")
        self.prompt_template.insert("0.0", saved_prompt if saved_prompt else default_prompt)

        ctk.CTkLabel(
            vector_frame,
            text="提示：使用 {context} 表示向量化检索到的上下文，使用 {question} 表示用户问题",
            font=("Microsoft YaHei UI", 10),
            text_color="gray"
        ).pack(anchor="w", pady=(0, 15))

        # TOP-K 设置
        topk_frame = ctk.CTkFrame(vector_frame, fg_color="transparent")
        topk_frame.pack(anchor="w", pady=(5, 10), fill="x")

        ctk.CTkLabel(
            topk_frame,
            text="召回片段数量 (TOP-K):",
            font=("Microsoft YaHei UI", 11)
        ).pack(side="left", padx=(0, 10))

        self.topk_entry = ctk.CTkEntry(topk_frame, width=60, height=30)
        self.topk_entry.pack(side="left")
        saved_topk = self.config_manager.get("top_k", "3")
        self.topk_entry.insert(0, str(saved_topk))

        ctk.CTkLabel(
            topk_frame,
            text="建议值: 3-10，值越大召回越全面但可能包含噪声",
            font=("Microsoft YaHei UI", 10),
            text_color="gray"
        ).pack(side="left", padx=(10, 0))

        # 深度思考开关
        thinking_frame = ctk.CTkFrame(vector_frame, fg_color="transparent")
        thinking_frame.pack(anchor="w", pady=(10, 5), fill="x")

        self.enable_thinking_var = ctk.BooleanVar(value=self.config_manager.get("enable_thinking", False))
        thinking_switch = ctk.CTkSwitch(
            thinking_frame,
            text="启用深度思考（思维链）",
            variable=self.enable_thinking_var,
            font=("Microsoft YaHei UI", 11)
        )
        thinking_switch.pack(side="left", padx=(0, 10))

        ctk.CTkLabel(
            thinking_frame,
            text="启用后模型会先展示思考过程（仅支持DeepSeek等支持该功能的模型）",
            font=("Microsoft YaHei UI", 10),
            text_color="gray"
        ).pack(side="left")

        # 查询改写开关
        rewrite_frame = ctk.CTkFrame(vector_frame, fg_color="transparent")
        rewrite_frame.pack(anchor="w", pady=(10, 5), fill="x")

        self.enable_query_rewrite_var = ctk.BooleanVar(
            value=self.config_manager.get("enable_query_rewrite", True)
        )
        rewrite_switch = ctk.CTkSwitch(
            rewrite_frame,
            text="启用查询改写",
            variable=self.enable_query_rewrite_var,
            font=("Microsoft YaHei UI", 11)
        )
        rewrite_switch.pack(side="left", padx=(0, 10))

        ctk.CTkLabel(
            rewrite_frame,
            text="问题含「主角/男主/女主」等元词时，先调一次模型改写为具体人名再检索"
                 "（仅命中元词才发起调用，失败自动回退原查询）",
            font=("Microsoft YaHei UI", 10),
            text_color="gray"
        ).pack(side="left")

        # 嵌入模型设置
        embedding_frame = ctk.CTkFrame(vector_frame, fg_color="#2C2C2C", corner_radius=10)
        embedding_frame.pack(fill="x", pady=(15, 5))

        embedding_inner = ctk.CTkFrame(embedding_frame, fg_color="transparent")
        embedding_inner.pack(fill="both", expand=True, padx=20, pady=15)

        ctk.CTkLabel(
            embedding_inner,
            text="嵌入模型设置",
            font=("Microsoft YaHei UI", 14, "bold")
        ).pack(anchor="w", pady=(0, 15))

        ctk.CTkLabel(
            embedding_inner,
            text="嵌入模型路径（用于向量检索）:",
            font=("Microsoft YaHei UI", 12)
        ).pack(anchor="w")

        embedding_path_frame = ctk.CTkFrame(embedding_inner, fg_color="transparent")
        embedding_path_frame.pack(anchor="w", pady=(5, 10), fill="x")

        self.embedding_path_entry = ctk.CTkEntry(embedding_path_frame, width=350, height=35)
        self.embedding_path_entry.pack(side="left", padx=(0, 10))
        saved_embedding_path = self.config_manager.get("embedding_model_path", "")
        self.embedding_path_entry.insert(0, saved_embedding_path)

        browse_embedding_btn = ctk.CTkButton(
            embedding_path_frame,
            text="浏览",
            width=80,
            height=35,
            command=self._browse_embedding_model
        )
        browse_embedding_btn.pack(side="left")

        ctk.CTkLabel(
            embedding_inner,
            text="提示：嵌入模型用于将问题转换为向量进行检索，留空则使用关键词检索模式",
            font=("Microsoft YaHei UI", 10),
            text_color="gray"
        ).pack(anchor="w")

        # 保存状态
        self.save_status_label = ctk.CTkLabel(
            vector_frame,
            text="✓ 配置已加载",
            text_color="green",
            font=("Microsoft YaHei UI", 11)
        )
        self.save_status_label.pack(anchor="w", pady=(10, 0))

    def _browse_embedding_model(self):
        self.lift()
        self.attributes('-topmost', True)
        self.after_idle(lambda: self.attributes('-topmost', False))

        folder = filedialog.askdirectory(parent=self, title="选择嵌入模型文件夹")
        if folder:
            self.embedding_path_entry.delete(0, "end")
            self.embedding_path_entry.insert(0, folder)

    def _save_vector_settings(self):
        import os

        prompt_text = self.prompt_template.get("0.0", "end-1c")
        top_k = self.topk_entry.get().strip()

        self.config_manager.set("prompt_template", prompt_text)
        self.config_manager.set("top_k", top_k if top_k else "3")
        self.config_manager.set("enable_thinking", self.enable_thinking_var.get())
        self.config_manager.set("enable_query_rewrite", self.enable_query_rewrite_var.get())
        self.config_manager.set("embedding_model_path", self.embedding_path_entry.get())
        self.config_manager.save()

        RAGRetriever.clear_cache()

        self.save_status_label.configure(text=f"✓ 配置已保存", text_color="#2E8B57")
        self._show_save_message("设置已保存！")

    def _show_message(self, message: str):
        toast = ctk.CTkLabel(self, text=message, fg_color="#1E90FF", corner_radius=10)
        toast.place(relx=0.5, rely=0.9, anchor="center")
        self.after(2000, lambda: toast.destroy())

    def _show_save_message(self, message: str):
        self._show_message(message)

    def on_closing(self):
        self._save_vector_settings()
        self.config_manager.save()
        self.destroy()
