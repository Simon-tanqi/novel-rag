"""
message_bubble.py — 聊天气泡UI组件
完全复用自原版，提供用户消息和AI消息的气泡显示
"""
import customtkinter as ctk
from tkinter import Menu


class MessageBubble(ctk.CTkFrame):
    """聊天气泡组件"""

    def __init__(self, master, message: str, is_user: bool = True, **kwargs):
        """
        初始化消息气泡

        Args:
            master: 父容器
            message: 消息文本
            is_user: 是否为用户消息
        """
        super().__init__(master, **kwargs)
        self.message = message
        self.is_user = is_user
        self.master = master

        self.configure(fg_color="transparent")

        # 设置颜色
        if is_user:
            bg_color = "#4A90D9"
            text_color = "white"
        else:
            bg_color = "#E0E0E0"
            text_color = "#333333"

        # 创建气泡框架
        self.bubble_frame = ctk.CTkFrame(self, fg_color=bg_color, corner_radius=15)

        # 计算最大宽度
        parent_width = self.master.winfo_width() if self.master else 800
        max_width = min(int(parent_width * 0.75), 650)
        max_width = max(max_width, 350)

        # 创建消息文本
        self.message_text = ctk.CTkLabel(
            self.bubble_frame,
            text=message,
            font=("Microsoft YaHei UI", 12),
            text_color=text_color,
            wraplength=max_width - 20,
            justify="left",
            anchor="w"
        )
        self.message_text.pack(padx=12, pady=8, fill="x")

        # 添加右键菜单
        self.bubble_frame.bind("<Button-3>", self._show_context_menu)
        self.message_text.bind("<Button-3>", self._show_context_menu)

        # 布局
        if is_user:
            self.bubble_frame.pack(anchor="e", padx=10, pady=5, fill="none")
        else:
            self.bubble_frame.pack(anchor="w", padx=10, pady=5, fill="none")

    def _show_context_menu(self, event):
        """显示右键菜单"""
        menu = Menu(self.master, tearoff=0)
        menu.add_command(label="复制", command=self._copy_message)
        menu.post(event.x_root, event.y_root)

    def _copy_message(self):
        """复制消息到剪贴板"""
        self.clipboard_clear()
        self.clipboard_append(self.message)
        self.update()

        # 显示复制成功提示
        toast = ctk.CTkLabel(self.master, text="已复制", fg_color="#2E8B57", corner_radius=10)
        toast.place(relx=0.5, rely=0.95, anchor="center")
        self.master.after(1500, lambda: toast.destroy())
