"""
chat_logger.py — 聊天记录日志模块
支持按项目分目录存储对话记录

目录结构:
chat_logs/
  ├── {项目名称}/
  │   ├── 2026-07-25.md
  │   └── 2026-07-25.txt
  └── default/
      └── ...
"""
from datetime import datetime
from pathlib import Path
import re
from typing import List, Dict, Optional

from utils import get_chat_logs_dir


class ChatLogger:
    """聊天记录日志管理器"""

    def __init__(self, project_name: Optional[str] = None):
        """
        初始化聊天日志管理器

        Args:
            project_name: 项目名称，为None时使用默认目录
        """
        self.set_project(project_name)

    def set_project(self, project_name: Optional[str]):
        """
        设置当前项目

        Args:
            project_name: 项目名称
        """
        if project_name:
            self.logs_dir = get_chat_logs_dir() / self._sanitize_name(project_name)
        else:
            self.logs_dir = get_chat_logs_dir() / "default"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize_name(self, name: str) -> str:
        """清理项目名称中的特殊字符"""
        return re.sub(r'[^\w\u4e00-\u9fff-]', '_', name)

    def save_message(self, role: str, content: str):
        """
        保存单条聊天记录

        Args:
            role: 角色 (user/assistant/system)
            content: 消息内容
        """
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            log_file = self.logs_dir / f"{today}.md"
            timestamp = datetime.now().strftime("%H:%M:%S")

            with open(log_file, "a", encoding="utf-8") as f:
                if role == "user":
                    f.write(f"## [{timestamp}] 👤 用户\n\n{content}\n\n---\n\n")
                elif role == "assistant":
                    f.write(f"## [{timestamp}] 🤖 AI\n\n{content}\n\n---\n\n")
                elif role == "system":
                    f.write(f"## [{timestamp}] ⚙ 系统\n\n{content}\n\n---\n\n")
                else:
                    f.write(f"## [{timestamp}] {role}\n\n{content}\n\n---\n\n")
        except Exception as e:
            print(f"⚠ 保存聊天记录失败: {e}")

    def load_today_history(self) -> List[Dict]:
        """加载今天的聊天记录"""
        today = datetime.now().strftime("%Y-%m-%d")
        return self._load_history_by_date(today)

    def load_all_history(self) -> List[Dict]:
        """加载所有历史聊天记录"""
        all_messages = []

        try:
            if not self.logs_dir.exists():
                return []

            log_files = sorted(self.logs_dir.glob("*.md"))

            for log_file in log_files:
                date_messages = self._load_history_from_file(log_file)
                all_messages.extend(date_messages)

            # 按时间戳排序
            all_messages.sort(key=lambda x: x.get("timestamp", ""))

            return all_messages

        except Exception as e:
            print(f"⚠ 加载聊天记录失败: {e}")
            return []

    def get_session_messages(self, max_turns: int = 15) -> List[Dict]:
        """
        获取当前会话的消息（用于多轮对话上下文）

        Args:
            max_turns: 最大轮数

        Returns:
            消息列表，按时间排序
        """
        history = self.load_today_history()

        # 如果消息不足，也可以加载最近几天的
        if len(history) < 2:
            history = self.load_all_history()

        # 限制轮数（每轮包含user+assistant两条消息）
        max_messages = max_turns * 2
        if len(history) > max_messages:
            history = history[-max_messages:]

        return history

    def _load_history_by_date(self, date: str) -> List[Dict]:
        """加载指定日期的聊天记录"""
        log_file = self.logs_dir / f"{date}.md"
        return self._load_history_from_file(log_file)

    def _load_history_from_file(self, log_file: Path) -> List[Dict]:
        """从指定文件加载聊天记录"""
        messages = []

        try:
            if not log_file.exists():
                return []

            with open(log_file, "r", encoding="utf-8") as f:
                content = f.read()

            # 匹配消息标题
            pattern = r'^## \[\d{2}:\d{2}:\d{2}\] (👤 用户|🤖 AI|⚙ 系统)'
            matches = list(re.finditer(pattern, content, re.MULTILINE))

            for i, match in enumerate(matches):
                role_text = match.group(1)
                if "👤 用户" in role_text:
                    role = "user"
                elif "🤖 AI" in role_text:
                    role = "assistant"
                else:
                    role = "system"

                timestamp_match = re.search(r'\[(\d{2}:\d{2}:\d{2})\]', match.group(0))
                timestamp = timestamp_match.group(1) if timestamp_match else ""
                date_str = log_file.stem
                full_timestamp = f"{date_str} {timestamp}"

                # 提取消息内容
                start_pos = match.end()
                if i + 1 < len(matches):
                    end_pos = matches[i + 1].start()
                else:
                    end_pos = len(content)

                msg_content = content[start_pos:end_pos]
                msg_content = re.sub(r'^\n+', '', msg_content)
                msg_content = re.sub(r'\n+---\s*$', '', msg_content)
                msg_content = msg_content.strip()

                if msg_content:
                    messages.append({
                        "role": role,
                        "content": msg_content,
                        "timestamp": full_timestamp
                    })

            return messages

        except Exception as e:
            print(f"⚠ 加载聊天记录失败: {e}")
            return []

    def get_history_file_list(self) -> List[str]:
        """获取所有历史记录文件名"""
        try:
            if not self.logs_dir.exists():
                return []
            files = sorted(self.logs_dir.glob("*.md"))
            return [f.stem for f in files]
        except Exception:
            return []

    def clear_today_history(self):
        """清空今天的聊天记录"""
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = self.logs_dir / f"{today}.md"
        if log_file.exists():
            try:
                log_file.unlink()
                return True
            except Exception as e:
                print(f"⚠ 删除历史记录失败: {e}")
        return False

    def clear_all_history(self):
        """清空所有聊天记录"""
        try:
            if self.logs_dir.exists():
                for f in self.logs_dir.glob("*.md"):
                    f.unlink()
                return True
        except Exception as e:
            print(f"⚠ 清空历史记录失败: {e}")
        return False
