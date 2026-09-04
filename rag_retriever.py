"""
rag_retriever.py — RAG检索核心
支持从项目级向量库（embeddings.npy + metadata.json）加载和检索

检索模式:
1. 向量检索: 使用嵌入模型编码查询，计算余弦相似度
2. 关键词检索: 基于关键词匹配（回退模式）
"""
import os
import re
import json
import time
import numpy as np
from typing import List, Dict, Optional

from utils import load_embedding_model


class RAGRetriever:
    """RAG检索器"""

    _cache: Dict[str, 'RAGRetriever'] = {}

    def __init__(
        self,
        vector_path: str,
        reranker_model_path: Optional[str] = None,
        embedding_model_path: Optional[str] = None,
        vector_file: Optional[str] = None,
        metadata_file: Optional[str] = None
    ):
        self.vector_path = vector_path
        self.reranker_model_path = reranker_model_path
        self.embedding_model_path = embedding_model_path
        self.vector_file = vector_file
        self.metadata_file = metadata_file
        self.documents: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.texts: List[str] = []
        self.reranker_model = None
        self.embedding_model = None
        self._embedding_loaded = False

        self.load_documents()
        self._load_embedding_model()

    @classmethod
    def get_or_create(
        cls,
        vector_path: str,
        reranker_model_path: Optional[str] = None,
        embedding_model_path: Optional[str] = None,
        vector_file: Optional[str] = None,
        metadata_file: Optional[str] = None
    ) -> 'RAGRetriever':
        """获取或创建RAGRetriever实例（带缓存）"""
        cache_key = f"{vector_path}#{reranker_model_path or 'none'}#{embedding_model_path or 'none'}#{vector_file or 'auto'}#{metadata_file or 'auto'}"
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        instance = cls(vector_path, reranker_model_path, embedding_model_path, vector_file, metadata_file)
        cls._cache[cache_key] = instance
        return instance

    @classmethod
    def clear_cache(cls):
        """清除缓存"""
        cls._cache.clear()

    def _load_embedding_model(self):
        """加载嵌入模型（本地目录 或 HF 模型名自动下载；失败则回退关键词检索）"""
        if self._embedding_loaded:
            return

        self._embedding_loaded = True

        if self.embedding_model_path:
            self.embedding_model = load_embedding_model(self.embedding_model_path)
            if self.embedding_model is not None:
                print(f"✓ 嵌入模型已加载: {self.embedding_model_path}")
            else:
                print(f"⚠ 嵌入模型不可用: {self.embedding_model_path}，使用关键词检索")
        else:
            print("⚠ 未配置嵌入模型，将使用关键词检索模式")
            self.embedding_model = None

    def load_documents(self):
        """加载文档和向量数据"""
        if not self.vector_path or not os.path.exists(self.vector_path):
            print(f"⚠ 向量路径不存在: {self.vector_path}")
            return

        print(f"📚 正在加载向量数据: {self.vector_path}")
        self.documents = []
        self.texts = []

        # 确定要加载的文件
        if self.vector_file and self.metadata_file:
            # 使用指定的文件名
            embeddings_file = os.path.join(self.vector_path, self.vector_file)
            metadata_file = os.path.join(self.vector_path, self.metadata_file)
            
            if os.path.isfile(embeddings_file) and os.path.isfile(metadata_file):
                self._load_project_vector_db(embeddings_file, metadata_file)
            else:
                print(f"⚠ 指定的文件不存在: {embeddings_file} 或 {metadata_file}")
                self._load_directory_vector_db()
        else:
            # 尝试标准格式
            embeddings_file = os.path.join(self.vector_path, "embeddings.npy")
            metadata_file = os.path.join(self.vector_path, "metadata.json")

            if os.path.isfile(embeddings_file) and os.path.isfile(metadata_file):
                # 标准项目级向量库格式
                self._load_project_vector_db(embeddings_file, metadata_file)
            elif os.path.isdir(self.vector_path):
                # 目录格式 - 尝试加载所有文件
                self._load_directory_vector_db()
            elif os.path.isfile(self.vector_path):
                # 单文件格式
                self._load_single_file(self.vector_path)

        # 对齐向量和文本数量
        if self.embeddings is not None and len(self.texts) > 0:
            min_len = min(len(self.texts), len(self.embeddings))
            if min_len < len(self.texts):
                self.texts = self.texts[:min_len]
                self.documents = self.documents[:min_len]
            if min_len < len(self.embeddings):
                self.embeddings = self.embeddings[:min_len]

        print(f"✓ 已加载 {len(self.documents)} 个文档片段")

    def _load_project_vector_db(self, embeddings_file: str, metadata_file: str):
        """加载项目级标准向量库"""
        try:
            # 加载向量
            self.embeddings = np.load(embeddings_file)
            if self.embeddings.ndim == 2:
                print(f"  向量维度: {self.embeddings.shape[0]} x {self.embeddings.shape[1]}")
            else:
                self.embeddings = None
                print("  ⚠ 向量文件格式不正确")

            # 加载元数据
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata_list = json.load(f)

            for item in metadata_list:
                text = item.get("text", "")
                chapter = item.get("chapter", "")
                chunk_id = item.get("chunk_id", 0)

                self.texts.append(text)
                self.documents.append({
                    "text": text,
                    "file": chapter,
                    "chunk_id": chunk_id,
                    "chapter": chapter
                })

            print(f"  元数据: {len(metadata_list)} 条")

        except Exception as e:
            print(f"  ✗ 加载项目向量库失败: {e}")
            self.embeddings = None

    def _load_directory_vector_db(self):
        """从目录加载向量库"""
        npy_files = []
        txt_files = []
        json_files = []

        for file in os.listdir(self.vector_path):
            if file.endswith('.npy'):
                npy_files.append(file)
            elif file.endswith('.txt') or file.endswith('.md'):
                txt_files.append(file)
            elif file.endswith('.json'):
                json_files.append(file)

        # 优先加载标准格式
        for npy_file in npy_files:
            file_path = os.path.join(self.vector_path, npy_file)
            self._load_single_file(file_path)

        # 加载JSON元数据
        for json_file in json_files:
            file_path = os.path.join(self.vector_path, json_file)
            self._load_json_file(file_path)

        # 加载纯文本
        for txt_file in txt_files:
            txt_name = txt_file.replace('.txt', '.npy').replace('.md', '.npy')
            if txt_name not in npy_files:
                file_path = os.path.join(self.vector_path, txt_file)
                self._load_text_file(file_path)

    def _load_single_file(self, file_path: str):
        """加载单个文件"""
        try:
            if file_path.endswith('.npy'):
                self._load_npy_file(file_path)
            elif file_path.endswith('.json'):
                self._load_json_file(file_path)
            else:
                self._load_text_file(file_path)
        except Exception as e:
            print(f"  ⚠ 加载文件失败 {file_path}: {e}")

    def _load_npy_file(self, file_path: str):
        """加载NPY向量文件"""
        try:
            data = np.load(file_path, allow_pickle=True)

            if isinstance(data, np.ndarray):
                if data.ndim == 2 and data.dtype == np.float32:
                    self.embeddings = data
                    print(f"  加载向量: {data.shape[0]} 个向量，维度 {data.shape[1]}")
                else:
                    # 可能包含文本数据
                    for i, item in enumerate(data):
                        if isinstance(item, dict):
                            text = item.get('text', str(item))
                            self.texts.append(text)
                            self.documents.append({
                                'text': text,
                                'file': os.path.basename(file_path),
                                'chunk_id': i + 1,
                                'chapter': item.get('chapter', f"片段 {i+1}")
                            })
                        elif isinstance(item, str):
                            self.texts.append(item)
                            self.documents.append({
                                'text': item,
                                'file': os.path.basename(file_path),
                                'chunk_id': i + 1,
                                'chapter': f"片段 {i+1}"
                            })
        except Exception as e:
            print(f"  ⚠ 加载NPY文件失败: {e}")

    def _load_json_file(self, file_path: str):
        """加载JSON元数据文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, dict):
                        text = item.get('text', item.get('content', str(item)))
                        chapter = item.get('chapter', item.get('title', f"片段 {i+1}"))
                        self.texts.append(text)
                        self.documents.append({
                            'text': text,
                            'file': os.path.basename(file_path),
                            'chunk_id': i + 1,
                            'chapter': chapter
                        })
                    elif isinstance(item, str):
                        self.texts.append(item)
                        self.documents.append({
                            'text': item,
                            'file': os.path.basename(file_path),
                            'chunk_id': i + 1,
                            'chapter': f"片段 {i+1}"
                        })

                print(f"  加载JSON元数据: {len(data)} 条")
        except Exception as e:
            print(f"  ⚠ 加载JSON文件失败: {e}")

    def _load_text_file(self, file_path: str):
        """加载纯文本文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            chunks = self._split_into_chunks(content)
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    self.texts.append(chunk.strip())
                    self.documents.append({
                        'text': chunk.strip(),
                        'file': os.path.basename(file_path),
                        'chunk_id': i + 1,
                        'chapter': f"{os.path.basename(file_path)} - {i + 1}"
                    })
        except Exception as e:
            print(f"  ⚠ 加载文本文件失败: {e}")

    def _split_into_chunks(self, text: str, chunk_size: int = 500) -> List[str]:
        """将文本分割成块"""
        chunks = []
        paragraphs = re.split(r'\n\s*\n', text)

        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para
            else:
                current_chunk += "\n" + para if current_chunk else para

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot_product / (norm1 * norm2))

    def _batch_cosine_similarity(self, query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """批量计算余弦相似度"""
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return np.zeros(matrix.shape[0])

        matrix_norms = np.linalg.norm(matrix, axis=1)
        dot_products = matrix @ query_vec

        with np.errstate(divide='ignore', invalid='ignore'):
            similarities = dot_products / (matrix_norms * query_norm)

        similarities[np.isnan(similarities)] = 0.0
        return similarities

    def retrieve(
        self,
        question: str,
        top_k: int = 3,
        enable_rerank: bool = False
    ) -> List[Dict]:
        """
        检索相关文档

        Args:
            question: 查询问题
            top_k: 返回的 top K 结果数
            enable_rerank: 是否启用重排序

        Returns:
            相关文档列表，每项包含 {text, score, chapter, chunk_id, file}
        """
        if not self.documents:
            print("⚠ 没有加载文档，无法检索")
            return []

        start_time = time.time()

        # 根据可用模式选择检索方法
        if self.embeddings is not None and len(self.texts) > 0 and self.embedding_model is not None:
            results = self._vector_retrieve(question, top_k * 3)
        else:
            results = self._keyword_retrieve(question, top_k * 3)

        retrieve_time = time.time() - start_time

        # 截取 top_k 结果
        final_results = results[:top_k]
        print(f"✓ 检索完成: {len(final_results)} 个结果，耗时 {retrieve_time:.3f}秒")

        return final_results

    @staticmethod
    def _expand_cjk_keywords(tokens: set) -> set:
        """
        中文关键词扩展：对长度 > 2 的连续中文 token 追加字二元组（bigram）。

        原因：连续中文没有空格分词，若把整句当作一个 token，原文很难精确包含
        整句，会导致漏召回。二元组能显著提升中文关键词命中率。
        """
        expanded = set(tokens)
        for token in tokens:
            if len(token) > 2 and re.fullmatch(r'[\u4e00-\u9fff]+', token):
                expanded.update(token[i:i + 2] for i in range(len(token) - 1))
        return expanded

    def _keyword_retrieve(self, question: str, top_k: int) -> List[Dict]:
        """关键词检索"""
        tokens = set(re.findall(r'[\w\u4e00-\u9fff]+', question.lower()))
        question_words = self._expand_cjk_keywords(tokens)
        results = []

        for doc in self.documents:
            doc_text = doc['text'].lower()
            score = 0

            for word in question_words:
                if word in doc_text:
                    score += 1
                    # 位置加权：靠前的匹配加分
                    position = doc_text.find(word)
                    if position < 100:
                        score += 0.5
                    elif position < 500:
                        score += 0.25

            if score > 0:
                results.append({
                    'score': score,
                    **doc
                })

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def _vector_retrieve(self, question: str, top_k: int) -> List[Dict]:
        """向量检索"""
        try:
            if not self.embedding_model:
                return self._keyword_retrieve(question, top_k)

            # 编码查询
            query_embedding = self.embedding_model.encode(
                question,
                normalize_embeddings=True,
                convert_to_numpy=True
            )

            # 计算相似度
            if isinstance(self.embeddings, np.ndarray) and self.embeddings.ndim == 2:
                sim_scores = np.dot(self.embeddings, query_embedding)
                top_indices = np.argsort(sim_scores)[::-1][:top_k]
                similarities = [(float(sim_scores[i]), int(i)) for i in top_indices]
            else:
                # 逐条计算
                similarities = []
                for i in range(min(len(self.embeddings), len(self.documents))):
                    doc_embedding = self.embeddings[i] if i < len(self.embeddings) else None
                    if doc_embedding is not None and isinstance(doc_embedding, np.ndarray):
                        if len(doc_embedding) == len(query_embedding):
                            sim = self._cosine_similarity(query_embedding, doc_embedding)
                            similarities.append((sim, i))

                similarities.sort(key=lambda x: x[0], reverse=True)

            # 构建结果
            results = []
            for similarity, idx in similarities[:top_k]:
                if similarity > 0.1:  # 最低相似度阈值
                    doc = self.documents[idx] if idx < len(self.documents) else {"text": "", "chapter": "", "chunk_id": 0, "file": ""}
                    results.append({
                        'score': round(similarity, 4),
                        **doc
                    })

            return results

        except Exception as e:
            print(f"⚠ 向量检索失败: {e}，回退到关键词检索")
            return self._keyword_retrieve(question, top_k)

    def get_stats(self) -> Dict:
        """获取检索器统计信息"""
        return {
            'vector_path': self.vector_path,
            'document_count': len(self.documents),
            'has_embeddings': self.embeddings is not None,
            'embedding_dim': self.embeddings.shape[1] if self.embeddings is not None and self.embeddings.ndim == 2 else 0,
            'has_embedding_model': self.embedding_model is not None,
            'has_reranker': self.reranker_model is not None,
        }
