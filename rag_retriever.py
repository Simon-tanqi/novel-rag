"""
rag_retriever.py — RAG检索核心
支持从项目级向量库（embeddings.npy + metadata.json）加载和检索

检索模式（互斥选择，二选一）:
1. 向量检索: 使用嵌入模型编码查询，与已存向量计算余弦相似度（默认，需本地嵌入模型）
2. 关键词检索: 基于关键词匹配（未配置嵌入模型 / 模型加载失败时的降级路径）

说明（2026-09 重构）:
- 移除了旧版多条未使用的文件加载路径（_load_single_file / _load_npy_file /
  _load_json_file / _load_text_file / _split_into_chunks），统一走
  「标准项目级向量库」格式（embeddings.npy + metadata.json）；
- 目录导入场景仍保留：兼容“目录中散落多个 npy/json/txt”的旧数据目录；
- 修正不一致：旧逻辑“有向量但无嵌入模型”时也走关键词——语义与
  “有向量”自相矛盾，现统一按  embedding_model 是否可用 决定检索通道。
"""
import os
import re
import json
import time
import numpy as np
from typing import List, Dict, Optional

from utils import load_embedding_model, load_reranker_model
from novel_context import chapter_aggregate_rerank


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
        self._reranker_loaded = False

        self.load_documents()
        self._load_embedding_model()
        self._load_reranker()

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
            self.embedding_model, device = load_embedding_model(self.embedding_model_path)
            if self.embedding_model is not None:
                print(f"✓ 嵌入模型已加载: {self.embedding_model_path}  [设备: {device}]")
            else:
                print(f"⚠ 嵌入模型不可用: {self.embedding_model_path}，使用关键词检索")
        else:
            print("⚠ 未配置嵌入模型，将使用关键词检索模式")
            self.embedding_model = None
            device = "cpu"

    def _load_reranker(self):
        """加载重排模型（CrossEncoder，仅在 reranker_model_path 配置时加载）"""
        if self._reranker_loaded:
            return
        self._reranker_loaded = True
        if not self.reranker_model_path:
            return
        self.reranker_model, device = load_reranker_model(self.reranker_model_path)
        if self.reranker_model is not None:
            print(f"✓ 重排模型已加载: {self.reranker_model_path}  [设备: {device}]")
        else:
            print(f"⚠ 重排模型不可用: {self.reranker_model_path}，将跳过精排")

    # ===================== 数据加载 =====================

    def load_documents(self):
        """加载文档和向量数据（统一按项目级标准格式加载）"""
        if not self.vector_path or not os.path.exists(self.vector_path):
            print(f"⚠ 向量路径不存在: {self.vector_path}")
            return

        print(f"📚 正在加载向量数据: {self.vector_path}")
        self.documents = []
        self.texts = []

        # 1) 显式指定的文件名（向量库导入场景）
        if self.vector_file and self.metadata_file:
            embeddings_file = os.path.join(self.vector_path, self.vector_file)
            metadata_file = os.path.join(self.vector_path, self.metadata_file)
            if os.path.isfile(embeddings_file) and os.path.isfile(metadata_file):
                self._load_project_vector_db(embeddings_file, metadata_file)
            else:
                print(f"⚠ 指定的向量文件不存在: {embeddings_file} 或 {metadata_file}，尝试目录扫描")
                self._load_directory_vector_db()
            self._align()
            return

        # 2) 标准项目级向量库格式
        embeddings_file = os.path.join(self.vector_path, "embeddings.npy")
        metadata_file = os.path.join(self.vector_path, "metadata.json")
        if os.path.isfile(embeddings_file) and os.path.isfile(metadata_file):
            self._load_project_vector_db(embeddings_file, metadata_file)
            self._align()
            return

        # 3) 目录/文件兼容加载（旧数据目录或单文件）
        if os.path.isdir(self.vector_path):
            self._load_directory_vector_db()
        elif os.path.isfile(self.vector_path):
            self._load_single_file(self.vector_path)
        self._align()

    def _align(self):
        """对齐向量与文本数量（截断到较小者）并确保向量已 L2 归一化

        检索用 np.dot 当余弦相似度计算，前提是向量已归一化；
        新入库向量在 step2 已归一化，这里对旧库/导入库做兜底归一化。
        """
        if self.embeddings is not None and len(self.texts) > 0:
            min_len = min(len(self.texts), len(self.embeddings))
            if min_len < len(self.texts):
                self.texts = self.texts[:min_len]
                self.documents = self.documents[:min_len]
            if min_len < len(self.embeddings):
                self.embeddings = self.embeddings[:min_len]
        if isinstance(self.embeddings, np.ndarray) and self.embeddings.ndim == 2:
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0  # 避免零向量除零
            self.embeddings = self.embeddings / norms
        print(f"✓ 已加载 {len(self.documents)} 个文档片段")

    def _load_project_vector_db(self, embeddings_file: str, metadata_file: str):
        """加载项目级标准向量库"""
        try:
            self.embeddings = np.load(embeddings_file)
            if self.embeddings.ndim == 2:
                print(f"  向量维度: {self.embeddings.shape[0]} x {self.embeddings.shape[1]}")
            else:
                self.embeddings = None
                print("  ⚠ 向量文件格式不正确（非二维数组）")

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
                    "chapter": chapter,
                })
            print(f"  元数据: {len(metadata_list)} 条")
        except Exception as e:
            print(f"  ✗ 加载项目向量库失败: {e}")
            self.embeddings = None

    def _load_directory_vector_db(self):
        """目录兼容加载：扫描目录下的 npy / json / txt，尽量两两配对"""
        try:
            entries = sorted(os.listdir(self.vector_path))
        except OSError as e:
            print(f"  ⚠ 目录扫描失败: {e}")
            return

        npy_files = [f for f in entries if f.endswith('.npy')]
        txt_files = [f for f in entries if f.endswith(('.txt', '.md'))]
        json_files = [f for f in entries if f.endswith('.json')]

        # 优先加载“成对”的 npy + 同名 json（旧版导出格式，如 xxx.npy + xxx.json）
        loaded_pair = False
        for npy_file in npy_files:
            base = os.path.splitext(npy_file)[0]
            json_candidates = [f for f in json_files
                               if os.path.splitext(f)[0] == base]
            json_file = json_candidates[0] if json_candidates else None
            if json_file:
                npy_path = os.path.join(self.vector_path, npy_file)
                json_path = os.path.join(self.vector_path, json_file)
                self._load_pair(npy_path, json_path)
                loaded_pair = True
        if loaded_pair:
            # 兼容旧项目：vector_db 下可能同时有标准格式残留，交给上层去重
            return

        # 标准格式优先（embeddings.npy + metadata.json）
        std_emb = os.path.join(self.vector_path, "embeddings.npy")
        std_meta = os.path.join(self.vector_path, "metadata.json")
        if os.path.isfile(std_emb) and os.path.isfile(std_meta):
            self._load_project_vector_db(std_emb, std_meta)
            return

        # 逐个加载（旧目录格式：任意 npy / txt / json）
        for npy_file in npy_files:
            self._load_single_file(os.path.join(self.vector_path, npy_file))
        for txt_file in txt_files:
            if os.path.splitext(txt_file)[0] + '.npy' not in npy_files:
                self._load_single_file(os.path.join(self.vector_path, txt_file))
        for json_file in json_files:
            self._load_single_file(os.path.join(self.vector_path, json_file))

    def _load_pair(self, npy_path: str, json_path: str):
        """加载成对向量库（npy 向量 + json 元数据）"""
        try:
            data = np.load(npy_path)
            if not (isinstance(data, np.ndarray) and data.ndim == 2
                    and data.dtype == np.float32):
                print(f"  ⚠ 跳过非标准向量文件: {npy_path}")
                return
            with open(json_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            if not isinstance(meta, list):
                meta = []
            self.embeddings = data
            for item in meta:
                if isinstance(item, dict):
                    text = item.get('text', item.get('content', ''))
                    chapter = item.get('chapter', item.get('title', ''))
                    chunk_id = item.get('chunk_id', 0)
                    self.texts.append(text)
                    self.documents.append({
                        'text': text,
                        'file': chapter,
                        'chunk_id': chunk_id,
                        'chapter': chapter or f"片段 {len(self.texts)}",
                    })
                elif isinstance(item, str):
                    self.texts.append(item)
                    self.documents.append({
                        'text': item,
                        'file': os.path.basename(npy_path),
                        'chunk_id': len(self.texts),
                        'chapter': f"片段 {len(self.texts)}",
                    })
        except Exception as e:
            print(f"  ⚠ 加载向量库对失败 {npy_path}: {e}")

    def _load_single_file(self, file_path: str):
        """加载单个文件（按扩展名分派）"""
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
        """加载 NPY 向量文件（纯向量文件无元数据时，仅记录形状）"""
        try:
            data = np.load(file_path, allow_pickle=True)
            if isinstance(data, np.ndarray) and data.ndim == 2 \
                    and data.dtype == np.float32:
                self.embeddings = data
                print(f"  加载向量: {data.shape[0]} 个向量，维度 {data.shape[1]}")
            elif isinstance(data, np.ndarray):
                # 可能内嵌文本对象（旧格式）
                for i, item in enumerate(data):
                    if isinstance(item, dict):
                        text = item.get('text', str(item))
                        self.texts.append(text)
                        self.documents.append({
                            'text': text,
                            'file': os.path.basename(file_path),
                            'chunk_id': i + 1,
                            'chapter': item.get('chapter', f"片段 {i+1}"),
                        })
                    elif isinstance(item, str):
                        self.texts.append(item)
                        self.documents.append({
                            'text': item,
                            'file': os.path.basename(file_path),
                            'chunk_id': i + 1,
                            'chapter': f"片段 {i+1}",
                        })
        except Exception as e:
            print(f"  ⚠ 加载NPY文件失败: {e}")

    def _load_json_file(self, file_path: str):
        """加载 JSON 元数据文件"""
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
                            'chapter': chapter,
                        })
                    elif isinstance(item, str):
                        self.texts.append(item)
                        self.documents.append({
                            'text': item,
                            'file': os.path.basename(file_path),
                            'chunk_id': i + 1,
                            'chapter': f"片段 {i+1}",
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
                        'chapter': f"{os.path.basename(file_path)} - {i + 1}",
                    })
        except Exception as e:
            print(f"  ⚠ 加载文本文件失败: {e}")

    def _split_into_chunks(self, text: str, chunk_size: int = 500) -> List[str]:
        """将文本分割成块（仅用于旧格式 txt 目录加载）"""
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

    # ===================== 检索 =====================

    def retrieve(
        self,
        question: str,
        top_k: int = 3,
        enable_rerank: bool = False
    ) -> List[Dict]:
        """
        检索相关文档（混合召回 + 可选精排）

        阶段1: 混合召回
            - 向量召远 (top_k*5)
            - 关键词召远 (top_k*5)
            - RRF 融合（k=60）→ 取 top_k*3
        阶段2: 精排（仅在 enable_rerank=True 且 reranker_model 已加载时）
            - CrossEncoder 重排所有候选
            - 按 rerank score 降序排
        阶段3: 章节聚合重排
            - 同章去重、取每章最高分
            - 截断 top_k

        Args:
            question: 查询问题
            top_k: 返回的 top K 结果数
            enable_rerank: 是否启用 CrossEncoder 精排

        Returns:
            相关文档列表，每项包含 {text, score, chapter, chunk_id, file}
        """
        if not self.documents:
            print("⚠ 没有加载文档，无法检索")
            return []

        start_time = time.time()
        candidate_pool_size = top_k * 5  # 召远阶段取多一些
        rerank_input_size = top_k * 3    # 精排阶段最多这么多 pair

        # ---------- 阶段1: 混合召远（RRF） ----------
        vec_results: List[Dict] = []
        if self.embedding_model is not None and isinstance(self.embeddings, np.ndarray) and self.embeddings.ndim == 2:
            vec_results = self._vector_retrieve(question, candidate_pool_size)
        kw_results: List[Dict] = self._keyword_retrieve(question, candidate_pool_size)

        if vec_results and kw_results:
            fused = self._rrf_fusion(vec_results, kw_results, k=60)
            fusion_mode = "RRF（向量+关键词）"
        elif vec_results:
            fused = vec_results
            fusion_mode = "仅向量"
        else:
            fused = kw_results
            fusion_mode = "仅关键词"

        # 截断到精排输入大小
        candidates = fused[:rerank_input_size]

        # ---------- 阶段2: CrossEncoder 精排 ----------
        if enable_rerank and self.reranker_model is not None and candidates:
            candidates = self._rerank(question, candidates)
            rerank_mode = f"CrossEncoder 精排（{len(candidates)}→{top_k}）"
        else:
            rerank_mode = "未精排（使用召远/融合分数）"

        # ---------- 阶段3: 章节聚合重排 + 截断 ----------
        final_results = chapter_aggregate_rerank(candidates, top_k)
        retrieve_time = time.time() - start_time
        print(
            f"✓ 检索完成: {fusion_mode} → {rerank_mode} → 章节聚合 → top{top_k}，"
            f"耗时 {retrieve_time:.3f}秒"
        )
        return final_results

    def _rrf_fusion(
        self,
        vec_results: List[Dict],
        kw_results: List[Dict],
        k: int = 60
    ) -> List[Dict]:
        """
        Reciprocal Rank Fusion（倒数排名融合）。

        对每个召远器，文档 d 的 RRF 分数 = 1 / (k + rank(d))，
        最终分数 = 各召远器 RRF 分数之和。常数 k=60 为论文推荐值。

        同一文档可能被两个召远器都返回，按 chunk_id 去重合并。
        """
        scores: Dict[int, float] = {}
        docs: Dict[int, Dict] = {}

        for rank, doc in enumerate(vec_results):
            cid = doc.get('chunk_id', id(doc))
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            docs[cid] = doc

        for rank, doc in enumerate(kw_results):
            cid = doc.get('chunk_id', id(doc))
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            if cid not in docs:
                docs[cid] = doc

        # 按 RRF 分数降序排
        sorted_cids = sorted(scores.keys(), key=lambda x: -scores[x])
        fused: List[Dict] = []
        for cid in sorted_cids:
            merged = dict(docs[cid])
            merged['score'] = round(scores[cid], 6)  # 融合后统一为 RRF 分数
            fused.append(merged)
        return fused

    def _rerank(self, question: str, results: List[Dict]) -> List[Dict]:
        """用 CrossEncoder 重排候选文档。失败时返回原顺序。"""
        if not results:
            return results
        pairs = [[question, r['text']] for r in results]
        try:
            scores = self.reranker_model.predict(
                pairs,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except TypeError:
            # 某些版本不接 show_progress_bar
            scores = self.reranker_model.predict(pairs, convert_to_numpy=True)
        except Exception as e:
            print(f"⚠ CrossEncoder 重排失败: {e}，返回召远原顺序")
            return results

        reranked: List[Dict] = []
        for r, s in zip(results, scores):
            doc = dict(r)
            doc['score'] = round(float(s), 4)
            reranked.append(doc)
        reranked.sort(key=lambda x: -x['score'])
        return reranked

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
            # 匹配用增广文本（注入前缀 + 原文）→ 指代片段可被主角名命中
            prefix = doc.get('coref_prefix', '') or ''
            doc_text = (prefix + doc['text']).lower()
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
        """向量检索（需嵌入模型 + 二维向量库）"""
        try:
            if not self.embedding_model:
                return self._keyword_retrieve(question, top_k)

            if not isinstance(self.embeddings, np.ndarray) \
                    or self.embeddings.ndim != 2:
                print("⚠ 向量库不是二维数组，回退到关键词检索")
                return self._keyword_retrieve(question, top_k)

            # 编码查询（注入侧已含主角名前缀，查询含主角名即可命中）
            query_embedding = self.embedding_model.encode(
                question,
                normalize_embeddings=True,
                convert_to_numpy=True
            )

            # 批量余弦相似度：矩阵点积（向量库已归一化时即余弦值）
            sim_scores = np.dot(self.embeddings, query_embedding)
            top_indices = np.argsort(sim_scores)[::-1][:top_k]

            # 构建结果
            results = []
            for i in top_indices:
                similarity = float(sim_scores[i])
                if similarity > 0.1:  # 最低相似度阈值
                    doc = self.documents[i] if i < len(self.documents) \
                        else {"text": "", "chapter": "", "chunk_id": 0, "file": ""}
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
