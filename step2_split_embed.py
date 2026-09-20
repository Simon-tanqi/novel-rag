"""
step2_split_embed.py — 切片与向量化模块
将文本切片并构建向量索引

输出格式:
- embeddings.npy: 向量数组 (numpy float32)
- metadata.json: 元数据列表，每项包含 {text, chapter, chunk_id}
"""
import os
import json
import time
import numpy as np
from typing import List, Dict, Callable, Optional

from utils import (
    resolve_split_params,
    resolve_split_spec,
    MIN_CHUNK_CHARS,
    MAX_CHUNK_CHARS,
    DEFAULT_OVERLAP_CHARS,
    extract_chapter_title,
    resolve_embedding_model,
    load_embedding_model,
    format_timestamp,
)
from chunking import build_chunks, estimate_tokens


def _fallback_record(text: str, book_id: str = "", book_title: str = "") -> Dict:
    """极端兜底：结构解析无产出时整篇作单块（保持可检索，不丢数据）。"""
    return {
        "text": text, "prefix": "", "embed_text": text,
        "book_id": book_id, "book_title": book_title,
        "chapter_id": "c1", "chapter_title": "正文", "chapter": "正文",
        "scene_id": "", "scene_summary": "",
        "chunk_index": 1, "global_index": 0, "chunk_id": 1,
        "chunk_length": len(text), "token_count": estimate_tokens(text),
        "total_token_count": estimate_tokens(text),
        "start": 0, "end": len(text), "chapter_hash": "",
        "prev_chunk_id": None, "next_chunk_id": None,
        "parent_id": None, "coref_prefix": "",
    }


def _load_previous_index(output_dir: str) -> Optional[Dict]:
    """读取已有索引（增量重切复用）：metadata.json + embeddings.npy + chapter_index.json。

    返回 None 表示无历史索引（首次建库）。老库无 chapter_index.json 时，
    退化用 metadata 里的 chapter_id/chapter_hash 归纳章节指纹。
    """
    meta_path = os.path.join(output_dir, "metadata.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except Exception:
        return None

    vectors = None
    emb_path = os.path.join(output_dir, "embeddings.npy")
    if os.path.exists(emb_path):
        try:
            vectors = np.load(emb_path)
        except Exception:
            vectors = None

    chapters: Dict[str, str] = {}
    chapter_path = os.path.join(output_dir, "chapter_index.json")
    if os.path.exists(chapter_path):
        try:
            with open(chapter_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            chapters = {c.get("chapter_id", ""): c.get("hash", "")
                        for c in data.get("chapters", [])}
        except Exception:
            chapters = {}
    if not chapters:
        for m in metadata:
            if m.get("chapter_id") and m.get("chapter_hash"):
                chapters[m["chapter_id"]] = m["chapter_hash"]

    return {"metadata": metadata, "embeddings": vectors, "chapters": chapters}
from novel_context import inject_coref_prefix, chapter_aggregate_rerank, build_character_table


def _pick_characters(chunk_texts: List[str]) -> List[str]:
    """主角表选择：从本书原文自动挖掘高频人名（角色表按项目隔离）。

    不再使用 novel_context.DEFAULT_CHARACTERS——那是跨书混排的示例名单，
    直接把《遮天》《盘龙》的角色注入本书，会让 coref_prefix 全部标错主语
    （如本书主角「叶星」的片段被标注为「狠人」）。
    这里以 known=[] 调 build_character_table，只用从本书全文挖掘的候选。
    """
    text = "\n".join(chunk_texts)
    return build_character_table(text, known=[])


def build_vector_index(
    text: str,
    chunk_size: int = MAX_CHUNK_CHARS,
    overlap: int = DEFAULT_OVERLAP_CHARS,
    output_dir: str = "./vector_db",
    embedding_model_path: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
    target_chars: Optional[int] = None,
    overlap_ratio: Optional[float] = None,
    book_id: str = "",
    book_title: str = ""
) -> bool:
    """
    构建向量索引

    切片规格（由 utils.resolve_split_spec 唯一发放，落盘 split_spec.json）：
    - 结构预处理：统一换行、识别「第X章/卷/序章/番外」等标题、连续空行归一为场景边界；
    - 章为一级元数据，章内按空行/场景切，场景内按段落与句子聚合；
    - 块长以 token 为准：目标 400 / 硬上限 480 / 最小 150 token（≈560/672/210 字）；
      低于最小值合并相邻段落，距上次切点不足 min 时不切（即使遇句末标点）；
    - 切点优先级：空行/段落边界 > 句末标点（。！？……；）> 逗号/冒号 > 硬切，
      并在目标长度附近前后搜索最佳切点，检查引号括号闭合；
    - 相邻重叠 10%-15%（50-100 字），与句边界对齐。

    Args:
        text: 输入文本
        chunk_size: 切片硬上限（字符数，缺省 672 ≈ 480 token）
        overlap: 相邻片段重叠长度下限（字符数，缺省 50）
        output_dir: 输出目录
        embedding_model_path: 嵌入模型路径（可选）
            - 置空（None / ""）→ 纯关键词检索模式，只落 metadata.json
            - 指定但加载失败 → 建库失败返回 False，不产出任何索引文件
        progress_callback: 进度回调 (step: int, total: int, message: str, percent: float)
        target_chars: 目标块长覆盖值（None → 560 ≈ 400 token，裁剪进 [min, max]）
        overlap_ratio: 重叠比例覆盖值（None → 0.125，裁剪进 [0, 0.5]）
        book_id: 书籍标识（写入 chunk 前缀与元数据）
        book_title: 书名（写入 chunk 前缀与元数据）

    Returns:
        是否成功
    """
    # 入口校验：纯空白语料不含任何有效信息，直接判失败，避免产出 0 信息量的
    # "假成功"索引（仅含空白的语料不得落盘）。
    if not (text or "").strip():
        print("✗ 语料为空（仅含空白字符），未产出任何有效切片")
        if progress_callback:
            progress_callback(-1, 100, "失败: 语料为空", 0)
        return False

    try:
        # 创建输出目录（记录是否由本次创建：建库失败时可回收「空目录」残留）
        _dir_preexisted = os.path.isdir(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # 规格切片参数：min/max/重叠/目标长度/重叠比例由 utils 唯一出口发放。
        # 实际生效的 spec 会落盘 split_spec.json，供检索端与运维对账。
        spec = resolve_split_spec(chunk_size, overlap, target_chars, overlap_ratio)

        # 文本切片（按章节边界切分，chunk 携带真实章节标题）
        if progress_callback:
            progress_callback(0, 100, "结构预处理 + 场景分层切片...", 0)

        # ---- 结构预处理 + 场景分层 + token 切片（唯一实现：chunking.build_chunks）----
        # 章为一级元数据；章内按空行/场景切，场景内按段落与句子聚合；
        # 块长以 token 为准（目标 400 / 硬上限 480 / 最小 150 token）；
        # 切点在目标长度附近按「空行/段落 > 句末标点 > 逗号/冒号 > 硬切」搜索；
        # 相邻块重叠 10%-15%（50-100 字）且与句边界对齐。
        result = build_chunks(text, book_id=book_id, book_title=book_title, spec=spec)
        records = result.get("records", [])
        if not records:
            records = [_fallback_record(text, book_id, book_title)]
            result = {"chapters": [], "parents": [], "records": records,
                      "book_id": book_id, "book_title": book_title, "spec": spec}

        chunks = [r["text"] for r in records]
        chapter_of_chunk = [r["chapter_title"] for r in records]

        # ---- 增量重切：章节 hash 未变的章复用已有向量与 coref 前缀 ----
        # 以章为粒度（chapter_hash 指纹）：仅重切正文发生变化的章，
        # 未变章直接复用旧向量，避免整库重编码。spec 变化会改变块文本，
        # 指纹随之变化 → 自然全量重切。
        prev = _load_previous_index(output_dir)
        reuse_flags = [False] * len(records)
        reused_vectors: List[Optional[np.ndarray]] = [None] * len(records)
        reused_coref = [""] * len(records)
        reused_count = 0
        if prev:
            prev_meta = prev.get("metadata", [])
            prev_index = {}
            for i, m in enumerate(prev_meta):
                if m.get("chapter_id") and m.get("chapter_hash"):
                    prev_index[(m["chapter_id"], m["chapter_hash"], m.get("chunk_index"))] = i
            for ri, rec in enumerate(records):
                if prev.get("chapters", {}).get(rec["chapter_id"]) != rec["chapter_hash"]:
                    continue  # 该章内容已变 → 重切重编码，不复用
                oi = prev_index.get((rec["chapter_id"], rec["chapter_hash"], rec["chunk_index"]))
                if oi is None or prev_meta[oi].get("text") != rec["text"]:
                    continue
                reused_coref[ri] = prev_meta[oi].get("coref_prefix", "") or ""
                vectors = prev.get("embeddings")
                if vectors is not None and oi < len(vectors):
                    reused_vectors[ri] = vectors[oi]
                    reuse_flags[ri] = True
            reused_count = sum(1 for f in reuse_flags if f)

        # ---- 指代消解增强：为疑似指代开头的 chunk 注入主语前缀 ----
        # 注入后的增广文本仅用于向量化与关键词检索，metadata 存回原始文本，
        # 保证展示给 LLM/用户的仍是原文；前缀使含「他/她」的片段可被主角名召回。
        # 主角表从本书原文自动挖掘（known=[]），避免跨书默认表注入错误主语。
        characters = _pick_characters(chunks)
        if progress_callback:
            progress_callback(10, 100, f"主角表(自动挖掘): {'、'.join(characters[:8])}", 10)
        augmented = inject_coref_prefix(chunks, chapter_of_chunk, characters=characters)
        # 复用块的旧前缀优先，其余用本次注入结果（增广文本，仅用于编码）
        coref_texts = [
            reused_coref[i] if reused_coref[i] else (aug if aug != chunks[i] else "")
            for i, aug in enumerate(augmented)
        ]

        chunk_count = len(records)
        if progress_callback:
            progress_callback(
                20, 100,
                f"切片完成: {chunk_count} 个片段（复用 {reused_count}）",
                20,
            )

        # ---- 父块表（分层召回：子块检索 + 父块/邻块扩展）----
        # build_chunks 已按「同章连续 2-4 子块」聚合父块并写入 child_indices /
        # child_chunk_ids；章节为硬边界，父块绝不跨章。
        parent_chunks = result.get("parents", [])

        # 编码文本 = 书/章/场景前缀 + 指代增广正文
        encode_texts = []
        for rec, coref in zip(records, coref_texts):
            body = coref if coref else rec["text"]
            rec["coref_prefix"] = coref
            rec["embed_text"] = rec.get("prefix", "") + body
            encode_texts.append(rec["embed_text"])

        # 元数据：book/chapter/scene/chunk_index/起止/token/prev-next 全量记录
        metadata = []
        for i, rec in enumerate(records):
            metadata.append({
                "text": rec["text"],
                "chapter": rec["chapter"],                 # 兼容旧字段
                "chapter_id": rec["chapter_id"],
                "chapter_title": rec["chapter_title"],
                "chapter_hash": rec["chapter_hash"],
                "scene_id": rec["scene_id"],
                "scene_summary": rec["scene_summary"],
                "book_id": rec["book_id"],
                "book_title": rec["book_title"],
                "chunk_id": rec["chunk_id"],
                "chunk_index": rec["chunk_index"],
                "chunk_length": rec["chunk_length"],
                "token_count": rec["token_count"],
                "total_token_count": rec["total_token_count"],
                "start": rec["start"],
                "end": rec["end"],
                "prefix": rec.get("prefix", ""),
                # 指代注入前缀（仅命中指代的 chunk 有值；检索阶段附加到查询文本）
                "coref_prefix": rec["coref_prefix"],
                "embed_text": rec["embed_text"],
                # 分层召回字段（在既有字段基础上追加，不修改既有字段）
                "parent_id": rec["parent_id"],
                "prev_chunk_id": rec["prev_chunk_id"],
                "next_chunk_id": rec["next_chunk_id"],
                "reused": bool(reuse_flags[i]),
            })

        # 尝试加载嵌入模型（本地目录 或 HF 模型名，自动选 GPU/CPU）
        embedding_model, device = load_embedding_model(embedding_model_path)
        if embedding_model is None:
            if embedding_model_path:
                # 规格：显式指定了嵌入模型却加载不到 → 建库失败，
                # 绝不产出「无向量的假就绪索引」——否则项目状态会被标成
                # ready、检索端拿到半成品。
                print(f"✗ 嵌入模型不可用: {embedding_model_path}（建库失败，"
                      f"未产出任何索引文件）")
                print("  如需完全离线，请把 embedding_model_path 置空后重试"
                      "（纯关键词检索模式）")
                if progress_callback:
                    progress_callback(-1, 100, "失败: 嵌入模型不可用", 0)
                # 回收本次创建的空目录，避免残留幽灵目录污染项目状态
                if not _dir_preexisted:
                    try:
                        if os.path.isdir(output_dir) and not os.listdir(output_dir):
                            os.rmdir(output_dir)
                    except OSError:
                        pass
                return False
            print("⚠ 未配置嵌入模型（embedding_model_path 置空），使用纯关键词"
                  "检索模式（仅生成 metadata.json）")
        else:
            print(f"  设备: {device}")

        # 生成向量（仅编码未复用块；编码文本 = 书/章/场景前缀 + 指代增广正文）
        embeddings = None
        if embedding_model:
            dim = None
            try:
                dim = int(embedding_model.get_sentence_embedding_dimension())
            except Exception:
                dim = None
            if dim is not None:
                valid = [v for v in reused_vectors if v is not None]
                if valid and int(np.asarray(valid[0]).shape[-1]) != dim:
                    # 旧库向量维度与当前模型不符 → 放弃复用，全量重编码
                    print(f"⚠ 旧向量维度 {np.asarray(valid[0]).shape[-1]} 与模型 {dim} 不一致，改为全量重编码")
                    reuse_flags = [False] * len(records)
                    reused_vectors = [None] * len(records)
                    reused_count = 0
                    # 同步已生成的 metadata 复用标记，避免落盘信息与真实编码行为不一致
                    for i, m in enumerate(metadata):
                        m["reused"] = False

            total = len(records)
            pending = [i for i in range(total) if not reuse_flags[i]]
            # GPU 用更大 batch 充分利用显存，CPU 用保守值避免内存压力
            batch_size = 256 if device == "cuda" else 32

            if progress_callback:
                progress_callback(
                    30, 100,
                    f"正在生成向量（复用 {reused_count} / 新增 {len(pending)}）...",
                    30,
                )

            start_time = time.time()
            new_vectors: Dict[int, np.ndarray] = {}
            for i0 in range(0, len(pending), batch_size):
                batch_idx = pending[i0:i0 + batch_size]
                batch_chunks = [encode_texts[i] for i in batch_idx]

                # 批量编码
                batch_embeddings = embedding_model.encode(
                    batch_chunks,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                arr = np.asarray(batch_embeddings, dtype=np.float32)
                for k, gi in enumerate(batch_idx):
                    new_vectors[gi] = arr[k]

                # 进度更新
                done = min(i0 + batch_size, len(pending))
                progress = 30 + (done / max(len(pending), 1)) * 60
                if progress_callback:
                    progress_callback(
                        done,
                        len(pending),
                        f"已处理 {done}/{len(pending)} 片段（复用 {reused_count}）",
                        progress
                    )

                # 每100个打印一次
                if done % 100 == 0 or done == len(pending):
                    elapsed = time.time() - start_time
                    print(f"  进度: {done}/{len(pending)} 新增片段, 耗时 {elapsed:.1f}s")

            # 组装：复用块取旧向量，其余取新编码结果
            vectors = []
            for i in range(total):
                if reuse_flags[i] and reused_vectors[i] is not None:
                    vectors.append(np.asarray(reused_vectors[i], dtype=np.float32))
                else:
                    vectors.append(new_vectors[i])
            embeddings = np.stack(vectors).astype(np.float32) if vectors else None

        # 保存结果
        if progress_callback:
            progress_callback(95, 100, "保存向量索引...", 95)

        # 打印注入统计（信息性，不影响功能）
        injected_count = sum(1 for p in coref_texts if p)
        if injected_count:
            print(f"ℹ 指代消解前缀注入: {injected_count}/{len(records)} 个片段")

        # 保存嵌入向量
        if embeddings is not None:
            embeddings_path = os.path.join(output_dir, "embeddings.npy")
            np.save(embeddings_path, embeddings)
            print(f"✓ 向量已保存: {embeddings_path} ({embeddings.shape[0]} x {embeddings.shape[1]})")

        # 保存元数据
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=1)
        print(f"✓ 元数据已保存: {metadata_path} ({len(metadata)} 条)")

        # 保存父块表（与向量库同目录，检索端按 parent_id 反查父块）
        parent_path = os.path.join(output_dir, "parent_chunks.json")
        with open(parent_path, 'w', encoding='utf-8') as f:
            json.dump(parent_chunks, f, ensure_ascii=False, indent=1)
        print(f"✓ 父块已保存: {parent_path} ({len(parent_chunks)} 条)")

        # 保存章节索引（增量重切的对账依据：章 hash + 块数）
        chapters_out = []
        for c in result.get("chapters", []):
            chapters_out.append({
                "chapter_id": c["chapter_id"],
                "chapter_title": c["chapter_title"],
                "hash": c["hash"],
                "chunk_count": c["chunk_count"],
            })
        if not chapters_out:  # 兜底单块场景
            chapters_out = [{
                "chapter_id": records[0]["chapter_id"],
                "chapter_title": records[0]["chapter_title"],
                "hash": records[0]["chapter_hash"],
                "chunk_count": len(records),
            }]
        chapter_index = {
            "book_id": book_id,
            "book_title": book_title,
            "generated_at": format_timestamp(),
            "chapters": chapters_out,
        }
        chapter_index_path = os.path.join(output_dir, "chapter_index.json")
        with open(chapter_index_path, 'w', encoding='utf-8') as f:
            json.dump(chapter_index, f, ensure_ascii=False, indent=1)
        print(f"✓ 章节索引已保存: {chapter_index_path} ({len(chapters_out)} 章)")

        # 保存实际生效的切片规格（供检索端对齐口径 + 运维/测试对账）
        lengths = sorted(r["chunk_length"] for r in records)
        tokens = sorted(r["token_count"] for r in records)
        spec_record = dict(spec)
        spec_record.update({
            "chunk_count": chunk_count,
            "parent_count": len(parent_chunks),
            "reused_chunk_count": reused_count,
            "chapter_count": len(chapters_out),
            "chunk_length_min": lengths[0] if lengths else 0,
            "chunk_length_median": lengths[len(lengths) // 2] if lengths else 0,
            "chunk_length_max": lengths[-1] if lengths else 0,
            "token_min": tokens[0] if tokens else 0,
            "token_median": tokens[len(tokens) // 2] if tokens else 0,
            "token_max": tokens[-1] if tokens else 0,
        })
        spec_path = os.path.join(output_dir, "split_spec.json")
        with open(spec_path, 'w', encoding='utf-8') as f:
            json.dump(spec_record, f, ensure_ascii=False, indent=1)
        print(
            f"✓ 切片规格已保存: {spec_path} "
            f"(token 目标 {spec['target_tokens']} / 上限 {spec['max_tokens']} / 最小 {spec['min_tokens']}；"
            f"重叠 {spec['overlap_chars']}-{spec['overlap_max_chars']} 字句边界对齐)"
        )

        if progress_callback:
            progress_callback(100, 100, "向量化完成！", 100)

        return True

    except ImportError as e:
        print(f"✗ 缺少依赖库: {e}")
        if progress_callback:
            progress_callback(-1, 100, f"缺少依赖库: {e}")
        return False
    except Exception as e:
        print(f"✗ 构建向量索引失败: {e}")
        import traceback
        traceback.print_exc()
        if progress_callback:
            progress_callback(-1, 100, f"构建失败: {str(e)}")
        return False


def build_vector_index_from_file(
    input_file: str,
    chunk_size: int = MAX_CHUNK_CHARS,
    overlap: int = DEFAULT_OVERLAP_CHARS,
    output_dir: str = "./vector_db",
    embedding_model_path: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
    target_chars: Optional[int] = None,
    overlap_ratio: Optional[float] = None,
    book_id: str = "",
    book_title: str = ""
) -> bool:
    """
    从文件构建向量索引

    Args:
        input_file: 输入文本文件路径
        chunk_size: 切片硬上限（字符数，规格区间 [MIN,MAX]，缺省 672 ≈ 480 token）
        overlap: 相邻片段重叠长度下限（字符数，缺省 50）
        output_dir: 输出目录
        embedding_model_path: 嵌入模型路径
        progress_callback: 进度回调
        target_chars: 目标块长覆盖值（None → 560 ≈ 400 token）
        overlap_ratio: 重叠比例覆盖值（None → 0.125）
        book_id: 书籍标识（缺省从文件路径推断）
        book_title: 书名（缺省从文件名推断；写入 chunk 前缀与元数据）

    Returns:
        是否成功
    """
    try:
        # 检测编码并读取
        encoding = 'utf-8'
        try:
            import chardet
            with open(input_file, 'rb') as f:
                raw = f.read(10000)
                result = chardet.detect(raw)
                encoding = result['encoding'] or 'utf-8'
        except ImportError:
            pass

        with open(input_file, 'r', encoding=encoding, errors='ignore') as f:
            text = f.read()

        if progress_callback:
            progress_callback(0, 100, f"读取文件: {len(text)} 字符", 0)

        # 书名/书 id 缺省从文件路径推断（写入 chunk 前缀与元数据，便于溯源）
        if not book_title:
            book_title = os.path.splitext(os.path.basename(input_file))[0]
        if not book_id:
            book_id = os.path.basename(os.path.dirname(os.path.abspath(input_file))) or book_title

        return build_vector_index(
            text=text,
            chunk_size=chunk_size,
            overlap=overlap,
            output_dir=output_dir,
            embedding_model_path=embedding_model_path,
            progress_callback=progress_callback,
            target_chars=target_chars,
            overlap_ratio=overlap_ratio,
            book_id=book_id,
            book_title=book_title,
        )

    except Exception as e:
        print(f"✗ 从文件构建向量索引失败: {e}")
        if progress_callback:
            progress_callback(-1, 100, f"失败: {str(e)}")
        return False


def get_index_stats(vector_db_path: str) -> Optional[Dict]:
    """
    获取向量索引统计信息

    Args:
        vector_db_path: 向量库路径

    Returns:
        统计信息字典或None
    """
    embeddings_path = os.path.join(vector_db_path, "embeddings.npy")
    metadata_path = os.path.join(vector_db_path, "metadata.json")

    stats = {
        "exists": os.path.exists(vector_db_path),
        "has_embeddings": os.path.exists(embeddings_path),
        "has_metadata": os.path.exists(metadata_path),
        "chunk_count": 0,
        "embedding_dim": 0
    }

    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            stats["chunk_count"] = len(metadata)
        except Exception:
            pass

    if os.path.exists(embeddings_path):
        try:
            embeddings = np.load(embeddings_path)
            if embeddings.ndim == 2:
                stats["chunk_count"] = embeddings.shape[0]
                stats["embedding_dim"] = embeddings.shape[1]
        except Exception:
            pass

    return stats if stats["has_embeddings"] and stats["has_metadata"] else None


# ===================== 命令行入口（保留） =====================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="切片与向量化工具")
    parser.add_argument("--input", required=True, help="输入文本文件")
    parser.add_argument("--output", default="./vector_db", help="输出目录")
    parser.add_argument("--chunk-size", type=int, default=MAX_CHUNK_CHARS,
                        help=f"切片硬上限（字符数，默认 {MAX_CHUNK_CHARS}）")
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP_CHARS,
                        help=f"相邻片段重叠字符数（默认 {DEFAULT_OVERLAP_CHARS}）")
    parser.add_argument("--model", default=None, help="嵌入模型路径")
    args = parser.parse_args()

    def print_progress(step, total, msg, percent=0):
        print(f"[{percent:3.0f}%] {msg}")

    success = build_vector_index_from_file(
        input_file=args.input,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        output_dir=args.output,
        embedding_model_path=args.model,
        progress_callback=print_progress
    )

    if success:
        print(f"✓ 向量化完成: {args.output}")
    else:
        print("✗ 向量化失败")
