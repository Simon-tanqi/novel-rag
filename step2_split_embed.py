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
    split_text_by_chapters,
    resolve_split_params,
    resolve_split_spec,
    MIN_CHUNK_CHARS,
    MAX_CHUNK_CHARS,
    DEFAULT_OVERLAP_CHARS,
    extract_chapter_title,
    resolve_embedding_model,
    load_embedding_model,
    build_parent_chunks,
)
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
    overlap_ratio: Optional[float] = None
) -> bool:
    """
    构建向量索引

    切片规格（由 utils.resolve_split_spec 唯一发放）：
    - 最优先：距上次切分点不足 MIN_CHUNK_CHARS(200) 字符 → 不切分，
      即使遇到句末标点也继续累积；
    - 可切区间：达到 200 字符后、512 字符前遇到 。！：？ → 在其后切分，
      并在窗口内优先选最接近 target_chars(400) 的标点；
    - 兜底：距上次切分点超过 MAX_CHUNK_CHARS(512) 仍无一级标点 →
      二级（；：，、）→ 三级（空白）→ 硬切；
    - 相邻重叠 = max(overlap, 前块长度 × overlap_ratio)。

    Args:
        text: 输入文本
        chunk_size: 切片硬上限（字符数，规格区间 [200, 512]，缺省 512）
        overlap: 相邻片段重叠长度下限（字符数，缺省 50）
        output_dir: 输出目录
        embedding_model_path: 嵌入模型路径（可选，为空使用关键词检索）
        progress_callback: 进度回调 (step: int, total: int, message: str, percent: float)
        target_chars: 目标块长覆盖值（None → 400，裁剪进 [min, max]）
        overlap_ratio: 重叠比例覆盖值（None → 0.15，裁剪进 [0, 0.5]）

    Returns:
        是否成功
    """
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 规格切片参数：min/max/重叠/目标长度/重叠比例由 utils 唯一出口发放。
        # 实际生效的 spec 会落盘 split_spec.json，供检索端与运维对账
        # （历史上 TARGET_CHUNK_CHARS/OVERLAP_RATIO 曾因未接线而静默失效）。
        spec = resolve_split_spec(chunk_size, overlap, target_chars, overlap_ratio)
        min_chars, max_chars = spec["min_chars"], spec["max_chars"]

        # 文本切片（按章节边界切分，chunk 携带真实章节标题）
        if progress_callback:
            progress_callback(0, 100, "正在切片...", 0)

        chunks, chapter_of_chunk = split_text_by_chapters(
            text, min_chars, max_chars, spec["overlap_chars"],
            target_chars=spec["target_chars"], overlap_ratio=spec["overlap_ratio"],
        )

        if not chunks:
            chunks = [text]
            chapter_of_chunk = ["正文"]
        if len(chapter_of_chunk) < len(chunks):  # 防御：章节列表与子块对齐
            chapter_of_chunk = list(chapter_of_chunk) + ["正文"] * (len(chunks) - len(chapter_of_chunk))

        # ---- 指代消解增强：为疑似指代开头的 chunk 注入主语前缀 ----
        # 注入后的增广文本仅用于向量化与关键词检索，metadata 存回原始文本，
        # 保证展示给 LLM/用户的仍是原文；前缀使含「他/她」的片段可被主角名召回。
        # 主角表从本书原文自动挖掘（known=[]），避免跨书默认表注入错误主语。
        characters = _pick_characters(chunks)
        if progress_callback:
            progress_callback(10, 100, f"主角表(自动挖掘): {'、'.join(characters[:8])}", 10)
        augmented = inject_coref_prefix(chunks, chapter_of_chunk, characters=characters)
        chunks_with_prefix = [
            aug if aug != orig else ""  # 仅标记真正注入过的
            for aug, orig in zip(augmented, chunks)
        ]

        chunk_count = len(chunks)
        if progress_callback:
            progress_callback(20, 100, f"切片完成: {chunk_count} 个片段", 20)

        # ---- 分层切片：子块检索 + 父块召回 + 邻居扩展 ----
        # 父块由同章节内连续的 2-4 个子块聚合（不跨章节），替代新增函数，
        # 既有切片逻辑与 chunk_id 生成规则完全不变。
        parent_chunks = build_parent_chunks(chunks, chapter_of_chunk)
        parent_of_chunk: Dict[int, str] = {}  # 子块下标(0-based) -> parent_id
        for parent in parent_chunks:
            for child_idx in parent.get("child_indices", []):
                parent_of_chunk[child_idx] = parent["parent_id"]

        child_chunk_ids = [
            [idx + 1 for idx in parent.get("child_indices", [])]
            for parent in parent_chunks
        ]
        for parent, ids in zip(parent_chunks, child_chunk_ids):
            parent["child_chunk_ids"] = ids  # 供检索端按 chunk_id 反查父块

        # 准备元数据（chapter 为真实章节标题，供检索结果溯源展示）
        metadata = []
        for i, chunk in enumerate(chunks):
            chapter = chapter_of_chunk[i] if i < len(chapter_of_chunk) else "正文"
            # 邻居链仅在同章节内建立（章节是硬边界，跨章邻居无上下文意义）
            prev_chunk_id = (
                i if (i > 0 and chapter_of_chunk[i - 1] == chapter) else None
            )
            next_chunk_id = (
                i + 2 if (i + 1 < len(chunks) and chapter_of_chunk[i + 1] == chapter) else None
            )
            metadata.append({
                "text": chunk,
                "chapter": chapter,
                "chunk_id": i + 1,
                "chunk_length": len(chunk),
                # 指代注入前缀（仅命中指代的 chunk 有值；检索阶段附加到查询文本）
                "coref_prefix": chunks_with_prefix[i] if i < len(chunks_with_prefix) else "",
                # 分层召回字段（新增，不删旧字段）
                "parent_id": parent_of_chunk.get(i),
                "prev_chunk_id": prev_chunk_id,
                "next_chunk_id": next_chunk_id,
            })

        # 尝试加载嵌入模型（本地目录 或 HF 模型名，自动选 GPU/CPU）
        embedding_model, device = load_embedding_model(embedding_model_path)
        if embedding_model is None:
            if embedding_model_path:
                print("⚠ 嵌入模型不可用，将使用关键词检索模式（仅生成 metadata.json）")
            else:
                print("⚠ 未配置嵌入模型，将使用关键词检索模式（仅生成 metadata.json）")
        else:
            print(f"  设备: {device}")

        # 生成向量（编码增广文本：带前缀的用前缀+正文，否则原文）
        embeddings = None
        if embedding_model:
            embeddings = []
            total = len(chunks)
            # GPU 用更大 batch 充分利用显存，CPU 用保守值避免内存压力
            batch_size = 256 if device == "cuda" else 32

            if progress_callback:
                progress_callback(30, 100, "正在生成向量...", 30)

            start_time = time.time()
            encode_texts = [
                (prefix + chunk) if prefix else chunk
                for prefix, chunk in zip(chunks_with_prefix, chunks)
            ]
            for i in range(0, total, batch_size):
                batch_end = min(i + batch_size, total)
                batch_chunks = encode_texts[i:batch_end]

                # 批量编码
                batch_embeddings = embedding_model.encode(
                    batch_chunks,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                embeddings.extend(batch_embeddings.tolist() if hasattr(batch_embeddings, 'tolist') else batch_embeddings)

                # 进度更新
                progress = 30 + (batch_end / total) * 60
                if progress_callback:
                    progress_callback(
                        batch_end,
                        total,
                        f"已处理 {batch_end}/{total} 片段",
                        progress
                    )

                # 每100个打印一次
                if (batch_end) % 100 == 0 or batch_end == total:
                    elapsed = time.time() - start_time
                    print(f"  进度: {batch_end}/{total} 片段, 耗时 {elapsed:.1f}s")

            # 转换为 numpy 数组
            embeddings = np.array(embeddings, dtype=np.float32)

        # 保存结果
        if progress_callback:
            progress_callback(95, 100, "保存向量索引...", 95)

        # 打印注入统计（信息性，不影响功能）
        injected_count = sum(1 for p in chunks_with_prefix if p)
        if injected_count:
            print(f"ℹ 指代消解前缀注入: {injected_count}/{len(chunks)} 个片段")

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

        # 保存实际生效的切片规格（供检索端对齐口径 + 运维/测试对账）
        lengths = sorted(len(c) for c in chunks)
        spec_record = dict(spec)
        spec_record.update({
            "chunk_count": chunk_count,
            "parent_count": len(parent_chunks),
            "chunk_length_min": lengths[0] if lengths else 0,
            "chunk_length_median": lengths[len(lengths) // 2] if lengths else 0,
            "chunk_length_max": lengths[-1] if lengths else 0,
        })
        spec_path = os.path.join(output_dir, "split_spec.json")
        with open(spec_path, 'w', encoding='utf-8') as f:
            json.dump(spec_record, f, ensure_ascii=False, indent=1)
        print(
            f"✓ 切片规格已保存: {spec_path} "
            f"(min={spec['min_chars']} target={spec['target_chars']} max={spec['max_chars']} "
            f"overlap={spec['overlap_chars']}/{spec['overlap_ratio']})"
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
    overlap_ratio: Optional[float] = None
) -> bool:
    """
    从文件构建向量索引

    Args:
        input_file: 输入文本文件路径
        chunk_size: 切片硬上限（字符数，规格区间 [200, 512]，缺省 512）
        overlap: 相邻片段重叠长度下限（字符数，缺省 50）
        output_dir: 输出目录
        embedding_model_path: 嵌入模型路径
        progress_callback: 进度回调
        target_chars: 目标块长覆盖值（None → 400）
        overlap_ratio: 重叠比例覆盖值（None → 0.15）

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

        return build_vector_index(
            text=text,
            chunk_size=chunk_size,
            overlap=overlap,
            output_dir=output_dir,
            embedding_model_path=embedding_model_path,
            progress_callback=progress_callback,
            target_chars=target_chars,
            overlap_ratio=overlap_ratio
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
