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

from utils import split_text_by_sentences, extract_chapter_title


def build_vector_index(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    output_dir: str = "./vector_db",
    embedding_model_path: Optional[str] = None,
    progress_callback: Optional[Callable] = None
) -> bool:
    """
    构建向量索引

    Args:
        text: 输入文本
        chunk_size: 切片大小（字符数）
        overlap: 重叠长度（字符数）
        output_dir: 输出目录
        embedding_model_path: 嵌入模型路径（可选，为空使用关键词检索）
        progress_callback: 进度回调 (step: int, total: int, message: str, percent: float)

    Returns:
        是否成功
    """
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 调整参数以适配 split_text_by_sentences
        # chunk_size 作为 max_chars，overlap 转换为句子数（约1-2句）
        max_chars = chunk_size
        min_chars = max(100, chunk_size // 3)
        overlap_sentences = max(1, overlap // 50) if overlap > 0 else 1

        # 文本切片
        if progress_callback:
            progress_callback(0, 100, "正在切片...", 0)

        chunks = split_text_by_sentences(text, min_chars, max_chars, overlap_sentences)

        if not chunks:
            chunks = [text]

        chunk_count = len(chunks)
        if progress_callback:
            progress_callback(20, 100, f"切片完成: {chunk_count} 个片段", 20)

        # 准备元数据
        metadata = []
        for i, chunk in enumerate(chunks):
            metadata.append({
                "text": chunk,
                "chapter": f"第 {i+1} 段",
                "chunk_id": i + 1,
                "chunk_length": len(chunk)
            })

        # 尝试加载嵌入模型
        embedding_model = None
        if embedding_model_path and os.path.exists(embedding_model_path):
            try:
                from sentence_transformers import SentenceTransformer
                if progress_callback:
                    progress_callback(25, 100, "正在加载嵌入模型...", 25)
                embedding_model = SentenceTransformer(embedding_model_path, local_files_only=True)
                embedding_model.max_seq_length = 512
                print(f"✓ 成功加载嵌入模型: {embedding_model_path}")
            except ImportError:
                print("⚠ 未安装 sentence_transformers，将使用关键词检索模式")
            except Exception as e:
                print(f"⚠ 加载嵌入模型失败: {e}，将使用关键词检索模式")
        else:
            if embedding_model_path:
                print(f"⚠ 嵌入模型路径不存在: {embedding_model_path}")
            print("⚠ 未配置嵌入模型，将使用关键词检索模式")

        # 生成向量
        embeddings = None
        if embedding_model:
            embeddings = []
            total = len(chunks)
            batch_size = 32

            if progress_callback:
                progress_callback(30, 100, "正在生成向量...", 30)

            start_time = time.time()
            for i in range(0, total, batch_size):
                batch_end = min(i + batch_size, total)
                batch_chunks = chunks[i:batch_end]

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
    chunk_size: int = 500,
    overlap: int = 50,
    output_dir: str = "./vector_db",
    embedding_model_path: Optional[str] = None,
    progress_callback: Optional[Callable] = None
) -> bool:
    """
    从文件构建向量索引

    Args:
        input_file: 输入文本文件路径
        chunk_size: 切片大小
        overlap: 重叠长度
        output_dir: 输出目录
        embedding_model_path: 嵌入模型路径
        progress_callback: 进度回调

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
            progress_callback=progress_callback
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
    parser.add_argument("--chunk-size", type=int, default=500, help="切片大小")
    parser.add_argument("--overlap", type=int, default=50, help="重叠长度")
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
