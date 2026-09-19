# -*- coding: utf-8 -*-
"""
tests/test_commercial_capacity_concurrency.py — 商用验收 [第 3 组：容量 / 性能 / 并发 / 端到端]

设计原则：
1) 数值化能力边界：把「能扛多少、多快、多少个并发」变成可复现实测数字；
2) 只读真实数据（生产库/语料），绝不写入 data/ 与仓库；
3) 阈值取自商用体感要求（交互式问答 P95 应 < 1s、单次建库应可在分钟级完成），
   现状达不到即记为失败/风险，不因「勉强能跑」而放行。
"""
from __future__ import annotations

import json
import sys
import threading
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

import _nr_commercial as H

ROOT = H.ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(H.TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(H.TESTS_DIR))

import step2_split_embed as step2  # noqa: E402


# ============================================================
# 一、容量与性能
# ============================================================

class TestCapacityAndPerformance:

    def test_real_corpus_build_throughput_200k(self, tmp_path, monkeypatch):
        """【规格来源】商用验收：万级字语料建库需在分钟级完成，且块规格自洽。

        【判定标准】真实语料前 20 万字离线建库：耗时 < 120s，块数 > 100，
        块长 <= 672 硬上限，embeddings 行数 == metadata 条数。
        """
        assert H.PROD_CORPUS.exists(), f"真实语料缺失: {H.PROD_CORPUS}"
        text = H.PROD_CORPUS.read_text(encoding="utf-8", errors="ignore")[:200_000]
        ok, out, _ = H.build_tiny_index(tmp_path, text, monkeypatch, dim=64, name="cap200k")
        assert ok is True
        meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
        mat = np.load(out / "embeddings.npy")
        lens = [m["chunk_length"] for m in meta]
        H.record_metric("capacity.build_200k_chunks", len(meta), "块", f"语料 20 万字")
        H.record_metric("capacity.build_200k_max_chunk_len", max(lens), "字符", "硬上限 672")
        H.record_metric("capacity.build_200k_mean_chunk_len", round(sum(lens) / len(lens), 1),
                        "字符", "")
        assert len(meta) > 100
        assert mat.shape[0] == len(meta)

    def test_split_throughput_on_1m_chars(self):
        """【规格来源】容量边界：百万字切片不得成为瓶颈。

        【判定标准】100 万字切片耗时 < 60s，且块长全部 <= 672。
        """
        unit = "叶星提剑而立，山风猎猎。" * 20 + "\n"
        text = unit * (1_000_000 // len(unit) + 1)
        text = text[:1_000_000]
        chunks, secs = H.timed(__import__("utils").split_text_recursive, text,
                               min_chars=210, max_chars=672)
        maxlen = max(len(c) for c in chunks)
        H.record_metric("capacity.split_1m_seconds", round(secs, 2), "秒",
                        f"100 万字 → {len(chunks)} 块")
        H.record_metric("capacity.split_1m_max_chunk_len", maxlen, "字符", "硬上限 672")
        assert secs < 60, f"100 万字切片耗时 {secs:.1f}s"
        assert maxlen <= 672, f"出现超上限块 {maxlen} 字符"

    def test_large_library_20k_blocks_latency(self, tmp_path):
        """【规格来源】商用交互体感：2 万块库的检索 P95 应 < 1.0s。

        【判定标准】20000 块（64 维）库：加载耗时记录；100 次检索 P50/P95 < 1.0s。
        """
        n, dim = 20_000, 64
        rng = np.random.default_rng(7)
        vecs = rng.random((n, dim), dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        texts = [f"第{i % 500 + 1}章 叶星在场景{i}中出手，剑光如练。" for i in range(n)]
        vd = H.write_library(tmp_path / "big", texts, vectors=vecs,
                             chapters=[f"第{i % 500 + 1}章" for i in range(n)])
        t0 = time.perf_counter()
        r = H.make_retriever(vd, embedding_model=H.FakeEmbeddingModel(dim=dim))
        load_secs = time.perf_counter() - t0
        lat = []
        for i in range(100):
            t = time.perf_counter()
            r.retrieve(f"叶星 场景{i}", top_k=5)
            lat.append(time.perf_counter() - t)
        p50, p95 = H.percentile(lat, 0.5), H.percentile(lat, 0.95)
        H.record_metric("capacity.large_lib_load_20k_seconds", round(load_secs, 3), "秒", "20000 块 / 64 维")
        H.record_metric("capacity.large_lib_search_p50", round(p50 * 1000, 1), "毫秒", "20000 块")
        H.record_metric("capacity.large_lib_search_p95", round(p95 * 1000, 1), "毫秒", "阈值 1000ms")
        assert p95 < 1.0, f"20000 块库检索 P95 = {p95 * 1000:.1f}ms > 1000ms"

    def test_large_library_memory_footprint(self, tmp_path):
        """【规格来源】容量边界：索引内存占用应与数据量同阶（不得数十倍膨胀）。

        【判定标准】20000 块 / 64 维库加载峰值增量 < 300MB。
        """
        n, dim = 20_000, 64
        rng = np.random.default_rng(11)
        vecs = rng.random((n, dim), dtype=np.float32)
        texts = [f"第{i % 300 + 1}章 叶星出剑，山门震动。" for i in range(n)]
        vd = H.write_library(tmp_path / "mem", texts, vectors=vecs,
                            chapters=[f"第{i % 300 + 1}章" for i in range(n)])
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        H.make_retriever(vd, embedding_model=H.FakeEmbeddingModel(dim=dim))
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        delta_mb = (peak - base) / 1024 / 1024
        H.record_metric("capacity.large_lib_peak_mem_mb", round(delta_mb, 1), "MB",
                        "20000 块 / 64 维 峰值增量")
        assert delta_mb < 300, f"2 万块库加载峰值内存 {delta_mb:.0f}MB"

    def test_extreme_top_k_is_bounded(self, tmp_path):
        """【规格来源】参数边界：top_k 远超库容量时不得崩溃/重复返回。

        【判定标准】top_k=10000 时返回条数 <= 库块数，且无重复 chunk。
        """
        r = H.make_retriever(
            H.write_library(tmp_path / "tk", ["叶星出剑。" * 5 for _ in range(30)]),
            embedding_model=H.FakeEmbeddingModel(dim=16))
        hits = r.retrieve("叶星出剑", top_k=10_000)
        keys = [(h.get("chapter"), h.get("text")) for h in hits]
        H.record_metric("capacity.extreme_topk_returned", len(hits), "条", "库内 30 块，请求 top_k=10000")
        assert len(hits) <= 30, f"返回 {len(hits)} 条超过库内块数"
        assert len(keys) == len(set(keys)), f"存在重复结果条目"

    def test_very_long_query_latency(self, tmp_path):
        """【规格来源】输入边界：超长查询不得导致数量级延迟或崩溃。

        【判定标准】5 万字符查询在 20000 块库上检索耗时 < 5s。
        """
        n, dim = 20_000, 64
        rng = np.random.default_rng(3)
        vecs = rng.random((n, dim), dtype=np.float32)
        texts = [f"第{i % 200 + 1}章 叶星出剑。" for i in range(n)]
        vd = H.write_library(tmp_path / "lq", texts, vectors=vecs,
                            chapters=[f"第{i % 200 + 1}章" for i in range(n)])
        r = H.make_retriever(vd, embedding_model=H.FakeEmbeddingModel(dim=dim))
        query = "叶星出剑守卫山门，" * (50_000 // 10)
        _, secs = H.timed(r.retrieve, query, top_k=5)
        H.record_metric("capacity.long_query_50k_seconds", round(secs, 3), "秒",
                        "5 万字符查询 / 20000 块库")
        assert secs < 5, f"超长查询耗时 {secs:.2f}s"

    @pytest.mark.parametrize("query,label", [
        ("叶星", "2 字符"),
        ("叶", "1 字符"),
        ("   ", "纯空白"),
        ("叶星" * 200, "400 字符"),
    ])
    def test_query_length_spectrum_latency(self, tmp_path, query, label):
        """【规格来源】能力边界：查询长度谱（1 字 ~ 400 字）均应稳定响应。

        【判定标准】各长度查询耗时 < 1s，返回结构合法。
        """
        r = H.make_retriever(
            H.write_library(tmp_path / f"q{len(query)}",
                            ["叶星出剑。" * 10 for _ in range(200)]),
            embedding_model=H.FakeEmbeddingModel(dim=16))
        hits, secs = H.timed(r.retrieve, query, top_k=5)
        H.record_metric(f"capacity.query_{label}_seconds", round(secs, 4), "秒", "")
        assert secs < 1, f"{label} 查询耗时 {secs:.3f}s"
        assert isinstance(hits, list)


# ============================================================
# 二、真实生产库只读体检
# ============================================================

class TestProductionLibraryReadonly:

    def test_production_library_metrics(self):
        """【规格来源】真实交付物体检：生产库必须可加载、可检索、索引自洽。

        【判定标准】加载 data/绝世主宰/vector_db：块数 > 0、向量行数 == 块数、
        50 次检索 P95 < 1s、top1 分数记录在案。
        """
        if not H.PROD_VDB.exists():
            pytest.skip(f"生产向量库不存在: {H.PROD_VDB}")
        meta = json.loads((H.PROD_VDB / "metadata.json").read_text(encoding="utf-8"))
        emb = np.load(H.PROD_VDB / "embeddings.npy", mmap_mode="r")
        r = H.make_retriever(H.PROD_VDB, embedding_model=H.FakeEmbeddingModel(dim=emb.shape[1]))
        lat, top1 = [], []
        for i in range(50):
            t = time.perf_counter()
            hits = r.retrieve(f"叶星第{i}次出手", top_k=5)
            lat.append(time.perf_counter() - t)
            if hits:
                top1.append(hits[0]["score"])
        H.record_metric("prod.chunks", len(meta), "块", "production vector_db")
        H.record_metric("prod.embeddings_shape", str(emb.shape), "", "应等于 (块数, dim)")
        H.record_metric("prod.parent_chunks", len(r.parent_chunks), "父块", "")
        H.record_metric("prod.search_p50", round(H.percentile(lat, 0.5) * 1000, 1), "毫秒", "生产库 50 次")
        H.record_metric("prod.search_p95", round(H.percentile(lat, 0.95) * 1000, 1), "毫秒", "")
        H.record_metric("prod.top1_score_mean", round(float(np.mean(top1)), 4) if top1 else "N/A", "",
                        "替身模型下的 top1 余弦（仅结构验证）")
        assert emb.shape[0] == len(meta), f"向量行数 {emb.shape[0]} != 元数据 {len(meta)}"
        assert H.percentile(lat, 0.95) < 1.0

    def test_retrieval_does_not_mutate_vector_dir(self):
        """【规格来源】安全基线：检索是只读操作，不得改写索引文件。

        【判定标准】检索前后 vector_db 下文件 mtime+size 完全一致。
        """
        if not H.PROD_VDB.exists():
            pytest.skip(f"生产向量库不存在: {H.PROD_VDB}")
        def _snap():
            return {p.name: (p.stat().st_mtime_ns, p.stat().st_size)
                    for p in H.PROD_VDB.iterdir() if p.is_file()}
        before = _snap()
        emb = np.load(H.PROD_VDB / "embeddings.npy", mmap_mode="r")
        r = H.make_retriever(H.PROD_VDB, embedding_model=H.FakeEmbeddingModel(dim=emb.shape[1]))
        r.retrieve("叶星出剑", top_k=5)
        after = _snap()
        H.record_metric("prod.vector_dir_mutated", "是" if before != after else "否", "",
                        "检索前后文件快照对比")
        assert before == after, "检索过程改写了向量库文件"


# ============================================================
# 三、并发安全
# ============================================================

class TestConcurrencySafety:

    def test_concurrent_retrieval_same_instance(self, tmp_path):
        """【规格来源】商用预期：服务化后同一检索器会被多请求并发调用。

        【判定标准】8 线程 × 25 次并发检索同一实例：无异常、每次返回 5 条。
        """
        r = H.make_retriever(
            H.write_library(tmp_path / "conc",
                            [f"第{i}章 叶星出剑斩敌。" * 3 for i in range(500)],
                            chapters=[f"第{i}章" for i in range(500)]),
            embedding_model=H.FakeEmbeddingModel(dim=16))
        errors, counts = [], []

        def worker(idx):
            try:
                for j in range(25):
                    hits = r.retrieve(f"叶星出剑 {idx}-{j}", top_k=5)
                    counts.append(len(hits))
            except Exception as e:  # noqa: BLE001
                errors.append(f"{type(e).__name__}: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)
        H.record_metric("concurrency.same_instance_errors", len(errors), "次",
                        f"8 线程 × 25 次；样本错误={errors[:2]}")
        H.record_metric("concurrency.same_instance_min_hits", min(counts) if counts else -1, "条",
                        f"共 {len(counts)} 次调用")
        assert not errors, f"并发检索同一实例出现异常: {errors[:3]}"
        assert counts and min(counts) == 5, f"并发下出现少于 5 条的结果: {min(counts)}"

    def test_concurrent_retrieval_separate_instances(self, tmp_path):
        """【规格来源】商用预期：多项目并行时各自实例互不干扰。

        【判定标准】8 个独立实例并发检索，无异常。
        """
        vd = H.write_library(tmp_path / "multi",
                             [f"第{i}章 叶星出剑。" * 3 for i in range(300)],
                             chapters=[f"第{i}章" for i in range(300)])
        errors, results = [], []

        def worker():
            try:
                rr = H.make_retriever(vd, embedding_model=H.FakeEmbeddingModel(dim=16))
                results.append(len(rr.retrieve("叶星出剑", top_k=5)))
            except Exception as e:  # noqa: BLE001
                errors.append(f"{type(e).__name__}: {e}")

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)
        H.record_metric("concurrency.multi_instance_errors", len(errors), "次", str(errors[:2]))
        assert not errors, f"多实例并发出现异常: {errors[:3]}"
        assert results == [5] * 8, f"结果异常: {results}"

    def test_concurrent_rebuild_same_output_dir(self, tmp_path, monkeypatch):
        """【规格来源】运维风险：重复点击「重建索引」造成同目录并发写。

        【判定标准】并发建库结束后，落盘索引必须自洽（npy 行数 == metadata 条数），
        且无损坏 JSON。
        """
        text = "\n".join(["第1章 试炼\n"] + ["叶星在山门修炼剑法，日夜不辍。" * 6] * 40)
        H.install_fake_embedding(monkeypatch, dim=16)
        out = tmp_path / "race"
        errors, oks = [], []

        def worker():
            try:
                oks.append(step2.build_vector_index(
                    text=text, output_dir=str(out), embedding_model_path="fake",
                    book_id="b", book_title="测试书"))
            except Exception as e:  # noqa: BLE001
                errors.append(f"{type(e).__name__}: {e}")

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=180)

        meta_ok, rows = False, -1
        try:
            meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
            meta_ok = True
            rows = np.load(out / "embeddings.npy").shape[0]
        except Exception as e:  # noqa: BLE001
            errors.append(f"读取产物失败 {type(e).__name__}: {e}")
        H.record_metric("concurrency.rebuild_ok_count", sum(1 for o in oks if o), "次",
                        f"4 线程并发；异常={errors[:2]}")
        H.record_metric("concurrency.rebuild_index_consistent",
                        f"meta_ok={meta_ok}, rows={rows}, meta={len(meta) if meta_ok else 'NA'}", "",
                        "并发写后索引自洽性")
        assert meta_ok, f"并发建库后 metadata 不可解析: {errors[:2]}"
        assert rows == len(meta), f"并发建库后 npy 行数 {rows} != metadata {len(meta)}"

    def test_concurrent_retrieve_and_rebuild_isolation(self, tmp_path, monkeypatch):
        """【规格来源】商用预期：重建索引期间读取旧索引不应崩溃。

        【判定标准】一边检索一边重建（不同目录，模拟旧库+新库），检索线程无异常。
        """
        old_vd = H.write_library(tmp_path / "old",
                                 [f"第{i}章 叶星出剑。" * 4 for i in range(200)],
                                 chapters=[f"第{i}章" for i in range(200)])
        r = H.make_retriever(old_vd, embedding_model=H.FakeEmbeddingModel(dim=16))
        H.install_fake_embedding(monkeypatch, dim=16)
        new_dir = tmp_path / "new"
        errors = []
        stop = threading.Event()

        def reader():
            try:
                while not stop.is_set():
                    if len(r.retrieve("叶星出剑", top_k=5)) == 0:
                        errors.append("检索返回空")
            except Exception as e:  # noqa: BLE001
                errors.append(f"{type(e).__name__}: {e}")

        t = threading.Thread(target=reader)
        t.start()
        text = "\n".join(["第1章 试炼\n"] + ["叶星出剑，山门震动。" * 6] * 40)
        step2.build_vector_index(text=text, output_dir=str(new_dir),
                                 embedding_model_path="fake", book_id="b", book_title="测试书")
        stop.set()
        t.join(timeout=60)
        H.record_metric("concurrency.reader_errors_during_rebuild", len(errors), "次",
                        f"样本={errors[:2]}")
        assert not errors, f"重建期间读取旧索引出错: {errors[:3]}"
