# -*- coding: utf-8 -*-
"""切片规格接线回归测试（防止"定义但零调用"再次发生）

覆盖三条规格：
1) min_chars=200 为最优先硬约束：不足 200 字符不切分，即使命中一级标点；
2) target_chars / overlap_ratio 真正接线：目标择优选标点、动态重叠生效，
   且 build_vector_index 把「实际生效规格」落盘 split_spec.json 供对账；
3) 检索端 assemble_context 按文本包含关系去重：父块覆盖子块时不重复输出。

历史上 TARGET_CHUNK_CHARS / OVERLAP_RATIO 曾在 utils 中定义但零调用
（实际跑 512 + 固定 50），本文件的作用是把该规格钉在测试上。
"""
import json

from utils import (
    split_text_recursive,
    resolve_split_spec,
    MIN_CHUNK_CHARS,
    MAX_CHUNK_CHARS,
)


# ===================== 规则 0：min_chars 最优先 =====================

class TestMinCharsHighestPriority:
    def test_punct_before_200_does_not_cut(self):
        """200 字符之前出现的一级标点一律不作为切分点"""
        text = "啊" * 30 + "。" * 5 + "呀" * 100  # 135 字符，含 5 个句号
        assert split_text_recursive(text, 200, 512, 50) == [text]

    def test_first_cut_never_before_200(self):
        text = "啊" * 199 + "。" + "呀" * 400  # 首个句号落在 200 字符之内
        chunks = split_text_recursive(text, 200, 512, 50)
        assert len(chunks[0]) >= MIN_CHUNK_CHARS
        assert not chunks[0].endswith("。")

    def test_only_punct_inside_200_512_window_cuts(self):
        """只有落在 [200, 512] 窗口内的一级标点才触发切分"""
        text = "啊" * 100 + "。" + "呀" * 99 + "？" + "嘿" * 300
        chunks = split_text_recursive(text, 200, 512, 50)
        assert chunks[0] == text[:201]
        assert chunks[0].endswith("？")
        assert len(chunks[0]) >= MIN_CHUNK_CHARS

    def test_target_mode_still_respects_min_and_max(self):
        """开启 target 择优后，仍不得在 200 之前切、不得越过 512 上限"""
        text = "啊" * 100 + "。" + "呀" * 500  # 窗口内无一级标点
        chunks = split_text_recursive(
            text, 200, 512, 50, target_chars=400, overlap_ratio=0.15
        )
        assert len(chunks[0]) == MAX_CHUNK_CHARS  # 硬切兜底，而非回退到句号处
        assert all(len(c) <= MAX_CHUNK_CHARS for c in chunks)


# ===================== 规则 2：target_chars / overlap_ratio 接线 =====================

class TestTargetAndRatioWiring:
    def test_target_chars_picks_nearest_punct(self):
        """同一文本，不同 target 选中不同标点 → 证明 target 真正参与切分"""
        text = "啊" * 250 + "。" + "呀" * 90 + "？" + "嘿" * 400
        near250 = split_text_recursive(text, 200, 512, 50, target_chars=250)
        near400 = split_text_recursive(text, 200, 512, 50, target_chars=400)
        assert near250[0] == text[:251] and near250[0].endswith("。")
        assert near400[0] == text[:342] and near400[0].endswith("？")
        assert len(near250[0]) != len(near400[0])

    def test_overlap_ratio_changes_dynamic_overlap(self):
        """重叠 = max(overlap_chars, 前块长度 × ratio)，ratio 越大下一块越靠前"""
        text = ("甲" * 250 + "。") * 6
        fixed = split_text_recursive(
            text, 200, 512, 50, target_chars=250, overlap_ratio=0.0
        )
        dynamic = split_text_recursive(
            text, 200, 512, 50, target_chars=250, overlap_ratio=0.3
        )
        assert fixed[1].startswith(fixed[0][-50:])
        assert dynamic[1].startswith(dynamic[0][-75:])
        assert len(dynamic[1]) > len(fixed[1])

    def test_resolve_split_spec_defaults_and_clamp(self):
        spec = resolve_split_spec()
        assert (spec["min_chars"], spec["target_chars"], spec["max_chars"]) == (200, 400, 512)
        assert spec["overlap_chars"] == 50
        assert abs(spec["overlap_ratio"] - 0.15) < 1e-9

        clamped = resolve_split_spec(512, 50, 999, 9)
        assert clamped["target_chars"] == MAX_CHUNK_CHARS
        assert clamped["overlap_ratio"] == 0.5

        low = resolve_split_spec(512, 50, 10, -1)
        assert low["target_chars"] == MIN_CHUNK_CHARS
        assert low["overlap_ratio"] == 0.0


class TestSpecDump:
    def test_build_vector_index_dumps_effective_spec(self, tmp_path, monkeypatch):
        """建库须落盘实际生效规格，避免文档承诺与运行值再次失配"""
        import step2_split_embed as s2

        monkeypatch.setattr(s2, "load_embedding_model", lambda spec: (None, "cpu"))

        text = "第一章 起始\n" + "叶凡踏入门中，看见一条长河。" * 30
        out = tmp_path / "vector_db"
        assert s2.build_vector_index(
            text, output_dir=str(out), target_chars=300, overlap_ratio=0.2
        ) is True

        spec = json.loads((out / "split_spec.json").read_text(encoding="utf-8"))
        meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))

        assert spec["min_chars"] == MIN_CHUNK_CHARS
        assert spec["max_chars"] == MAX_CHUNK_CHARS
        assert spec["target_chars"] == 300
        assert abs(spec["overlap_ratio"] - 0.2) < 1e-9
        assert spec["chunk_count"] == len(meta)  # 与元数据对账
        assert 0 < spec["chunk_length_median"] <= spec["chunk_length_max"] <= MAX_CHUNK_CHARS


# ===================== 规则 3：检索端父子块去重 =====================

class TestAssembleContextDedup:
    def test_parent_supersedes_covered_child(self):
        import main as main_mod

        child = "叶凡踏入门中，看见一条长河。"
        parent = child * 3
        hits = [
            {"text": child, "chapter": "第一章", "is_parent": False, "is_neighbor": False},
            {"text": parent, "chapter": "第一章", "is_parent": True, "is_neighbor": False},
        ]
        ctx = main_mod.assemble_context(hits)
        assert "[父块:第一章]" in ctx
        assert ctx.count(child) == 3  # 仅父块内出现，子块未重复输出

    def test_longer_item_replaces_shorter_in_place(self):
        import main as main_mod

        child = "庞博惊呼，却无人应答！"
        parent = child * 2
        hits = [
            {"text": parent, "chapter": "第二章", "is_parent": True, "is_neighbor": False},
            {"text": child, "chapter": "第二章", "is_parent": False, "is_neighbor": False},
        ]
        ctx = main_mod.assemble_context(hits)
        assert ctx.count(child) == 2
        assert "[来源:第二章]" not in ctx  # 子块被父块覆盖

    def test_flat_hits_keep_legacy_output(self):
        import main as main_mod

        hits = [
            {"text": "第一段。", "chapter": "第一章"},
            {"text": "第二段。", "chapter": "第二章"},
        ]
        ctx = main_mod.assemble_context(hits)
        assert ctx == "[来源:第一章]\n第一段。\n\n---\n\n[来源:第二章]\n第二段。"
