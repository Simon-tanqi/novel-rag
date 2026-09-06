# -*- coding: utf-8 -*-
"""测试 novel_context：指代消解前缀注入 + 章节聚合重排。"""
import pytest

from novel_context import (
    DEFAULT_CHARACTERS,
    build_character_table,
    inject_coref_prefix,
    chapter_aggregate_rerank,
)


class TestInjectCorefPrefix:
    def test_chunk_starting_with_pronoun_gets_prefix(self):
        """chunk 以「他」开头且前文出现过主角 → 注入该主角名"""
        chunks = [
            "叶凡踏入荒古禁地，九条龙尸横空而过。",
            "他捡到一枚古玉，上面刻着神秘的纹路。",
        ]
        chapters = ["第一章 荒古禁地", "第一章 荒古禁地"]
        out = inject_coref_prefix(chunks, chapters, characters=["叶凡"])
        assert out[0] == chunks[0]                      # 首块显式点名，不注入
        assert out[1].startswith("【前文主语：叶凡】")   # 指代块注入
        assert "他捡到一枚古玉" in out[1]               # 原文保留

    def test_no_pronoun_chunk_unchanged(self):
        chunks = ["叶凡抬头望向天空。", "远处传来兽吼声。"]
        out = inject_coref_prefix(chunks, ["第一章"] * 2, characters=["叶凡"])
        assert out == chunks

    def test_unresolvable_pronoun_unchanged(self):
        """chunk 以「他」开头但前文与行内均无主角名 → 不注入（不制造噪音）"""
        chunks = ["他站在一片陌生的荒原上。"]
        out = inject_coref_prefix(chunks, ["第一章"], characters=["叶凡"])
        assert out == chunks

    def test_inline_name_takes_priority(self):
        """指代句行内点名（他叫叶凡/他对叶凡说）→ 用行内名而非前文名"""
        chunks = [
            "林雷握着盘龙戒指，心中一动。",
            "他对迪莉娅说：我们走吧。",
        ]
        out = inject_coref_prefix(chunks, ["第一章"] * 2,
                                  characters=["林雷", "迪莉娅"])
        assert "迪莉娅" in out[1].split("\n")[0]

    def test_quoted_lines_not_used_as_subject_source(self):
        """对话行（引号结尾）不贡献主语；后续指代应回退到更早的叙述行"""
        chunks = [
            "庞博喊道：这里太危险了！",
            "他拉住叶凡，转身就跑。",
        ]
        out = inject_coref_prefix(chunks, ["第一章"] * 2, characters=["庞博"])
        assert out[1].startswith("【前文主语：庞博】")

    def test_window_crosses_chunk_boundary(self):
        """跨 chunk 连续：前一 chunk 的主角名对后一 chunk 的指代可见"""
        chunks = [
            "林雷闭关十年，终于突破。",
            "他睁开眼，看向窗外。",
            "他笑了。",
        ]
        out = inject_coref_prefix(chunks, ["第1章"] * 3, characters=["林雷"])
        assert out[1].startswith("【前文主语：林雷】")
        assert out[2].startswith("【前文主语：林雷】")

    def test_returns_same_length(self):
        chunks = ["短。", "他继续说。", "完了。"]
        out = inject_coref_prefix(chunks, ["第一章"] * 3, characters=["林雷"])
        assert len(out) == len(chunks)


class TestChapterAggregateRerank:
    def _hit(self, chapter, score, text="片段"):
        return {"chapter": chapter, "score": score, "text": text}

    def test_deduplicates_same_chapter(self):
        """同章多条命中 → 仅保留最高分，覆盖更多章节"""
        hits = [
            self._hit("第三章", 0.9),
            self._hit("第三章", 0.8),
            self._hit("第三章", 0.7),
            self._hit("第五章", 0.6),
        ]
        out = chapter_aggregate_rerank(hits, top_k=3)
        chapters = [h["chapter"] for h in out]
        assert chapters == ["第三章", "第五章"]  # 去重且按分排序

    def test_respects_top_k(self):
        hits = [self._hit(f"第{i}章", 1.0 - i * 0.1) for i in range(1, 8)]
        out = chapter_aggregate_rerank(hits, top_k=3)
        assert len(out) == 3
        assert [h["chapter"] for h in out] == ["第1章", "第2章", "第3章"]

    def test_empty_input(self):
        assert chapter_aggregate_rerank([], top_k=3) == []

    def test_no_chapter_fallback(self):
        hits = [{"chapter": "", "score": 0.5, "text": "x"},
                self._hit("第一章", 0.9)]
        out = chapter_aggregate_rerank(hits, top_k=2)
        # 无章节条目兜底排最后
        assert out[-1]["chapter"] == ""


class TestBuildCharacterTable:
    def test_merges_known_and_frequent(self):
        text = ("叶凡与庞博同行。叶凡说：我们走。庞博点头。"
                "叶凡又笑。庞博也笑了。")
        table = build_character_table(text, known=["姬紫月"])
        assert "姬紫月" in table          # 已知名单保留
        assert "叶凡" in table            # 高频自动补充
        assert "庞博" in table
        assert "我们" not in table        # 停用词过滤
