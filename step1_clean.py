"""
step1_clean.py — 文本清洗模块
提供小说文本清洗功能，支持多种清洗规则和自定义脏数据关键词

清洗规则:
- remove_empty_lines: 去除空行
- remove_page_numbers: 去除页码
- remove_ads: 去除广告/水印
- merge_paragraphs: 合并段落
- fix_encoding: 修复编码错误
- remove_pinyin: 去除拼音残留
"""
import re
import os
from typing import List, Optional

from utils import get_root_dir, is_chapter_title


# ===================== 编码错误映射表 =====================
GBK_ERROR_MAP = {
    '入马': '人马', '入物': '人物', '入生': '人生', '入间': '人间', '入命': '人命',
    '入心': '人心', '入眼': '人眼', '入手': '人手', '入面': '人面',
    '入身': '人身', '入头': '人头', '入脸': '人脸', '入群': '人群', '入流': '人流',
    '入海': '人海', '入山': '人山', '入世': '人世', '入界': '人界',
    '入道': '人道', '入德': '人德', '入品': '人品', '入格': '人格', '入性': '人性',
    '入情': '人情', '入欲': '人欲', '入意': '人意', '入志': '人志',
    '入力': '人力', '入才': '人才', '入杰': '人杰', '入雄': '人雄', '入豪': '人豪',
    '入王': '人王', '入皇': '人皇', '入帝': '人帝', '入圣': '人圣', '入仙': '人仙',
    '入神': '人神', '入魔': '人魔', '入鬼': '人鬼', '入妖': '人妖', '入精': '人精',
    '入灵': '人灵', '入魂': '人魂', '入魄': '人魄', '入气': '人气', '入势': '人势',
    '入运': '人运', '入寿': '人寿', '入福': '人福', '入祸': '人祸',
    '入灾': '人灾', '入劫': '人劫', '入难': '人难', '入苦': '人苦', '入乐': '人乐',
    '入喜': '人喜', '入怒': '人怒', '入哀': '人哀', '入愁': '人愁', '入忧': '人忧',
    '入思': '人思', '入念': '人念', '入想': '人想', '入知': '人知', '入觉': '人觉',
    '入悟': '人悟', '入明': '人明', '入暗': '人暗', '入阴': '人阴', '入阳': '人阳',
    '夭马': '天马', '夭皇': '天皇', '夭赋': '天赋', '夭才': '天才', '夭道': '天道',
    '夭地': '天地', '夭罚': '天罚', '夭威': '天威', '夭命': '天命', '夭意': '天意',
    '夭机': '天机', '夭象': '天象', '夭文': '天文', '夭数': '天数', '夭运': '天运',
    '夭光': '天光', '夭色': '天色', '夭云': '天云', '夭风': '天风', '夭雨': '天雨',
    '夭雷': '天雷', '夭电': '天电', '夭火': '天火', '夭水': '天水',
    '夭山': '天山', '夭海': '天海', '夭河': '天河', '夭星': '天星', '夭月': '天月',
    '夭德': '天德', '夭心': '天心', '夭理': '天理', '夭法': '天法',
    '夭路': '天路', '夭途': '天途', '夭门': '天门', '夭涯': '天涯',
    '夭宇': '天宇', '夭宙': '天宙', '夭空': '天空', '夭际': '天际',
    '夭时': '天时', '夭人': '天人', '夭物': '天物', '夭志': '天志',
    '夭规': '天规', '夭律': '天律', '夭则': '天则', '夭经': '天经',
    '夭纬': '天纬', '夭纲': '天纲', '夭常': '天常', '夭伦': '天伦',
    '夭梯': '天梯', '夭桥': '天桥', '夭窗': '天窗', '夭顶': '天顶',
    '夭底': '天底', '夭角': '天角', '夭穹': '天穹',
    '竞然': '竟然', '竞是': '竟是', '竞有': '竟有', '竞敢': '竟敢', '竞会': '竟会',
    '镇龘压': '镇压', '镇龘': '镇',
}

GARBLED_CHARS = {
    '龘': '压', '垚': '堆', '犇': '奔', '骉': '马', '鱻': '鲜', '麤': '粗',
    '飝': '飞', '靐': '雷', '焱': '火', '淼': '水', '鑫': '金', '森': '木',
}

# 拼音 → 汉字映射。
# 注意：dict 字面量中后键会静默覆盖前键，曾出现 'yīn' 重复定义
# （'音' 被 '阴' 覆盖）导致映射静默失效；新增键前请确认无重复。
PINYIN_MAP = {
    'sè': '色', 'rì': '日', 'zhàn': '战', 'bō': '波',
    'jīng': '精', 'mō': '摸', 'chén': '沉', 'xùn': '迅',
    'nù': '怒', 'qì': '气', 'yì': '意', 'wēi': '威',
    'zhèn': '震', 'hōng': '轰', 'cāng': '苍', 'máng': '茫',
    'hán': '寒', 'qīng': '清', 'yōu': '幽', 'xuán': '玄',
    'míng': '明', 'àn': '暗', 'guāng': '光', 'yuè': '月',
    'xīng': '星', 'lóng': '龙', 'fēng': '风', 'yún': '云',
    'léi': '雷', 'diàn': '电', 'huǒ': '火', 'shuǐ': '水',
    'shān': '山', 'hé': '河', 'hǎi': '海', 'tiān': '天',
    'dì': '地', 'rén': '人', 'xīn': '心', 'yǎn': '眼',
    'yǔ': '雨', 'yǐng': '影', 'shēng': '声', 'yīn': '音',
    'dào': '道', 'fǎ': '法', 'lì': '力', 'dà': '大',
    'xiǎo': '小', 'gāo': '高', 'dī': '低', 'cháng': '长',
    'duǎn': '短', 'zhòng': '重', 'kuài': '快', 'màn': '慢',
    'qián': '前', 'nán': '南', 'běi': '北', 'dōng': '东',
    'xī': '西', 'zhōng': '中', 'mén': '门', 'shū': '书',
    'gōng': '功', 'dòu': '斗', 'shā': '杀', 'zhǎn': '斩',
    'sǐ': '死', 'yǒu': '有', 'wú': '无', 'shí': '实',
    'zhēn': '真', 'hǎo': '好', 'xǐ': '喜', 'lè': '乐',
    'yáng': '阳', 'lěng': '冷', 'rè': '热',
    'chūn': '春', 'jiǔ': '九', 'bǎi': '百', 'qiān': '千',
    'wàn': '万', 'shén': '神', 'xiān': '仙', 'mó': '魔',
}

# 清洗规则映射
RULE_FUNCTIONS = {
    "去除空行": "remove_empty_lines",
    "去除页码": "remove_page_numbers",
    "去除广告": "remove_ads",
    "合并段落": "merge_paragraphs",
    "修复编码": "fix_encoding",
    "去除拼音": "remove_pinyin",
    "结构归一": "normalize_structure",
}

# 默认启用的规则：新切片引擎（chunking.py）以「空行/段落边界」为最高优先级
# 切分点，而「合并段落」会把段落与场景边界抹平（只剩标题行），故默认关闭；
# 仍可在 GUI / CLI 显式勾选，兼容旧流程。
DEFAULT_RULES = [name for name in RULE_FUNCTIONS if name != "合并段落"]


# ===================== 编码检测 =====================
def detect_encoding(file_path: str) -> str:
    """
    检测文件编码

    Args:
        file_path: 文件路径

    Returns:
        编码名称
    """
    try:
        import chardet
        with open(file_path, 'rb') as f:
            raw = f.read(10000)
            result = chardet.detect(raw)
            encoding = result['encoding']
            confidence = result['confidence']
            print(f"  检测编码: {encoding} (置信度: {confidence:.2%})")
            return encoding
    except ImportError:
        for enc in ['utf-8', 'gb18030', 'gbk', 'gb2312']:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    content = f.read(10000)
                    if len(content) > 1000:
                        print(f"  检测编码: {enc}")
                        return enc
            except Exception:
                continue
        print("  无法检测编码，使用utf-8")
        return 'utf-8'


# ===================== 清洗规则实现 =====================
def normalize_structure(text: str) -> str:
    """结构预处理：统一换行 → 压缩行内空白与缩进 → 连续空行归一为 1 个空行。

    空行是**场景边界**（章内按空行分场景，见 chunking.parse_structure），
    必须保留且唯一化：末步归一化只做「连续空行 → 1 个空行」，
    不得把空行整体抹平，否则下游丢失段落/场景边界，切块只能退化为纯标点找点。
    """
    if not text:
        return ""
    text = (text.replace('\r\n', '\n').replace('\r', '\n')
                .replace('\u2028', '\n').replace('\u2029', '\n')
                .replace('\u0085', '\n').replace('\u3000', ' '))
    text = re.sub(r'[ \t]+', ' ', text)

    lines: List[str] = []
    blank = False
    for line in text.split('\n'):
        stripped = line.strip()
        if stripped:
            lines.append(stripped)
            blank = False
        elif lines and not blank:
            lines.append('')  # 场景边界：连续空行折叠为 1 个
            blank = True
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return '\n'.join(lines)


# 内部别名（与既有 _xxx 风格一致；RULE_FUNCTIONS 映射到公共名）
_normalize_structure = normalize_structure


def _remove_empty_lines(text: str) -> str:
    """去除空行和仅含空白的行"""
    lines = text.split('\n')
    cleaned = [line for line in lines if line.strip()]
    return '\n'.join(cleaned)


def _remove_page_numbers(text: str) -> str:
    """去除页码（数字行、'第X页'等）"""
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # 跳过纯数字行
        if re.match(r'^\d+\s*$', stripped):
            continue
        # 跳过页码格式
        if re.match(r'^第[一二三四五六七八九十百千万\d]+页\s*$', stripped):
            continue
        # 跳过 - 数字 - 格式
        if re.match(r'^-?\d+-?\s*$', stripped):
            continue
        cleaned.append(line)
    return '\n'.join(cleaned)


# ===================== 广告/水印行判定 =====================
# 强导航信号：整行命中任一 → 高度疑似书源水印行，删除。
# （站点名 / 网址 / 祈使语 / 更新提示 —— 结构特征，正文几乎不会整行出现）
SITE_MARKERS = [
    '看小说到网', '笔趣阁', '顶点小说', '燃文', '看书网', '16977',
    'www.', 'http://', 'https://', '.com', '.net', 'txt下载', '全本',
    '本站', '首发', '无弹窗', '手打更新', '最快更新', '更新最快',
    '最新章节', '阅读最新章节', '本章未完', '阅读网址', '手机版阅读网址',
    '手机阅读', '加入书签', '推荐阅读', '天才一秒记住', '请收藏',
    '请牢记', '如果您觉得', '请到', '支持正版', '看书',
]
# 泛词：不单独触发删除（正文高频），仅作为行尾括号水印的辅助匹配词
WATERMARK_HINTS = (
    '下载', '免费阅读', '全文', '提供', '正版', '手打', '小游戏',
)
# 行尾括号水印匹配（先剔除再判定，避免正文误删）
_TRAILING_BRACKET_RE = re.compile(
    r'[（(【\[][^）)】\]]{0,60}?(?:未完待续|本章未完|首发|阅读网址|请收藏|'
    r'推荐阅读|手打|更新最快|最新章节|免费阅读|请牢记|看小说到网).{0,40}?'
    r'[）)】\]]\s*$'
)


# 整行水印判定的行长上限：超过该长度的行不判为水印（正文长句保护）。
# 由 80 放宽到 200 —— 书源水印常见整句式（网址 + 更新提示 + 祈使语，
# 长度易超 80 字）会被旧上限漏删；判定依据是 SITE_MARKERS 多词同现的
# 结构特征，放宽上限不显著增加正文误杀。
MAX_WATERMARK_LINE_CHARS = 200


def _line_is_watermark(stripped: str) -> bool:
    """判断整行是否为广告/水印行（命中即应删除）

    判定前须已剔除行尾括号水印。策略（防误杀）：
    - 行长超过 MAX_WATERMARK_LINE_CHARS(200) → 不判为水印（正文长句保护）；
    - 含任一强导航信号（站点/网址/祈使语/更新提示）→ 删除；
    - 仅含泛词时：不删除 —— 正文里“下载/免费/正版”等词很常见，
      单凭泛词删整行会误杀正常叙述。
    """
    if not stripped or len(stripped) > MAX_WATERMARK_LINE_CHARS:
        return False
    if any(marker in stripped for marker in SITE_MARKERS):
        return True
    return False


def _remove_ads(text: str, custom_words: Optional[List[str]] = None) -> str:
    """
    去除广告和水印（防误杀设计）

    - 每行先剔除行尾括号水印（如“（本章未完，请点击下一页继续阅读）”），
      剩余正文保留——绝不因正文里出现“免费/下载”等泛词删整行；
    - 剔除水印尾巴后，若整行仍命中强水印词 / 弱词+语境组合 → 整行删除；
    - custom_words 视为强水印词：短行命中即删除。

    Args:
        text: 输入文本
        custom_words: 自定义关键词列表
    """
    custom = list(custom_words) if custom_words else []

    lines = text.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append(line)
            continue
        # 1) 剔除行尾括号水印（保留正文）
        m = _TRAILING_BRACKET_RE.search(stripped)
        if m:
            stripped = stripped[:m.start()].rstrip()
            if not stripped:
                continue  # 整行只是括号水印，删
        # 2) 整行水印判定
        if _line_is_watermark(stripped):
            continue
        # 3) 自定义脏词（仅短行命中才删——长句是正文叙述，不整行删）
        if any(kw in stripped for kw in custom) and len(stripped) <= 40:
            continue
        cleaned.append(stripped)
    return '\n'.join(cleaned)


def _merge_paragraphs(text: str) -> str:
    """合并段落（将正文内的换行合并为空格，但保留章节标题行结构）

    章标题（第X章 / 序章 / 楔子…）是文档的结构分隔符，若被合并进正文行，
    后续按章节切片将无法定位章节边界。因此标题行单独保留并作为分段。
    """
    lines = text.split('\n')
    merged = []  # 元素: 正文行 or 章节标题行
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _looks_like_chapter_title(stripped):
            # 标题另起一段（保留结构）
            merged.append(stripped)
        else:
            if merged and not _looks_like_chapter_title(merged[-1]):
                merged[-1] = merged[-1] + stripped
            else:
                merged.append(stripped)
    # 标题与正文之间用换行分隔，正文段落之间用空格（保持可读性）
    result = []
    for i, line in enumerate(merged):
        if i > 0 and _looks_like_chapter_title(line):
            result.append('')  # 标题前空行
        result.append(line)
    return '\n'.join(result).strip('\n')


def _looks_like_chapter_title(line: str) -> bool:
    """章节标题判定（转调 utils.is_chapter_title，全项目单一实现）。

    标题需同时满足「行首第X章 + 短行 + 不含句读」；判定逻辑集中在此，
    避免清洗阶段误吞带！？的标题。
    """
    return is_chapter_title(line)


def _fix_encoding(text: str) -> str:
    """修复编码错误"""
    for wrong, correct in GBK_ERROR_MAP.items():
        text = text.replace(wrong, correct)
    for garbled, correct in GARBLED_CHARS.items():
        text = text.replace(garbled, correct)
    return text


def _remove_pinyin(text: str) -> str:
    """去除拼音残留"""
    for pinyin, hanzi in PINYIN_MAP.items():
        text = text.replace(pinyin, hanzi)
    return text


# ===================== 主清洗函数 =====================
def clean_text(
    text: str,
    rules: Optional[List[str]] = None,
    custom_words: Optional[List[str]] = None,
    progress_callback=None
) -> str:
    """
    完整的文本清洗流程

    Args:
        text: 原始文本
        rules: 清洗规则列表（中文名称）
        custom_words: 自定义脏数据关键词
        progress_callback: 进度回调函数 (step: int, total: int, message: str)

    Returns:
        清洗后的文本
    """
    default_rules = list(DEFAULT_RULES)
    active_rules = rules if rules else default_rules
    total_steps = len(active_rules) + 1  # +1 for final normalization
    current_step = 0

    # 1. 修复编码
    if "修复编码" in active_rules:
        current_step += 1
        if progress_callback:
            progress_callback(current_step, total_steps, "修复编码错误...")
        text = _fix_encoding(text)

    # 2. 去除拼音
    if "去除拼音" in active_rules:
        current_step += 1
        if progress_callback:
            progress_callback(current_step, total_steps, "去除拼音残留...")
        text = _remove_pinyin(text)

    # 3. 去除空行
    if "去除空行" in active_rules:
        current_step += 1
        if progress_callback:
            progress_callback(current_step, total_steps, "去除空行...")
        text = _remove_empty_lines(text)

    # 4. 去除页码
    if "去除页码" in active_rules:
        current_step += 1
        if progress_callback:
            progress_callback(current_step, total_steps, "去除页码...")
        text = _remove_page_numbers(text)

    # 5. 去除广告
    if "去除广告" in active_rules:
        current_step += 1
        if progress_callback:
            progress_callback(current_step, total_steps, "去除广告和水印...")
        text = _remove_ads(text, custom_words)

    # 6. 合并段落
    if "合并段落" in active_rules:
        current_step += 1
        if progress_callback:
            progress_callback(current_step, total_steps, "合并段落...")
        text = _merge_paragraphs(text)

    # 7. 结构归一（统一换行 + 空行唯一化为场景边界 + 剔章首章尾空行）
    current_step += 1
    if progress_callback:
        progress_callback(current_step, total_steps, "结构预处理（换行/空行归一）...")
    text = _normalize_structure(text)

    if progress_callback:
        progress_callback(total_steps, total_steps, "清洗完成！")

    return text


# ===================== 文件清洗便捷函数 =====================
def clean_file(
    input_path: str,
    output_path: str,
    rules: Optional[List[str]] = None,
    custom_words: Optional[List[str]] = None,
    progress_callback=None
) -> bool:
    """
    清洗文件并保存

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        rules: 清洗规则列表
        custom_words: 自定义关键词
        progress_callback: 进度回调

    Returns:
        是否成功
    """
    try:
        # 检测编码并读取
        encoding = detect_encoding(input_path)
        with open(input_path, 'r', encoding=encoding, errors='ignore') as f:
            text = f.read()

        original_length = len(text)
        if progress_callback:
            progress_callback(0, 100, f"读取文件: {original_length} 字符")

        # 清洗
        cleaned = clean_text(text, rules, custom_words, progress_callback)

        # 保存
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned)

        if progress_callback:
            progress_callback(100, 100, f"清洗完成: {original_length} → {len(cleaned)} 字符")

        return True

    except Exception as e:
        print(f"清洗文件失败: {e}")
        if progress_callback:
            progress_callback(-1, 100, f"清洗失败: {str(e)}")
        return False


def get_available_rules() -> List[str]:
    """获取所有可用的清洗规则名称"""
    return list(RULE_FUNCTIONS.keys())


def get_default_rules() -> List[str]:
    """获取默认启用的清洗规则（排除「合并段落」，见 DEFAULT_RULES 注释）。"""
    return list(DEFAULT_RULES)


# ===================== 命令行入口（保留） =====================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="文本清洗工具")
    parser.add_argument("--input", required=True, help="输入文件路径")
    parser.add_argument("--output", default=None, help="输出文件路径")
    parser.add_argument("--rules", nargs='*', help="清洗规则")
    parser.add_argument("--words", nargs='*', help="自定义关键词")
    args = parser.parse_args()

    output_path = args.output
    if not output_path:
        base, ext = os.path.splitext(args.input)
        output_path = f"{base}_clean{ext}"

    def print_progress(step, total, msg):
        print(f"[{step}/{total}] {msg}")

    success = clean_file(
        args.input, output_path,
        rules=args.rules,
        custom_words=args.words,
        progress_callback=print_progress
    )

    if success:
        print(f"✓ 清洗完成: {output_path}")
    else:
        print("✗ 清洗失败")
