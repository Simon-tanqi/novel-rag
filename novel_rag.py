"""
novel_rag.py — NovelRAG 命令行入口（无 GUI 依赖）

一条命令从「一本新小说的 txt」跑到「可问答的 RAG 项目」，
让任何人在只提供 **小说文本 + API Key** 的情况下快速跑通全流程：

    # 1) 建库：复制原文 → 清洗 → 句子级切片 → 向量化 → 注册项目
    python novel_rag.py ingest ./盘龙.txt --name 盘龙

    # 2) 单次问答（检索原文片段 + LLM 依据片段作答）
    python novel_rag.py ask --name 盘龙 "林雷最强的剑是什么？"

    # 3) 交互式多轮问答
    python novel_rag.py chat --name 盘龙

    # 4) 查看本地已建项目
    python novel_rag.py list

    # 5) 生成一段原创示例小说并跑通全流程（用于验证环境，无需联网语料）
    python novel_rag.py demo

API Key 提供方式（任选其一，优先级从高到低）：
    1. 环境变量  DEEPSEEK_API_KEY   （推荐，密钥不落盘）
    2. config.json 中 models[].api_key
    3. ask/chat 的 --api-key 参数

嵌入模型（语义检索，可选）：
    默认使用 BAAI/bge-small-zh-v1.5（首次自动下载缓存）；
    也可填本地模型目录路径（如 models/bge-small-zh-v1.5）实现纯离线加载，或清空以使用纯关键词检索。
"""
import argparse
import os
import shutil
import sys
from typing import List, Optional

from api_client import APIClient, APIClientError
from config_manager import ConfigManager
from project_manager import ProjectManager
from rag_retriever import RAGRetriever
from format_loader import is_supported_format, load_raw_text
from step1_clean import RULE_FUNCTIONS, clean_file
from step2_split_embed import build_vector_index_from_file
from utils import DEFAULT_EMBEDDING_MODEL, load_env_file

# 最先加载项目根目录 .env（幂等，仅补未设置键）
load_env_file()

def _enable_utf8_stdout():
    """Windows 控制台默认 GBK，切换为 UTF-8 输出避免编码报错"""
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass


def _print_progress(step, total, msg, percent=0):
    print(f"  [{percent:3.0f}%] {msg}")


def _get_config() -> ConfigManager:
    return ConfigManager()


def _resolve_embedding_spec(cfg: ConfigManager) -> str:
    """嵌入模型优先级：config.json > 环境变量 EMBEDDING_MODEL > 内置默认"""
    return (
        cfg.get("embedding_model_path", "")
        or os.environ.get("EMBEDDING_MODEL", "").strip()
        or DEFAULT_EMBEDDING_MODEL
    )


def _get_project(name: Optional[str]) -> dict:
    pm = ProjectManager()
    project = pm.get_project_by_name(name) if name else pm.get_current_project()
    if not project:
        tip = f"「{name}」" if name else ""
        print(f"✗ 找不到项目{tip}。请先运行: python novel_rag.py ingest <小说.txt> --name <名字>")
        sys.exit(1)
    return project


def _build_prompt(
    template: str,
    novel_name: str,
    context: str,
    question: str,
    history: str = "",
) -> str:
    """按 GUI 同款逻辑组装 Prompt（兼容含/不含 {novel_name} 的模板）"""
    try:
        if novel_name:
            base = template.format(novel_name=novel_name, context=context, question=question)
        else:
            base = template.format(context=context, question=question)
    except (KeyError, ValueError, IndexError):
        # 模板包含无法解析的占位符时退化为直接拼接，保证可用
        base = f"{template}\n\n以下是提供的原文片段：\n{context}\n\n用户的问题：{question}\n\n根据原文片段回答："
    if history:
        return history + base
    return base


def _build_context(hits: List[dict]) -> str:
    if not hits:
        return "未找到相关的原文片段。"
    return "\n\n---\n\n".join(
        f"[来源:{h.get('chapter', '未知')}]\n{h['text']}" for h in hits
    )


def _get_api_params(cfg: ConfigManager, api_key: Optional[str], model_id: Optional[str]):
    """返回 (api_url, api_key, model_name)；缺失时给出友好提示并退出"""
    model = cfg.get_active_model()
    if not model:
        print(
            "✗ 未配置任何模型。请先设置环境变量 DEEPSEEK_API_KEY，\n"
            "  或执行: cp config.example.json config.json  再在 config.json 中填写。"
        )
        sys.exit(1)
    api_url = model.get("api_url", "")
    key = api_key or model.get("api_key", "")
    name = model_id or model.get("model_id") or model.get("name", "")
    if not api_url or not key:
        print(
            "✗ 缺少 API Key 或 API URL。\n"
            "  请任选其一：\n"
            "    1) export DEEPSEEK_API_KEY=sk-xxxx   （推荐）\n"
            "    2) 在 config.json 的 models[].api_key 中填写\n"
            "    3) 本命令加 --api-key sk-xxxx"
        )
        sys.exit(1)
    return api_url, key, name


# ===================== 子命令 =====================

def cmd_ingest(args) -> None:
    """复制原文 → 清洗 → 切片 → 向量化 → 注册项目"""
    if not os.path.isfile(args.file):
        print(f"✗ 源文件不存在: {args.file}")
        sys.exit(1)

    # 格式检查：仅支持 txt / epub
    if not is_supported_format(args.file):
        ext = os.path.splitext(args.file)[1]
        print(f"✗ 不支持的文件格式: {ext}（仅支持 .txt / .epub）")
        sys.exit(1)

    name = args.name or os.path.splitext(os.path.basename(args.file))[0]
    pm = ProjectManager()
    if pm.project_name_exists(name):
        print(f"✗ 已存在同名项目「{name}」（可用 python novel_rag.py list 查看）。")
        sys.exit(1)

    # 清洗规则：未指定则启用全部内置规则（与 GUI 默认一致）
    rules = args.rules or list(RULE_FUNCTIONS.keys())

    print(f"📖 开始为小说「{name}」构建 RAG 项目 ...")
    project = pm.create_project(
        name=name,
        source_path=args.file,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        clean_rules=rules,
        custom_dirty_words=args.words,
    )
    project_id = project["id"]

    # 1) 复制原文到项目源目录（已在目标位置则跳过，避免同文件复制报错）
    source_target = project["source_path"]
    os.makedirs(os.path.dirname(source_target), exist_ok=True)
    if os.path.abspath(args.file) == os.path.abspath(source_target):
        print(f"① 源文件已在项目源目录，跳过复制: {source_target}")
    else:
        print(f"① 复制原文 → {source_target}")
        shutil.copy2(args.file, source_target)

    # 1.5) epub → txt 转换（如果是 epub 格式）
    file_ext = os.path.splitext(source_target)[1].lower()
    if file_ext == ".epub":
        print("① epub → txt 文本提取 ...")
        try:
            raw_text = load_raw_text(source_target)
        except ImportError as e:
            print(f"✗ {e}")
            sys.exit(1)
        except Exception as e:
            print(f"✗ epub 解析失败: {e}")
            sys.exit(1)
        # 提取后的 txt 路径（与源文件同目录同名，换后缀）
        txt_path = os.path.splitext(source_target)[0] + ".txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(raw_text)
        # 后续流程使用提取后的 txt 而非 epub
        source_target = txt_path
        print(f"   提取完成: {len(raw_text)} 字符 → {txt_path}")

    # 2) 清洗
    print("② 文本清洗 ...")
    cleaned_path = project["cleaned_path"]
    ok = clean_file(
        source_target,
        cleaned_path,
        rules=rules,
        custom_words=args.words,
        progress_callback=_print_progress,
    )
    if not ok:
        print("✗ 清洗失败")
        sys.exit(1)
    pm.update_project(project_id, {"status": "cleaned"})

    # 3) 切片 + 向量化
    print("③ 切片与向量化 ...")
    # 嵌入模型优先级：--embedding 参数 > config/环境变量/内置默认。
    # 显式传空串（--embedding "" / demo 场景）表示强制纯关键词模式，不下载模型。
    if args.embedding is not None:
        embedding = args.embedding
    else:
        embedding = _resolve_embedding_spec(_get_config())
    vector_db_path = project["vector_db_path"]
    ok = build_vector_index_from_file(
        input_file=cleaned_path,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        output_dir=vector_db_path,
        embedding_model_path=embedding if embedding else None,
        progress_callback=_print_progress,
    )
    if not ok:
        print("✗ 向量化失败")
        sys.exit(1)

    pm.update_project(project_id, {"status": "ready"})
    pm.set_current_project(project_id)

    print(f"\n✅ 项目「{name}」构建完成！向量库位于: {vector_db_path}")
    print(f"   试试问答: python novel_rag.py ask --name {name} \"你的问题\"")
    print(f"   或开 GUI: python main.py")


def _prepare_retriever(cfg: ConfigManager, project: dict) -> RAGRetriever:
    vector_path = project.get("vector_db_path", "")
    if not vector_path or not os.path.isdir(vector_path):
        print(f"✗ 项目「{project.get('name')}」尚未构建向量库，请先运行 ingest。")
        sys.exit(1)
    return RAGRetriever.get_or_create(
        vector_path,
        reranker_model_path=cfg.get("reranker_model_path", ""),
        embedding_model_path=_resolve_embedding_spec(cfg),
        vector_file=project.get("vector_file"),
        metadata_file=project.get("metadata_file"),
    )


def cmd_ask(args) -> None:
    """单次 RAG 问答"""
    cfg = _get_config()
    project = _get_project(args.name)
    api_url, api_key, model_name = _get_api_params(cfg, args.api_key, args.model_id)

    prompt_template = cfg.get_prompt_template()
    top_k = int(cfg.get("top_k", 3))
    enable_thinking = cfg.get("enable_thinking", False)
    # 重排：命令行 --rerank/--no-rerank 优先，否则取 config.enable_rerank
    enable_rerank = args.rerank if args.rerank is not None else bool(cfg.get("enable_rerank", False))

    retriever = _prepare_retriever(cfg, project)
    print("🔍 正在检索原文片段 ...")
    hits = retriever.retrieve(args.question, top_k=top_k, enable_rerank=enable_rerank)
    context = _build_context(hits)

    if prompt_template:
        prompt = _build_prompt(
            prompt_template,
            project.get("name", ""),
            context,
            args.question,
        )
    else:
        prompt = args.question

    print("🤖 正在调用 LLM ...")
    client = APIClient(api_url, api_key, model_name)
    reply = client.call_api(prompt, enable_thinking=enable_thinking)

    print("\n" + "=" * 50)
    print(f"Q: {args.question}\n")
    print(f"A: {reply}\n")
    if hits:
        print("── 命中原文片段（来源） ──")
        for h in hits:
            score = f"[score={h.get('score'):.3f}]" if isinstance(h.get('score'), (int, float)) else ""
            print(f"  · {h.get('chapter', '未知')} {score}")


def cmd_chat(args) -> None:
    """交互式多轮 RAG 问答（含最近上下文拼接）"""
    cfg = _get_config()
    project = _get_project(args.name)
    api_url, api_key, model_name = _get_api_params(cfg, args.api_key, args.model_id)

    prompt_template = cfg.get_prompt_template()
    top_k = int(cfg.get("top_k", 3))
    enable_thinking = cfg.get("enable_thinking", False)
    # 重排：命令行 --rerank/--no-rerank 优先，否则取 config.enable_rerank
    enable_rerank = args.rerank if args.rerank is not None else bool(cfg.get("enable_rerank", False))
    retriever = _prepare_retriever(cfg, project)
    novel_name = project.get("name", "")

    history_messages: List[dict] = []
    print(f"💬 进入对话模式（项目: {novel_name}）。输入 exit / q 退出。\n")

    while True:
        try:
            question = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit", "q", "退出"):
            print("再见！")
            break

        hits = retriever.retrieve(question, top_k=top_k, enable_rerank=enable_rerank)
        context = _build_context(hits)

        # 多轮历史：截取最近 20 条消息（约 10 轮），单条截断 200 字（与 GUI 一致）
        history_text = ""
        recent = history_messages[-20:]
        if recent:
            lines = ["以下是与用户之前的对话历史（供参考）："]
            for msg in recent:
                role = "用户" if msg["role"] == "user" else "助手"
                lines.append(f"{role}: {msg['content'][:200]}")
            lines.append("---以上是历史记录---\n")
            history_text = "\n".join(lines)

        if prompt_template:
            prompt = _build_prompt(prompt_template, novel_name, context, question, history_text)
        else:
            prompt = question

        reply = APIClient(api_url, api_key, model_name).call_api(prompt, enable_thinking=enable_thinking)
        print(f"AI: {reply}\n")

        history_messages.append({"role": "user", "content": question})
        history_messages.append({"role": "assistant", "content": reply})


def cmd_list(_args) -> None:
    """列出本地项目与状态"""
    pm = ProjectManager()
    projects = pm.get_all_projects()
    if not projects:
        print("暂无项目。先导入一本小说: python novel_rag.py ingest <小说.txt> --name <名字>")
        return
    print(f"共 {len(projects)} 个项目:\n")
    print(f"{'项目名称':<16} {'状态':<10} 向量库")
    print("-" * 70)
    for p in projects:
        status = pm.get_project_status(p["id"])
        print(f"{p['name']:<16} {status:<10} {p.get('vector_db_path', '')}")


def cmd_demo(args) -> None:
    """内置原创 demo：优先复用随仓库预置的向量库；全新环境则完整构建。

    路径：
      1) demo 已注册且向量库就绪 → 打印就绪信息 + 尝鲜问题卡
      2) data/demo 数据在但未注册（clone 后首跑）→ ensure_demo_project 自动注册复用
      3) 全新环境 → 生成原文到 data/demo/source → 走 cmd_ingest 完整构建

    --force: 删除现有 demo 项目与数据目录后重建（如升级为真实语义向量库）
    --no-embedding: 强制纯关键词模式，不触发嵌入模型下载
    """
    root = os.path.dirname(os.path.abspath(__file__))
    demo_source_dir = os.path.join(root, "data", "demo", "source")
    demo_file = os.path.join(demo_source_dir, "demo_novel.txt")

    pm = ProjectManager()
    project = pm.get_project_by_name("demo")

    # --force：删除现有 demo 项目（delete_project 连带删除 data/demo 目录）
    if args.force and project:
        print(f"✓ 已指定 --force，删除现有 demo 项目并重建 ...")
        pm.delete_project(project["id"])
        project = None

    # 目录可能在删除/重建后已不存在，此处统一重新创建
    os.makedirs(demo_source_dir, exist_ok=True)

    if project:
        vector_path = project.get("vector_db_path", "")
        has_vectors = bool(vector_path) and os.path.isfile(
            os.path.join(vector_path, "embeddings.npy"))
        mode = "语义向量库" if has_vectors else "关键词检索模式（未含向量文件）"
        print(f"✓ demo 项目已就绪（{mode}），可直接问答：")
        print(f"    python novel_rag.py ask --name demo \"灰衣老者是谁？\"")
        print(f"  或开 GUI: python main.py")
        if not has_vectors:
            print("  提示: 如需真实语义向量库，先安装 sentence-transformers，再执行 demo --force")
        print_demo_qa()
        return

    # 写入/更新 demo 原文（幂等；force 重建后目录已重新创建）
    with open(demo_file, "w", encoding="utf-8") as f:
        f.write(_DEMO_STORY)

    # 数据在但未注册（如删除 novel_config.json 后）→ ensure 自动注册复用
    pm.ensure_demo_project()
    project = pm.get_project_by_name("demo")
    if project:
        print(f"✓ demo 项目已注册（复用 data/demo 现有数据），向量库: "
              f"{project['vector_db_path']}")
        print_demo_qa()
        return

    # 全新构建：完整 ingest 流程（清洗 → 切片 → 向量化 → 注册）
    print(f"✓ 已生成原创示例小说（{len(_DEMO_STORY)} 字，虚构无版权）: {demo_file}")
    ingest_args = argparse.Namespace(
        file=demo_file,
        name="demo",
        chunk_size=500,
        overlap=50,
        rules=None,
        words=None,
        # 默认与 ingest 一致（config/环境变量/内置 bge 模型）；
        # --no-embedding 时强制关键词模式，不触发模型下载
        embedding="" if getattr(args, "no_embedding", False) else None,
    )
    cmd_ingest(ingest_args)
    print_demo_qa()


# 原创示例小说（用于 demo / 冒烟测试；内容为虚构创作，无版权风险）
_DEMO_STORY = """第一章 山门初开
青云山连绵三百里，主峰凌云直插云霄。每年秋分，山下的村子便有人上山拜师，可真正能走到山顶剑庐的，十年未必有一人。
少年沈青背着破旧书箱，在石阶上歇了第五次。他擦了擦汗，从书箱里摸出半个干饼，就着山泉水咽了下去。祖父临终前告诉他，剑庐藏着一卷青云剑诀，若是能拜入剑庐学得此诀，便可下山替村里守井。
“小子，你背的是什么书？”石阶旁的松树下，一个灰衣老者忽然开口。
沈青一惊，连忙行礼：“回老丈，是《青山药典》，家祖行医留下的。”
灰衣老者挑了挑眉：“哦？药典里可写着，秋分露水泡的茶，能解百毒？”
“写着呢。”沈青认真道，“但需配上三年以上的老陈皮，否则药性太烈。”
老者哈哈一笑，指了指山顶：“那你上来吧。记住，剑庐第一课，不教剑，教做人。”

第二章 剑庐第一课
沈青跟着老者上了山，才发现剑庐并不大，只有三进院落，前院练剑，中院藏书，后院住人。弟子共七人，他是最小的那个。
大师兄赵铁山是个黑脸汉子，见面便给了沈青一把扫帚：“新来的，先把前院三百块青砖扫干净。剑庐第一课：心要静。”
沈青没有抱怨。他扫了三天砖，第四天清晨，灰衣老者——原来就是剑庐主人柳青山——把他叫到中院，指着一面木墙道：“墙上有一千道剑痕，你看得出哪一道最深？”
沈青凑近看了半晌，指着墙角一道几乎被磨平的浅痕：“这道。因为它被后来所有剑痕都避开了。”
柳青山眼中精光一闪：“何以见得？”
“剑痕深，是力到；剑痕避让，是意在。能让后来的剑客主动绕开，说明这一剑曾令所有人忌惮。这应是祖师爷留下的定山一剑。”
柳青山沉默许久，忽然道：“明日辰时，来后山。我教你青云剑诀第一式。”

第三章 青云剑诀
青云剑诀共七式，讲究“剑随云走，意在剑先”。第一式叫“云起”，第二式“山隐”，第三式“雨落”，第七式才是压箱底的“剑开天门”。
沈青学得极快。旁人练“云起”要三个月，他只用了十天。柳青山却并不高兴，反而罚他每日多劈柴两个时辰。
“你太聪明了，聪明人会走捷径。”柳青山说，“剑开天门那一式，祖师爷练了四十年才敢出剑。你若要学，先把‘雨落’练到一万遍。”
沈青没有反驳。他白天练剑，夜里翻《青山药典》，把山上的草药认了个遍。第三年秋，他终于把“雨落”练满一万遍。出剑时，剑尖点落的水珠竟能悬在空中一瞬。

第四章 井水之争
沈青下山那年十九岁。村里的井却被人占了——邻村钱庄的护院李彪仗着学过几手粗浅拳脚，说这井本是他家祖产，要村里每月交两贯钱。
沈青站在井边，看着李彪腰间的刀，忽然想起柳青山的话：“剑庐弟子，不轻易出剑；一旦出剑，就要对得起剑。”
“李叔，”沈青拱手道，“井水关乎全村活命。我愿与您比试一场，若我输了，每月三贯；若我赢了，请您撤了这规矩。”
李彪冷笑，拔刀便砍。沈青侧身，剑不出鞘，只以剑鞘轻点李彪手腕。当啷一声，钢刀落地。李彪呆立当场，半晌才道：“你……你这是哪家功夫？”
“青云山，剑庐，雨落。”沈青收剑，躬身道，“得罪了。”
自此，村里恢复了安宁。沈青在井边立了一块小碑，碑上刻着一行字：剑庐第一课，教做人。

第五章 剑开天门
十年后，柳青山病重。沈青赶回剑庐，跪在榻前。
“你已练成雨落，可以学最后一式了。”柳青山咳了两声，“剑开天门，不是破天，是破自己的心障。当年我师父说，这一剑要等一个‘愿意为一口井出剑’的人，才传得下去。”
沈青含泪点头。他在后山练了整整一个月。出剑那日，满山云雾忽然向两侧分开，一道剑光直上九霄，久久不散。
青云山的老人们说，那一夜，他们看见天门开了。
后来，沈青成了剑庐第二任主人。他把《青山药典》和青云剑诀并排放在中院书架上，对新入门的弟子只说一句话：
“先学会为一碗水、一口井出剑，再谈剑开天门。”
"""


# 内置 demo 尝鲜问题卡（问题 + 参考答案 + 预期章节）
# 供 eval_retrieval.py 评估与 CLI demo 命令共用（单一事实源）；
# 答案均出自 _DEMO_STORY 原文，可向 demo 项目直接提问验证。
DEMO_QA = [
    {"question": "青云剑诀的心法口诀是什么？",
     "answer": "剑随云走，意在剑先。",
     "chapter": "第三章 青云剑诀"},
    {"question": "沈青为何会被罚每日多劈柴两个时辰？",
     "answer": "柳青山认为他太聪明、聪明人会走捷径，罚他磨心性。",
     "chapter": "第三章 青云剑诀"},
    {"question": "剑庐第一课教什么？",
     "answer": "心要静（大师兄赵铁山先让他扫三百块青砖，言“剑庐第一课：心要静”；老者亦言“不教剑，教做人”）。",
     "chapter": "第二章 剑庐第一课"},
    {"question": "沈青用什么招式击败了李彪？",
     "answer": "雨落——剑不出鞘，只以剑鞘轻点李彪手腕，钢刀落地。",
     "chapter": "第四章 井水之争"},
    {"question": "剑开天门是什么意思？",
     "answer": "不是破天，是破自己的心障。",
     "chapter": "第五章 剑开天门"},
    {"question": "灰衣老者是谁？",
     "answer": "剑庐主人柳青山（沈青入门后揭晓）。",
     "chapter": "第二章 剑庐第一课"},
    {"question": "沈青背的是什么书？",
     "answer": "《青山药典》（家祖行医留下的）。",
     "chapter": "第一章 山门初开"},
    {"question": "沈青下山那年村里的井被谁占了？",
     "answer": "邻村钱庄的护院李彪，仗着学过拳脚说是他家祖产，要每月两贯钱。",
     "chapter": "第四章 井水之争"},
    {"question": "沈青学完雨落用了多久？",
     "answer": "第三年秋才把“雨落”练满一万遍（约三年）。",
     "chapter": "第三章 青云剑诀"},
    {"question": "青云剑诀共几式？",
     "answer": "共七式：云起、山隐、雨落……第七式剑开天门（压箱底）。",
     "chapter": "第三章 青云剑诀"},
    {"question": "沈青在井边立碑刻了什么字？",
     "answer": "“剑庐第一课，教做人”。",
     "chapter": "第四章 井水之争"},
    {"question": "大师兄给了沈青什么？",
     "answer": "一把扫帚（让他先扫干净前院三百块青砖）。",
     "chapter": "第二章 剑庐第一课"},
]


def print_demo_qa() -> None:
    """打印 demo 尝鲜问题卡：照着问必有答案（用于 demo 命令收尾/就绪提示）"""
    print("\n  尝鲜问题卡（照着问，答案都出自 demo 原文）：")
    for i, qa in enumerate(DEMO_QA, 1):
        print(f"    {i:>2}. {qa['question']}")
        print(f"       参考答案: {qa['answer']}")
    print("\n  例如: python novel_rag.py ask --name demo \"沈青背的是什么书？\"")


# ===================== CLI =====================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="novel_rag",
        description="NovelRAG 命令行工具：一本小说 txt + API Key → 可问答的 RAG 项目",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="构建 RAG 项目（清洗→切片→向量化）")
    p_ingest.add_argument("file", help="小说文件路径（.txt 或 .epub）")
    p_ingest.add_argument("--name", help="项目名（默认取文件名）")
    p_ingest.add_argument("--chunk-size", type=int, default=512, help="切片硬上限（字符数，默认 512）")
    p_ingest.add_argument("--overlap", type=int, default=50, help="相邻片段重叠字符数（默认 50）")
    p_ingest.add_argument("--embedding", help="嵌入模型：本地目录路径 或 HF 模型名（默认 bge-small-zh-v1.5）")
    p_ingest.add_argument("--rules", nargs="*", help="清洗规则子集（默认全部规则）")
    p_ingest.add_argument("--words", nargs="*", help="自定义脏词（用于去除广告规则）")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="单次 RAG 问答")
    p_ask.add_argument("question", help="问题")
    p_ask.add_argument("--name", help="项目名（默认最近使用的项目）")
    p_ask.add_argument("--top-k", type=int, help="召回片段数（默认取 config）")
    p_ask.add_argument("--api-key", help="临时指定 API Key")
    p_ask.add_argument("--model-id", help="临时指定模型 ID")
    p_ask.add_argument("--rerank", action="store_true", default=None, help="强制开启 CrossEncoder 重排（覆盖 config）")
    p_ask.add_argument("--no-rerank", action="store_false", dest="rerank", help="强制关闭重排（覆盖 config）")
    p_ask.set_defaults(func=cmd_ask)

    p_chat = sub.add_parser("chat", help="交互式多轮 RAG 问答")
    p_chat.add_argument("--name", help="项目名（默认最近使用的项目）")
    p_chat.add_argument("--api-key", help="临时指定 API Key")
    p_chat.add_argument("--model-id", help="临时指定模型 ID")
    p_chat.add_argument("--rerank", action="store_true", default=None, help="强制开启 CrossEncoder 重排（覆盖 config）")
    p_chat.add_argument("--no-rerank", action="store_false", dest="rerank", help="强制关闭重排（覆盖 config）")
    p_chat.set_defaults(func=cmd_chat)

    p_list = sub.add_parser("list", help="列出本地项目")
    p_list.set_defaults(func=cmd_list)

    p_demo = sub.add_parser(
        "demo",
        help="内置原创示例小说：优先复用预置向量库，或一键完整构建",
    )
    p_demo.add_argument(
        "--force", action="store_true",
        help="删除现有 demo 项目与数据后重建（如升级为真实语义向量库）",
    )
    p_demo.add_argument(
        "--no-embedding", action="store_true",
        help="强制纯关键词模式，不触发嵌入模型下载（离线冒烟）",
    )
    p_demo.set_defaults(func=cmd_demo)

    return parser


if __name__ == "__main__":
    _enable_utf8_stdout()
    args = build_parser().parse_args()
    try:
        args.func(args)
    except APIClientError as e:
        # API 调用失败（网络 / 鉴权 / 限流重试耗尽）：友好提示而非堆栈崩溃
        print(f"✗ API 调用失败: {e.message}")
        print("  请检查网络连接、config.json 中的 api_url/api_key，或稍后重试。")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n已取消")
        sys.exit(130)
