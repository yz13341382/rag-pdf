# -*- coding: utf-8 -*-
"""
rag.py —— 这个项目的心脏，所有 RAG 逻辑都在这里。

为什么单独放一个文件？
    因为 RAG 一共就 8 个动作，读一遍这个文件你就全懂了。
    后面 build_index.py / search.py / ask.py / app.py 都只是"调用这里的函数"，
    它们本身几乎没有逻辑。所以：**这个文件是教材，其它文件是遥控器。**

八个动作，按 RAG 流程排好：

    阶段一（离线，只跑一次）
      1. extract_pages()   读 PDF，一页页抽出文字
      2. clean_page()      洗掉页眉页脚（不洗会污染检索结果）
      3. split_into_chunks() 把正文切成一条条 chunk  ← 本项目最关键的一步
      4. embed_texts()     把每条文字变成一串数字（向量），调 API
      5. build_index()     把 chunk + 向量存到磁盘

    阶段二（在线，每次提问都跑）
      6. load_index()      从磁盘读回 chunk 和向量
      7. search()          给一个问题，找出最相关的几条
      8. answer()          问题 + 相关条目一起交给大模型，生成回答

名词对照：
    chunk   = 文本块，检索的最小单位。本项目一个 chunk = 一条建议
    embed   = 向量化，把文字变成一串浮点数，语义相近的数字也相近
    index   = 索引，就是"所有 chunk 的向量"整体，存成文件随时读
    top_k   = 每次检索取回几条，本项目默认 5
"""

from __future__ import annotations

import json
import math
import os
import pickle
import re
import time
from pathlib import Path

import numpy as np

# ============================================================
# 0. 路径与配置
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"      # 文本块（一行一个 JSON）
VECTORS_FILE = DATA_DIR / "vectors.npy"      # 向量矩阵（每行对应一个 chunk）
BM25_FILE = DATA_DIR / "bm25.pkl"            # 关键词索引（纯本地，不需要任何密钥）


def load_config() -> dict:
    """
    从 .env 读配置。没有 .env 就给你一个清晰的报错，而不是神秘崩溃。
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    cfg = {
        "pdf_path": os.getenv("PDF_PATH", r"C:\Users\Sun\Desktop\HowToLiveBetter.pdf"),
        # 向量化用的服务（把文字变成数字）
        "embed_api_key": os.getenv("EMBEDDING_API_KEY", ""),
        "embed_base_url": os.getenv("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1"),
        "embed_model": os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
        # 生成答案用的服务（把资料写成回答）
        "llm_api_key": os.getenv("LLM_API_KEY", ""),
        "llm_base_url": os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
        "llm_model": os.getenv("LLM_MODEL", "deepseek-chat"),
        # 检索方式：
        #   hybrid —— 向量 + 关键词 两路召回后融合（推荐，默认）
        #   vector —— 只用语义向量，需要 embedding 密钥
        #   bm25   —— 只用关键词，纯本地，**不需要任何密钥**
        "retrieval_mode": os.getenv("RETRIEVAL_MODE", "hybrid").strip().lower(),
    }
    return cfg


def require_key(value: str, name: str) -> str:
    """检查密钥是否填了，没填就给一句人话提示。"""
    if not value or "你的" in value or value.strip() in ("", "sk-"):
        raise SystemExit(
            f"\n[缺少配置] {name} 还没填。\n"
            f"请打开项目里的 .env 文件，把 {name} 换成真实的密钥。\n"
            f"怎么申请：看 README.md 的「第三步：拿密钥」。\n"
        )
    return value


# ============================================================
# 阶段一 · 1. 读 PDF
# ============================================================

def extract_pages(pdf_path: str | Path) -> list[str]:
    """
    把 PDF 一页页读成纯文本，返回一个列表：['第1页文字', '第2页文字', ...]

    用的库是 pypdf。注意：**它只能读"文字版 PDF"**。
    如果你的 PDF 是扫描件（整页是图片），抽出来会是空的，
    那种得先上 OCR（比如 RapidOCR），本项目不做。
    """
    from pypdf import PdfReader

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise SystemExit(f"\n[找不到文件] {pdf_path}\n请检查 .env 里的 PDF_PATH 是否正确。\n")

    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return pages


# ============================================================
# 阶段一 · 2. 洗掉页眉页脚
# ============================================================

# 页脚长这样：  12 / 392
PAGE_FOOTER_RE = re.compile(r"^\s*\d+\s*/\s*\d+\s*$")
# 页眉长这样：  高性价比人生指南 1. 不要早死
PAGE_HEADER_RE = re.compile(r"^\s*高性价比人生指南(\s.*)?$")


def clean_page(text: str) -> str:
    """
    洗掉每页的页眉和页脚。

    为什么必须洗？因为一条建议常常跨页，
    页眉页脚会**硬插在条目的中间**，变成这样：

        • 收益：中国疾控中心 2025 年调查了 828 起蘑菇中毒事件……
        • 证据等级：A
        12 / 392                          ← 页脚
        高性价比人生指南 1. 不要早死        ← 页眉
        • 来源：Mushroom Poisoning Outbreaks……

    如果不洗，这些噪音会被一起送进大模型，也会污染向量。
    """
    keep = []
    for line in text.split("\n"):
        if PAGE_FOOTER_RE.match(line):
            continue
        if PAGE_HEADER_RE.match(line):
            continue
        keep.append(line)
    return "\n".join(keep)


# ============================================================
# 阶段一 · 3. 切分 —— 本项目最关键的一步
# ============================================================

# 条目的第一个字段，也是整个文档最可靠的锚点
COST_RE = re.compile(r"^\s*•\s*成本[：:]")
# 任意字段行
FIELD_RE = re.compile(r"^\s*•\s*(成本|说人话|收益|证据等级|来源|备注)[：:]")
# 条目标题开头：  1. 系安全带，前排后排都系
ITEM_START_RE = re.compile(r"^(\d{1,3})\.\s*")
# 章节标题：      1. 不要早死（编号 1~34，编号后**必须有空格**，名字不以数字开头）
#
# 这里两个约束都是被真实数据逼出来的：
#   要求「\s+」 —— 因为正文里的 DOI 写成 10.1056/NEJMoa1408544、
#                   统计值写成 1.27（1.18 到 1.37），编号后都没有空格，会被误判成章节。
#   要求名字不以数字开头 —— 再挡一层 1.7 公斤 这类数值。
CHAPTER_TITLE_RE = re.compile(r"^(\d{1,2})\.\s+([^\d\s].{1,28})$")

# 单条建议的正常长度上限。实测最长的一条 4058 字符，
# 超过这个数说明这个 chunk 吞进了别的东西（通常是文档尾部的附录）。
MAX_REASONABLE_CHARS = 6000
_END_PUNCT = "。！？）」』"


def _trim_after_last_field(block: list[str]) -> list[str]:
    """
    兜底保护：把 chunk 尾部"不属于这条建议"的内容切掉。

    为什么需要它？
        文档最后一条建议的后面，直接跟着 docs/ 目录里的附录长文（约 900 行）。
        如果不管，最后一条 chunk 会膨胀到 3 万多字符，
        检索时它会把附录里的无关内容一起喂给大模型。

    怎么判断哪里是结尾？
        找到最后一个字段行（• 备注 或 • 来源），从它往后扫描，
        遇到第一个以句末标点结尾的行就停 —— 那是这条建议最后一个完整句子。
        中文 PDF 里"续行"的行尾基本不带标点，所以这个判据在实际数据上很稳。

    注意：这个函数只在 chunk 异常长的时候才被调用，
        正常条目（最长 4058 字符）根本走不到这里，所以不会误伤。
    """
    last_field = -1
    for k, ln in enumerate(block):
        if FIELD_RE.match(ln):
            last_field = k
    if last_field < 0:
        return block

    for k in range(last_field + 1, len(block)):
        s = block[k].rstrip()
        if s and s[-1] in _END_PUNCT:
            return block[:k + 1]
    return block


def _find_title_start(lines: list[str], anchor: int, max_back: int = 5) -> int | None:
    """
    从「• 成本：」这一行往上找条目的标题行。

    为什么不能简单"看上面一行"？
        因为 PDF 排版会把长标题折成两行，抽出来就是：

            39. 65 岁以上的女性去做一次双能 X 线骨密度检查，绝经后有骨质疏松危险因素
            的不用等到 65 岁          ← 这是标题的续行，不是正文
            • 成本：去医院做一次双能 X 线骨密度检查……

        只看上一行，就会把续行当成"不是标题"而漏掉整条。

    所以正确做法是：从锚点往上回溯，最多看 5 行，
    遇到以「数字.」开头的行就是标题的第一行。
    遇到「•」开头的行说明已经越到上一条去了，立刻停手。
    """
    for k in range(anchor - 1, max(-1, anchor - 1 - max_back), -1):
        s = lines[k].strip()
        if not s:
            continue
        if s.startswith("•"):
            return None                      # 越界到上一条，这条没有标题
        if ITEM_START_RE.match(s):
            return k                         # 找到了标题第一行
    return None


def split_into_chunks(pages: list[str]) -> list[dict]:
    """
    把全文切成 chunk。这里是整个项目**最有技术含量**的地方，值得慢读。

    为什么不按"每 500 字一刀"切？
        因为硬切会把一句话拦腰砍断，检索出来的东西是残句。
        更糟的是：一条建议的成本、收益、来源会被切到不同 chunk 里，
        模型只看到「证据等级：A」却不知道说的是哪条。

    这份文档的天然边界非常清楚 ——
        「• 成本：」是每条建议固定出现的第一个字段。
    于是算法变成三步（这个思路比代码更值得记）：

        1. 找出所有「• 成本：」行       ← 654 个，就是 654 条的分界
        2. 从每个锚点往前回溯，找到「数字.」开头的行 = 标题起点
        3. 一条 chunk = 本次标题起点 → 下一条标题起点之前

    通用规则（记住这个，比记住代码有用）：
        切分要顺着文档**自身的结构**走 ——
        编号、标题、空行、Markdown 的 ##，都是作者免费给你的边界。
        实在找不到结构，才退回去按字数切 + 加重叠。
    """
    lines = "\n".join(clean_page(p) for p in pages).split("\n")

    # --- 第 1 步：找锚点 ---
    anchors = [i for i, ln in enumerate(lines) if COST_RE.match(ln)]

    # --- 第 2 步：找到每条建议的起点 ---
    starts: list[int] = []
    for a in anchors:
        s = _find_title_start(lines, a)
        starts.append(s if s is not None else a)

    # --- 第 3 步：按起点切片 ---

    # 第一条建议出现之前（前言、各节简介区）也藏着章节标题，
    # 取最后一个当作起始章节，否则前 20 多条会被错误地标成「前言」。
    chapter = "前言"
    for k in range(0, starts[0]):
        m = CHAPTER_TITLE_RE.match(lines[k].strip())
        if m and 1 <= int(m.group(1)) <= 34:
            chapter = lines[k].strip()

    out: list[dict] = []

    for idx, start in enumerate(starts):
        end = starts[idx + 1] - 1 if idx + 1 < len(starts) else len(lines) - 1
        block = [lines[k].strip() for k in range(start, end + 1)]

        # 章节标题夹在上一条的结尾和下一条之间，中间还跟着一段章节引言，
        # 所以要**从中间截断**，而不是只看末尾。
        # 判断依据：行首「数字.」+ 短行 + 编号 ≤ 34（正文里的引用都写成
        # 「见第 9 节第 13 条」，不会顶格以编号开头，所以不会误伤）
        next_chapter = None
        for k, ln in enumerate(block):
            if k == 0:
                continue                      # 第 0 行是条目标题，跳过
            m = CHAPTER_TITLE_RE.match(ln)
            if m and 1 <= int(m.group(1)) <= 34:
                next_chapter = ln
                block = block[:k]             # 后面的章节引言不属于这一条
                break

        # 去掉尾部空行
        while block and not block[-1]:
            block.pop()

        # 异常长度保护（正常条目不会触发）
        if sum(len(x) for x in block) > MAX_REASONABLE_CHARS:
            block = _trim_after_last_field(block)
            while block and not block[-1]:
                block.pop()

        if not block:
            chapter = next_chapter or chapter
            continue

        text = "\n".join(block).strip()

        # 标题 = 开头的连续非「•」行（可能是折了两行的长标题）
        title_lines = []
        for ln in block:
            if ln.startswith("•"):
                break
            if ln:
                title_lines.append(ln)
        title_raw = " ".join(title_lines)
        tm = ITEM_START_RE.match(title_raw)
        num = int(tm.group(1)) if tm else 0
        title = title_raw[tm.end():].strip() if tm else title_raw

        # 证据等级抽到元数据里，方便以后按等级筛选
        lv = re.search(r"证据等级[：:]\s*([ABCD])", text)

        out.append({
            "id": len(out),
            "num": num,
            "title": title,
            "chapter": chapter,
            "evidence": lv.group(1) if lv else "",
            "text": text,
        })

        if next_chapter:
            chapter = next_chapter

    return out


def make_embed_text(chunk: dict) -> str:
    """
    决定"拿哪段文字去做向量"。

    注意这是个容易吃亏的细节：只拿正文去算，
    标题里的关键词（比如"系安全带"）就浪费了。
    把章节名和标题也拼进去，检索命中率会明显变好。
    """
    return f"{chunk['chapter']} | {chunk['title']}\n{chunk['text']}"


# ============================================================
# 阶段一 · 4. 向量化（调用 API）
# ============================================================

def _embed_client(cfg: dict):
    from openai import OpenAI
    return OpenAI(
        api_key=require_key(cfg["embed_api_key"], "EMBEDDING_API_KEY"),
        base_url=cfg["embed_base_url"],
    )


def embed_texts(texts: list[str], cfg: dict | None = None,
                batch_size: int = 16, verbose: bool = True) -> np.ndarray:
    """
    把一批文字变成向量矩阵，形状 (条数, 维度)。

    三件必须知道的事：

    1) 要分批。API 一次只收几十条，654 条得分 41 批发。
    2) 单条太长要截断。bge-m3 上限 8192 token，我们截到 4000 字符足够。
    3) **必须做 L2 归一化**。
       归一化之后，向量的长度都变成 1，
       于是"余弦相似度"就等于"点积"，检索时一个矩阵乘法就搞定，
       而且分数天然落在 -1 ~ 1 之间，好判断。
    """
    cfg = cfg or load_config()
    client = _embed_client(cfg)

    vectors: list[list[float]] = []
    total = len(texts)

    for start in range(0, total, batch_size):
        batch = [t[:4000] for t in texts[start:start + batch_size]]

        last_err = None
        for attempt in range(3):
            try:
                resp = client.embeddings.create(model=cfg["embed_model"], input=batch)
                break
            except Exception as e:
                last_err = e
                if attempt == 2:
                    raise SystemExit(
                        f"\n[向量化失败] 第 {start}~{start+len(batch)} 条出错：{e}\n"
                        f"常见原因：密钥错了 / 余额没了 / 模型名写错 / 网络不通。\n"
                    )
                time.sleep(2 * (attempt + 1))
        else:
            raise SystemExit(f"[向量化失败] {last_err}")

        # API 返回的顺序不一定和输入一致，按 index 排回去
        for d in sorted(resp.data, key=lambda d: d.index):
            vectors.append(d.embedding)

        if verbose:
            done = min(start + batch_size, total)
            print(f"  向量化进度 {done}/{total}", end="\r", flush=True)

    if verbose:
        print(" " * 40, end="\r")

    arr = np.asarray(vectors, dtype=np.float32)

    # L2 归一化：每个向量除以自己的长度
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.clip(norms, 1e-10, None)
    return arr


# ============================================================
# 阶段一 · 5. 存盘
# ============================================================

def save_chunks(chunks: list[dict]) -> None:
    """把文本块存盘（一行一个 JSON，方便直接用文本工具查看）。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def save_index(chunks: list[dict], vectors: np.ndarray) -> None:
    """文本 + 向量 一起存盘。"""
    save_chunks(chunks)
    np.save(VECTORS_FILE, vectors)


def build_index(cfg: dict | None = None, embed: bool = True,
                bm25: bool = True) -> tuple[list[dict], np.ndarray | None]:
    """
    一键建索引：读 PDF → 洗 → 切 → 存文本 → 建关键词索引 → 向量化。

    两个开关：
        embed=False  —— 不调 embedding 接口（--dry-run，省钱）
        bm25=False   —— 不建关键词索引

    注意顺序：**关键词索引排在向量化前面**，
    因为它不需要任何密钥、几秒钟就建好。
    这样即使你还没配密钥，也能立刻拿它跑通检索。
    """
    cfg = cfg or load_config()

    print("\n[1/4] 读 PDF …")
    pages = extract_pages(cfg["pdf_path"])
    print(f"      共 {len(pages)} 页")

    print("[2/4] 清洗 + 切分 …")
    chunks = split_into_chunks(pages)
    print(f"      得到 {len(chunks)} 条 chunk")

    if not embed and not bm25:
        return chunks, None

    print("[3/4] 存盘 …")
    save_chunks(chunks)
    print(f"      {CHUNKS_FILE.name}")

    if bm25:
        save_bm25(build_bm25(chunks))
        print(f"      {BM25_FILE.name}   ← 关键词索引，纯本地，不需要密钥")

    if not embed:
        return chunks, None

    print(f"[4/4] 调用 embedding 模型向量化（{cfg['embed_model']}）…")
    vectors = embed_texts([make_embed_text(c) for c in chunks], cfg=cfg)
    print(f"      向量维度 {vectors.shape[1]}，共 {vectors.shape[0]} 条")
    np.save(VECTORS_FILE, vectors)
    print(f"      {VECTORS_FILE.name}")

    return chunks, vectors


# ============================================================
# 阶段二 · 6. 读回索引
# ============================================================

_INDEX_CACHE: dict = {}


def load_chunks() -> list[dict]:
    """
    只读文本块。

    为什么把 load_index 拆开？
        因为 BM25（关键词检索）只需要文本，不需要向量。
        拆开之后，"没填密钥"也能先把检索这条路跑通。
    """
    if "chunks" not in _INDEX_CACHE:
        if not CHUNKS_FILE.exists():
            raise SystemExit(
                "\n[还没有索引] 请先建索引：\n"
                "    .venv\\Scripts\\python.exe build_index.py --bm25-only   ← 不需要密钥\n"
                "    .venv\\Scripts\\python.exe build_index.py               ← 建完整索引\n"
            )
        with open(CHUNKS_FILE, encoding="utf-8") as f:
            _INDEX_CACHE["chunks"] = [json.loads(line) for line in f if line.strip()]
    return _INDEX_CACHE["chunks"]


def load_vectors() -> np.ndarray:
    """读回向量矩阵，并检查它和文本块的数量对得上。"""
    if "vectors" not in _INDEX_CACHE:
        if not VECTORS_FILE.exists():
            raise SystemExit(
                "\n[还没有向量索引] 需要先做向量化（要有 embedding 密钥）：\n"
                "    .venv\\Scripts\\python.exe build_index.py\n"
                "如果暂时没有密钥，可以先用关键词检索：\n"
                "    .venv\\Scripts\\python.exe search.py \"你的问题\" --mode=bm25\n"
            )
        vectors = np.load(VECTORS_FILE)
        chunks = load_chunks()
        if len(chunks) != vectors.shape[0]:
            raise SystemExit(
                f"[索引不匹配] {len(chunks)} 条文本 对不上 {vectors.shape[0]} 条向量。\n"
                "多半是换了切分方式但没重新建索引，重跑 build_index.py 即可。\n"
            )
        _INDEX_CACHE["vectors"] = vectors
    return _INDEX_CACHE["vectors"]


def load_index() -> tuple[list[dict], np.ndarray]:
    """从磁盘读回 chunk 和向量（旧接口，保留兼容）。"""
    return load_chunks(), load_vectors()


# ============================================================
# 阶段二 · 7. 检索
#
#   这里有三种检索方式，不是炫技，是因为"哪种更好"取决于你问什么：
#
#     vector  语义检索。问"怎么不被骗"能找到讲"诈骗"的段落，
#             即使两者一个字都不重叠。但对精确词不灵 ——
#             问"头孢"不一定能排在最前面。
#     bm25    关键词检索。经典算法，算的是词频和逆文档频率。
#             对"头孢""0x0000007B"这种精确词极准，但不懂近义词。
#     hybrid  两路都跑，再融合。这才是生产环境的做法。
#
#   bm25 还有个额外好处：**纯本地，不需要任何密钥、不花一分钱**。
#   所以没配密钥的时候，可以直接拿它先把整条链路跑通。
# ============================================================

# ------------------------------------------------------------ 7a. 极简中文分词

_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    """
    中文分词，用的是"单字 + 相邻两字 bigram"。

    为什么不引入 jieba 之类的分词库？
        中文里"蘑菇中毒"用 bigram 切出来是：蘑 / 菇 / 中 / 毒 / 蘑菇 / 菇中 / 中毒。
        词的边界信息其实被保留下来了，而且不需要加载词典（省几十兆内存和启动时间）。
        对 654 条这个规模，效果完全够用。

    英文和数字直接整块当一个 token —— 这样 "0x0000007B" 这种代码不会被切碎，
    对检索报错信息、型号这类内容很重要。
    """
    tokens: list[str] = []
    for seg in _TOKEN_RE.findall(text.lower()):
        if "\u4e00" <= seg[0] <= "\u9fff":
            tokens.extend(seg)                                     # 单字
            tokens.extend(seg[i:i + 2] for i in range(len(seg) - 1))  # bigram
        else:
            tokens.append(seg)
    return tokens


# ------------------------------------------------------------ 7b. BM25

class BM25:
    """
    BM25：上世纪 70 年代提出，到今天仍是关键词检索的最强基线。

    一句话解释它比"数一数出现几次"聪明在哪：
        一个词在文档里出现越多，这篇越相关（词频 TF）；
        但一个词在所有文档里都出现（比如"的""我们"），它就没什么区分度（逆文档频率 IDF）；
        还要防止长文档单纯因为字多而占便宜（长度归一化）。
    这三件事乘起来，就是 BM25。

    参数 k1 控制词频饱和（出现 10 次不该比出现 5 次相关两倍），
    b 控制长度归一化的强度。1.5 / 0.75 是经典默认值。
    """

    def __init__(self, tokenized_docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        from collections import Counter

        self.k1 = k1
        self.b = b
        self.n = len(tokenized_docs)
        self.doc_len = [len(d) for d in tokenized_docs]
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 1.0

        self.tfs = [Counter(d) for d in tokenized_docs]

        df = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        # 加 0.5 平滑，避免 N 很小时除零
        self.idf = {
            t: math.log(1 + (self.n - c + 0.5) / (c + 0.5))
            for t, c in df.items()
        }

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        for t in set(tokenize(query)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in enumerate(self.tfs):
                f = tf.get(t)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                out[i] += idf * f * (self.k1 + 1) / denom
        return out


def build_bm25(chunks: list[dict]) -> BM25:
    return BM25([tokenize(make_embed_text(c)) for c in chunks])


def save_bm25(index: BM25) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(BM25_FILE, "wb") as f:
        pickle.dump(index, f)


def load_bm25() -> BM25:
    if "bm25" not in _INDEX_CACHE:
        if not BM25_FILE.exists():
            raise SystemExit(
                "\n[还没有关键词索引] 先建一次（不需要密钥）：\n"
                "    .venv\\Scripts\\python.exe build_index.py --bm25-only\n"
            )
        with open(BM25_FILE, "rb") as f:
            _INDEX_CACHE["bm25"] = pickle.load(f)
    return _INDEX_CACHE["bm25"]


# ------------------------------------------------------------ 7c. 融合两路结果

def rrf(rank_lists: list[list[int]], weights: list[float] | None = None,
        k: int = 60) -> list[tuple[int, float]]:
    """
    RRF（Reciprocal Rank Fusion）：把多路召回结果合成一个排名。

    为什么不能把两边的分数直接相加？
        因为量纲完全不同 —— 向量相似度是 0~1 的小数，
        BM25 分数可能是 0~30 的大数。直接相加，BM25 会把结果完全支配。

    RRF 的做法很聪明：**只看"排第几名"，不看具体多少分**。
        融合分 = Σ 1 / (60 + 排名)
    排名第一贡献 1/61，第二贡献 1/62……天然不受量纲影响。
    两路都排在前面的条目，融合分会明显更高。

    这就是工业界混合检索的标准做法，面试可以展开讲。
    """
    weights = weights or [1.0] * len(rank_lists)
    fused: dict[int, float] = {}
    for ranks, w in zip(rank_lists, weights):
        for rank, idx in enumerate(ranks):
            fused[idx] = fused.get(idx, 0.0) + w / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: -x[1])


# ------------------------------------------------------------ 7d. 统一的检索入口

_MODE_HINTED: set[str] = set()


def resolve_mode(mode: str | None = None, cfg: dict | None = None) -> str:
    """
    算出这次到底会用哪种检索方式，包含"少一个索引时自动降级"。

    单独抽出来，是因为界面层也要知道实际用的模式 ——
    否则会闹出"实际跑的是关键词检索，标签却写着融合分"这种笑话。
    """
    cfg = cfg or load_config()
    mode = (mode or cfg["retrieval_mode"]).lower()
    has_vec = VECTORS_FILE.exists()
    has_bm = BM25_FILE.exists()

    if mode == "vector" and not has_vec:
        raise SystemExit(
            "\n[没有向量索引] 要么先跑 build_index.py，"
            "要么改用关键词检索：--mode=bm25\n"
        )
    if mode == "bm25" and not has_bm:
        raise SystemExit(
            "\n[没有关键词索引] 先跑一次（不需要密钥、很快）：\n"
            "    .venv\\Scripts\\python.exe build_index.py --bm25-only\n"
        )

    if mode == "hybrid" and not (has_vec and has_bm):
        fallback = "bm25" if has_bm else ("vector" if has_vec else None)
        if fallback is None:
            raise SystemExit(
                "\n[还没有任何索引] 先跑：\n"
                "    .venv\\Scripts\\python.exe build_index.py --bm25-only\n"
            )
        if fallback not in _MODE_HINTED:
            _MODE_HINTED.add(fallback)
            print(f"  [提示] 混合检索需要向量和关键词两种索引，"
                  f"现在只有一种，本次自动改用 {fallback}")
        mode = fallback

    return mode


def search(question: str, top_k: int = 5, cfg: dict | None = None,
           mode: str | None = None) -> list[tuple[dict, float]]:
    """
    给一个问题，返回最相关的 top_k 条，带分数。

    mode 可以是 vector / bm25 / hybrid。
    默认走 .env 里的 RETRIEVAL_MODE，缺索引时自动降级到能跑的那一种。
    """
    cfg = cfg or load_config()
    mode = resolve_mode(mode, cfg)
    chunks = load_chunks()
    k = min(top_k, len(chunks))

    # ---- 单路：向量 ----
    if mode == "vector":
        vectors = load_vectors()
        qvec = embed_texts([question], cfg=cfg, verbose=False)[0]
        scores = vectors @ qvec
        order = np.argsort(-scores)[:k]
        return [(chunks[i], float(scores[i])) for i in order]

    # ---- 单路：关键词 ----
    if mode == "bm25":
        scores = load_bm25().scores(question)
        order = np.argsort(-scores)[:k]
        return [(chunks[i], float(scores[i])) for i in order]

    # ---- 混合：两路召回 + RRF 融合 ----
    pool = min(max(top_k * 4, 20), len(chunks))     # 每路先多召回一些

    qvec = embed_texts([question], cfg=cfg, verbose=False)[0]
    v_scores = load_vectors() @ qvec
    b_scores = load_bm25().scores(question)

    v_rank = list(np.argsort(-v_scores)[:pool])
    b_rank = list(np.argsort(-b_scores)[:pool])

    fused = rrf([v_rank, b_rank])[:k]
    return [(chunks[i], float(s)) for i, s in fused]


def search_detail(question: str, top_k: int = 5, cfg: dict | None = None,
                  mode: str | None = None) -> tuple[list[tuple[dict, float]], dict]:
    """
    跟 search 一样，但额外返回"每一路各自的前 5 名"。

    做混合检索必须能看到这个 —— 否则你无法判断
    "是向量这一路没召回到，还是融合环节把它排下去了"。
    """
    cfg = cfg or load_config()
    hits = search(question, top_k=top_k, cfg=cfg, mode=mode)
    chunks = load_chunks()

    def _top5(scores: np.ndarray) -> list[tuple[int, float]]:
        return [(int(i), float(scores[i])) for i in np.argsort(-scores)[:5]]

    detail: dict = {"chunks": chunks}
    if VECTORS_FILE.exists() and cfg.get("embed_api_key"):
        try:
            qvec = embed_texts([question], cfg=cfg, verbose=False)[0]
            detail["vector_top"] = _top5(load_vectors() @ qvec)
        except SystemExit:
            pass
    if BM25_FILE.exists():
        detail["bm25_top"] = _top5(load_bm25().scores(question))
    return hits, detail


# ============================================================
# 阶段二 · 8. 生成回答
# ============================================================

SYSTEM_PROMPT = """你是一个严谨的资料助手，负责根据给定资料回答用户问题。

规则：
1. 只依据「参考资料」回答，不要使用你自己的知识补充。
2. 资料里没有的内容，直接说「资料里没有提到」，不要猜。
3. 回答时标注引用了第几条，例如「（第 12 条）」。
4. 每条建议都带「证据等级」：A 是证据最硬，B 次之，C 最弱。
   如果引用的条目有等级，顺便提一句，帮用户判断可信度。
5. 用中文、说人话，不要堆砌术语。"""


def build_context(hits: list[tuple[dict, float]]) -> str:
    """把检索到的几条拼成一段文字，塞进提示词。"""
    parts = []
    for chunk, score in hits:
        parts.append(
            f"[第 {chunk['num']} 条 · 所属章节：{chunk['chapter']} · 证据等级：{chunk['evidence'] or '无'}]\n"
            f"{chunk['text']}"
        )
    return "\n\n" + ("\n" + "=" * 60 + "\n").join(parts)


def answer(question: str, top_k: int = 5, cfg: dict | None = None,
           stream: bool = False, mode: str | None = None):
    """
    完整的 RAG 问答：检索 → 拼提示词 → 交给大模型 → 返回答案 + 引用来源。

    提示词的结构就是这个项目的灵魂，值得背下来：

        你是……（人设）
        【参考资料】
        <检索到的内容>
        【用户问题】
        <问题>

    模型只在"参考资料"这个范围内作答，幻觉就被压住了。
    """
    from openai import OpenAI

    cfg = cfg or load_config()
    hits = search(question, top_k=top_k, cfg=cfg, mode=mode)

    user_prompt = (
        f"【参考资料】{build_context(hits)}\n\n"
        f"【用户问题】{question}"
    )

    client = OpenAI(
        api_key=require_key(cfg["llm_api_key"], "LLM_API_KEY"),
        base_url=cfg["llm_base_url"],
    )

    if stream:
        def gen():
            resp = client.chat.completions.create(
                model=cfg["llm_model"],
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                stream=True,
            )
            for piece in resp:
                delta = piece.choices[0].delta.content
                if delta:
                    yield delta
        return gen(), hits

    resp = client.chat.completions.create(
        model=cfg["llm_model"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content, hits
