# -*- coding: utf-8 -*-
"""
第二步：只做检索，不调用大模型。

**这一步比第三步重要。**
RAG 效果差，九成不是大模型不行，而是先检索错了 ——
找到的资料是错的，模型再聪明也只能基于错的资料编。

所以先把这一关检验好：随便问几个问题，看挑出来的这几条对不对。

用法：
    .venv\\Scripts\\python.exe search.py "怎么预防蘑菇中毒"
    .venv\\Scripts\\python.exe search.py "头孢 喝酒" --mode=bm25
    .venv\\Scripts\\python.exe search.py "怎么防止被骗" --mode=vector
    .venv\\Scripts\\python.exe search.py                   进交互模式

三种模式的区别：
    bm25    关键词匹配，**不需要任何密钥**，没配 key 时也能跑
    vector  语义匹配，需要 embedding 密钥
    hybrid  两路都跑再融合（默认，效果最好）
"""

import sys

import rag

MODE_LABEL = {
    "vector": "相似度",
    "bm25": "BM25 分",
    "hybrid": "融合分",
}


def show_hit(rank: int, chunk: dict, score: float,
             max_score: float, label: str) -> None:
    # 不同模式的分数不是一个量纲（余弦相似度 0~1、BM25 可以到 30），
    # 所以进度条按"相对第一名的比例"画，这样哪种模式都直观
    ratio = (score / max_score) if max_score else 0
    bar = "█" * max(1, int(ratio * 20))

    print(f"\n┌─ 第 {rank} 名   {label} {score:.4f}  {bar}")
    print(f"│ 第 {chunk['num']} 条 · {chunk['chapter']} · 证据等级 {chunk['evidence'] or '无'}")
    print(f"│ {chunk['title']}")
    print("├" + "─" * 68)
    text = chunk["text"]
    print("\n".join("│ " + ln for ln in text[:420].split("\n")))
    if len(text) > 420:
        print(f"│ ……（本条共 {len(text)} 字符）")
    print("└" + "─" * 68)


def show_channels(detail: dict) -> None:
    """把两路召回各自的排名摊开 —— 看懂这个就懂了混合检索。"""
    if "vector_top" not in detail and "bm25_top" not in detail:
        return

    chunks = detail["chunks"]
    print("\n" + "─" * 70)
    print("  两路召回各自的前 5 名")
    print("─" * 70)
    for key, name in [("vector_top", "向量（懂语义）"), ("bm25_top", "BM25（认字面）")]:
        if key not in detail:
            continue
        print(f"\n  【{name}】")
        for rank, (idx, sc) in enumerate(detail[key], 1):
            c = chunks[idx]
            print(f"    {rank}. {sc:8.4f}  第 {c['num']:>3} 条  {c['title'][:42]}")


def run_once(question: str, top_k: int, mode: str | None) -> None:
    hits, detail = rag.search_detail(question, top_k=top_k, mode=mode)
    chunks = detail["chunks"]

    print("\n" + "=" * 70)
    print(f"  问题：{question}")
    print(f"  从 {len(chunks)} 条里取回最相关的 {len(hits)} 条")
    print("=" * 70)

    if not hits:
        print("  没找到任何结果。")
        return

    # 用实际生效的模式来标注分数（可能已经从 hybrid 降级成 bm25 了）
    mode_used = rag.resolve_mode(mode)
    label = MODE_LABEL.get(mode_used, "分数")
    max_score = hits[0][1]

    for i, (chunk, score) in enumerate(hits, 1):
        show_hit(i, chunk, score, max_score, label)

    show_channels(detail)

    print("\n怎么判断好不好：")
    print("  · 都答得上问题 → 检索没问题，可以去 ask.py 让大模型写答案了")
    print("  · 相关的没被找回来 → 调 chunk 的切法，或者换 embedding 模型")
    print("  · 混进了明显不相关的 → 把 top_k 调小，或者加一层重排序")


def main() -> None:
    top_k = 5
    mode = None
    questions = []

    for a in sys.argv[1:]:
        if a.startswith("--top="):
            top_k = int(a.split("=", 1)[1])
        elif a.startswith("--mode="):
            mode = a.split("=", 1)[1].lower()
            if mode not in ("bm25", "vector", "hybrid"):
                raise SystemExit("--mode 只能是 bm25 / vector / hybrid")
        else:
            questions.append(a)

    if questions:
        run_once(" ".join(questions), top_k, mode)
        return

    print("=" * 70)
    print("  检索测试台（按 Ctrl+C 退出）")
    print("=" * 70)
    print("  试试这些问题：")
    print("    · 怎么预防蘑菇中毒")
    print("    · 被公司裁员了能拿多少赔偿")
    print("    · 孩子发烧什么时候必须去医院")
    print("    · 租房押金被扣怎么办")
    print()
    print("  加 --mode=bm25 可以对比一下关键词检索的结果。")
    print()

    while True:
        try:
            q = input("问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if not q:
            continue
        run_once(q, top_k, mode)


if __name__ == "__main__":
    main()
