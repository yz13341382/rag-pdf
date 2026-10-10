# -*- coding: utf-8 -*-
"""
第三步：完整问答。检索 + 大模型生成，也就是真正意义上的 RAG。

跟 search.py 的区别只有一点：
    search.py 找出资料就停下，让你自己看；
    ask.py   把资料塞进提示词，让大模型读完之后写人话回答。

用法：
    .venv\\Scripts\\python.exe ask.py "怎么预防蘑菇中毒"
    .venv\\Scripts\\python.exe ask.py              不加问题就进交互模式
"""

import sys

import rag


def run_once(question: str, top_k: int, mode: str | None = None) -> None:
    print("\n" + "=" * 70)
    print(f"  问题：{question}")
    print("=" * 70)

    print("\n【检索到的资料】")
    stream, hits = rag.answer(question, top_k=top_k, stream=True, mode=mode)
    for i, (chunk, score) in enumerate(hits, 1):
        print(f"  {i}. 第 {chunk['num']} 条 · {chunk['chapter']} "
              f"· 证据 {chunk['evidence'] or '无'} · 相似度 {score:.3f}")
        print(f"     {chunk['title'][:60]}")

    print("\n【回答】")
    for piece in stream:
        print(piece, end="", flush=True)
    print("\n")


def main() -> None:
    top_k = 5
    mode = None
    questions = []

    for a in sys.argv[1:]:
        if a.startswith("--top="):
            top_k = int(a.split("=", 1)[1])
        elif a.startswith("--mode="):
            mode = a.split("=", 1)[1].lower()
        else:
            questions.append(a)

    if questions:
        run_once(" ".join(questions), top_k, mode)
        return

    print("=" * 70)
    print("  文档问答（按 Ctrl+C 退出）")
    print("  这份文档有 654 条建议，覆盖健康、法律、金钱、租房、育儿等。")
    print("=" * 70)

    while True:
        try:
            q = input("\n问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if not q:
            continue
        run_once(q, top_k, mode)


if __name__ == "__main__":
    main()
