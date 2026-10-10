# -*- coding: utf-8 -*-
"""
第一步：建索引。把 PDF 变成一堆能被检索的向量。

这是整个 RAG 里唯一"重"的一步 —— 要读 395 页、切 654 条、调 41 次 embedding 接口。
跑完之后结果存在 data/ 目录里，以后每次提问都直接读它，不用再跑。

用法：
    .venv\\Scripts\\python.exe build_index.py             完整跑一遍
    .venv\\Scripts\\python.exe build_index.py --dry-run   只看切分结果，不花钱

    --dry-run 是个好习惯：先确认"切得对不对"，再去花额度做向量化。
    因为 RAG 效果差，八成是切分和检索的问题，不是大模型的问题。
"""

import argparse
import sys
from collections import Counter

import rag


def print_stats(chunks: list[dict]) -> None:
    lens = [len(c["text"]) for c in chunks]
    lens_sorted = sorted(lens)

    print(f"\n  条目数      {len(chunks)}")
    print(f"  总字数      {sum(lens):,}")
    print(f"  单条长度    最短 {min(lens)} / 中位 {lens_sorted[len(lens)//2]} / 最长 {max(lens)}")

    print(f"\n  证据等级分布（A 最硬 → C 最弱）")
    lv = Counter(c["evidence"] for c in chunks)
    for k in ["A", "B", "C", ""]:
        if lv.get(k):
            label = k if k else "未标注"
            print(f"    {label:>6}  {lv[k]:>4} 条")

    print(f"\n  按章节分布")
    for chap, n in Counter(c["chapter"] for c in chunks).items():
        print(f"    {n:>4} 条  {chap}")


def show_samples(chunks: list[dict], n: int) -> None:
    print(f"\n{'=' * 70}")
    print(f"  前 {n} 条切分结果（这就是以后喂给大模型的原料）")
    print(f"{'=' * 70}")
    for c in chunks[:n]:
        print(f"\n┌─ 第 {c['num']} 条 · {c['chapter']} · 证据等级 {c['evidence'] or '无'}")
        print(f"│ 标题：{c['title']}")
        print(f"│ 长度：{len(c['text'])} 字符")
        print("├" + "─" * 68)
        body = c["text"]
        print("\n".join("│ " + line for line in body[:500].split("\n")))
        if len(body) > 500:
            print(f"│ ……（还有 {len(body) - 500} 字符）")
        print("└" + "─" * 68)


def main() -> None:
    ap = argparse.ArgumentParser(description="建 RAG 索引")
    ap.add_argument("--dry-run", action="store_true",
                    help="只做抽取和切分，什么都不存（检查切分用）")
    ap.add_argument("--bm25-only", action="store_true",
                    help="只建关键词索引，不做向量化（不需要密钥，几秒钟）")
    ap.add_argument("--show", type=int, default=0, metavar="N",
                    help="打印前 N 条切分结果供人工检查")
    args = ap.parse_args()

    cfg = rag.load_config()

    print("=" * 70)
    print("  建索引：PDF → 清洗 → 切分 → 存文本 → 关键词索引 → 向量化")
    print("=" * 70)
    print(f"  文档：{cfg['pdf_path']}")
    if not args.dry_run and not args.bm25_only:
        print(f"  向量模型：{cfg['embed_model']}")

    chunks, vectors = rag.build_index(
        cfg,
        embed=not (args.dry_run or args.bm25_only),
        bm25=not args.dry_run,
    )

    print_stats(chunks)

    if args.show:
        show_samples(chunks, args.show)

    print()
    print("=" * 70)
    if args.dry_run:
        print("  【试跑模式】什么都没存。")
        print("  看着没问题，就正式跑。没有密钥也能先跑起来：")
        print("      .venv\\Scripts\\python.exe build_index.py --bm25-only")
        print("  配好密钥之后跑完整版：")
        print("      .venv\\Scripts\\python.exe build_index.py")
    elif args.bm25_only:
        print("  完成！关键词索引已经建好，不需要任何密钥就能检索了：")
        print("      .venv\\Scripts\\python.exe search.py \"怎么预防蘑菇中毒\"")
        print()
        print("  但这一步只让「检索」能跑；要「回答」还需要两步：")
        print("      1. 在 .env 里填两个密钥（看 README 第三步）")
        print("      2. 建完整索引： .venv\\Scripts\\python.exe build_index.py")
    else:
        print("  完成！完整索引已就绪。试试：")
        print("      .venv\\Scripts\\python.exe search.py \"怎么预防蘑菇中毒\"")
        print("      .venv\\Scripts\\python.exe ask.py \"被裁员了能拿多少赔偿\"")
    print("=" * 70)


if __name__ == "__main__":
    main()
