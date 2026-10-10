# -*- coding: utf-8 -*-
"""
检索效果对照评测：vector / bm25 / hybrid 三路，看谁把「正确的那一条」召回到前 5。

为什么需要它：
    「混合检索比单路好」这句话，面试官一定会追问「好在哪、好多少」。
    凭感觉说没用，得有一组固定的问题 + 人工标注的正确条目，跑出 Recall@5 的对比。

用法：
    .venv\\Scripts\\python.exe eval_retrieval.py
    .venv\\Scripts\\python.exe eval_retrieval.py --top_k=10

评测集说明：
    questions 里每一条是（口语化提问，正确条目标题的开头几个字）。
    标题取自 PDF 里「数字. 标题」那一行，用开头片段做匹配，避免抄全文。
    这些提问刻意写成普通人的说法（「老板拖欠工资找谁」），
    而不是文档里的原话（「被欠薪先投诉劳动监察」），这样才测得出语义检索的价值。
"""

import argparse
import time

import rag

# ---------------------------------------------------------------
# 评测集：12 条问题，覆盖 12 个不同章节
# ---------------------------------------------------------------
CASES: list[tuple[str, str]] = [
    ("坐后排也要系安全带吗", "系安全带，前排后排都系"),
    ("骑车不戴头盔会怎么样", "骑摩托车、电动自行车戴头盔并扣好"),
    ("戒烟到底有什么好处", "戒烟，越早越好"),
    ("手机老是弹通知，怎么才能静下来工作", "关掉非必要通知，工作时把手机放到视线之外"),
    ("周末睡懒觉会不会打乱作息", "固定起床时间，周末也一样"),
    ("怎么才能让一直拖着的事真的做起来", "把「打算做」写成「几点、在哪、遇到什么就做什么」"),
    ("自动续费老是忘记取消，一直扣钱怎么办", "关掉所有自动续费，改为到期手动续"),
    ("失业以后能领什么钱", "失业了先在线申领失业保险金"),
    ("老板拖欠工资我应该找谁", "被欠薪先投诉劳动监察"),
    ("被诈骗转账了怎么紧急止损", "发现被骗，立刻打 110 或 96110 要求止付"),
    ("有人突然倒地没有呼吸怎么办", "有人倒地没呼吸，立刻用力按压胸口"),
    ("房东不退押金怎么办", "押金的数额、退还时间和扣减情形，必须写进合同"),
    ("怎么防止账号被盗", "邮箱、支付、社交账号都开二次验证"),
    ("开车撞了人第一件事应该做什么", "出了交通事故先停车、救人、报警，不要跑"),
    ("在群里转发没核实的消息会不会出事", "群里不转发不知真假的灾情"),
    ("想创业但没钱，能不能借钱开店", "只拿亏得起的钱创业，不动家底、不借钱开张"),
    ("在外地看病医保怎么直接结算", "先办门诊慢特病认定再办异地备案"),
    ("路上看见老人摔倒了要不要扶", "老人摔倒、有人倒地，先蹲下喊他"),
    ("做技术接私活之前要考虑什么", "动手前先问三句"),
    ("写游戏外挂赚钱合法吗", "不做、不卖游戏外挂和辅助"),
]


def find_target(chunks: list[dict], title_prefix: str) -> int | None:
    """按标题开头找正确条目的下标；找不到返回 None（用来发现评测集本身写错了）。"""
    for i, c in enumerate(chunks):
        if c["title"].startswith(title_prefix):
            return i
    return None


def rank_of(hits: list[tuple[dict, float]], target_idx: int,
            chunks: list[dict]) -> int:
    """正确条目在这条问题的结果里排第几；没出现返回 0。"""
    by_num_chapter = (chunks[target_idx]["num"], chunks[target_idx]["chapter"])
    for rank, (c, _) in enumerate(hits, start=1):
        if (c["num"], c["chapter"]) == by_num_chapter:
            return rank
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="检索效果对照评测")
    ap.add_argument("--top_k", type=int, default=5, help="取回条数，默认 5")
    ap.add_argument("--mode", default=None, help="只测某一路：vector / bm25 / hybrid")
    args = ap.parse_args()

    cfg = rag.load_config()
    chunks = rag.load_chunks()
    modes = [args.mode] if args.mode else ["bm25", "vector", "hybrid"]

    # 先校验评测集：标题写错了要立刻发现，否则测出来的是假分数
    targets: list[int] = []
    for q, prefix in CASES:
        idx = find_target(chunks, prefix)
        if idx is None:
            raise SystemExit(f"[评测集有误] 找不到标题以「{prefix}」开头的条目")
        targets.append(idx)

    print(f"\n评测集 {len(CASES)} 条，top_k = {args.top_k}\n")
    print(f"{'问题':<26}{'正确条目的排名（0 = 没进前 K）':<34}")
    print("-" * 78)

    summary: dict[str, dict[str, float]] = {}
    for mode in modes:
        hit = 0
        mrr = 0.0
        ranks: list[int] = []
        for (q, _), tgt in zip(CASES, targets):
            hits = rag.search(q, top_k=args.top_k, cfg=cfg, mode=mode)
            r = rank_of(hits, tgt, chunks)
            ranks.append(r)
            if r:
                hit += 1
                mrr += 1.0 / r
        summary[mode] = {
            "recall": hit / len(CASES),
            "mrr": mrr / len(CASES),
            "ranks": ranks,
        }
        if mode == modes[0]:
            for (q, _), r in zip(CASES, ranks):
                print(f"{q:<26}{r if r else '未召回':<34}")

    print("-" * 78)
    print(f"{'路数':<10}{'Recall@' + str(args.top_k):<16}{'MRR':<10}")
    for mode, s in summary.items():
        print(f"{mode:<10}{s['recall'] * 100:>5.0f}%{'':<10}{s['mrr']:>6.3f}")

    print("\n每条问题在三路下的排名（0 = 没进前 K）")
    print(f"{'问题':<26}" + "".join(f"{m:<10}" for m in modes))
    for i, (q, _) in enumerate(CASES):
        print(f"{q:<26}" + "".join(f"{summary[m]['ranks'][i] or '-':<10}" for m in modes))
    print()


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"耗时 {time.time() - t0:.1f} 秒\n")
