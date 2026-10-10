# -*- coding: utf-8 -*-
"""
第四步：网页版。把 ask.py 包成能点、能看的界面。

跑起来：
    .venv\\Scripts\\python.exe -m streamlit run app.py
或者直接双击 start.bat。

跟 ask.py 的逻辑完全一样，只是把命令行输出换成网页组件。
多了一个能力：**把检索到的原文摊开给你看**。
这一步很关键 —— 能看见"模型是照着哪段话说的"，
你才有资格判断它是真回答还是编的。
"""

import streamlit as st

import rag

st.set_page_config(page_title="高性价比人生指南 · 问答", page_icon="📖", layout="wide")


# ---------------------------------------------------------------- 侧边栏
with st.sidebar:
    st.title("⚙️ 设置")

    top_k = st.slider(
        "每次取回几条资料（top_k）", 1, 10, 5,
        help="太小可能漏掉关键信息，太大则会把不相关的内容也塞给模型，反而干扰它。",
    )

    show_refs = st.checkbox("显示检索到的原文", value=True)

    mode = st.selectbox(
        "检索方式",
        ["hybrid", "vector", "bm25"],
        index=0,
        format_func=lambda x: {
            "hybrid": "hybrid 混合（推荐）",
            "vector": "vector 语义",
            "bm25": "bm25 关键词",
        }[x],
        help="hybrid 两路召回后融合，效果最好；"
             "bm25 纯本地不需要密钥；vector 只走向量语义。",
    )

    st.divider()

    st.caption("**当前配置**")
    try:
        _cfg = rag.load_config()
        st.caption(f"向量模型：`{_cfg['embed_model']}`")
        st.caption(f"回答模型：`{_cfg['llm_model']}`")
    except Exception:
        pass

    if st.button("清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------- 顶部
st.title("📖 高性价比人生指南 · 问答助手")
st.caption(
    "这份文档有 654 条建议，按性价比排序，覆盖健康、法律、金钱、租房、育儿。"
    "下面每一条回答都来自文档原文，可以点开核对。"
)

# ---------------------------------------------------------------- 状态
if "messages" not in st.session_state:
    st.session_state.messages = []

if not rag.CHUNKS_FILE.exists():
    st.error(
        "还没有建索引。先在命令行跑一次：\n\n"
        "```\n.venv\\Scripts\\python.exe build_index.py --bm25-only\n```\n\n"
        "这一步不需要密钥。配好密钥之后，再跑一次不带参数的 `build_index.py` "
        "补上向量索引。"
    )
    st.stop()


# ---------------------------------------------------------------- 渲染历史
SCORE_LABEL = {"vector": "相似度", "bm25": "BM25 分", "hybrid": "融合分"}


def render_refs(hits, mode: str = "hybrid"):
    """把检索到的原文折叠展示，方便核对。"""
    label = SCORE_LABEL.get(rag.resolve_mode(mode), "分数")
    with st.expander(f"📎 这次用到的 {len(hits)} 条原文"):
        for i, (chunk, score) in enumerate(hits, 1):
            st.markdown(
                f"**{i}. 第 {chunk['num']} 条 · {chunk['chapter']}** "
                f"　`证据等级 {chunk['evidence'] or '无'}`　`{label} {score:.3f}`"
            )
            st.markdown(f"> {chunk['title']}")
            st.text(chunk["text"][:800] + ("……" if len(chunk["text"]) > 800 else ""))
            if i < len(hits):
                st.divider()


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("hits") and show_refs:
            render_refs(msg["hits"], msg.get("mode", "hybrid"))


# ---------------------------------------------------------------- 输入
question = st.chat_input("问点什么，比如：怎么预防蘑菇中毒")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            stream, hits = rag.answer(question, top_k=top_k, stream=True, mode=mode)
            text = st.write_stream(stream)
        except SystemExit as e:
            st.error(str(e))
            st.stop()

        if show_refs:
            render_refs(hits, mode)

    st.session_state.messages.append(
        {"role": "assistant", "content": text, "hits": hits, "mode": mode}
    )
