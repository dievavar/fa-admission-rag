"""Streamlit UI: streamlit run app.py"""
from __future__ import annotations

import streamlit as st

from src.config import get_settings
from src.rag import ask

settings = get_settings()
st.set_page_config(page_title="Консультант абитуриента Финуниверситета", page_icon="🎓")
st.title("Ассистент абитуриента Финансового университета")
st.caption("Учебный MVP с GigaChat, RAG и Qdrant. Не является официальным сервисом университета.")
st.warning(
    f"Загружены демонстрационные сведения приёмной кампании {settings.admission_year} года. "
    "Для поступления в следующие годы проверяйте официальные объявления на fa.ru."
)
with st.sidebar:
    st.subheader("Конфигурация")
    st.write(f"Генерация: `{settings.model}`")
    st.write(f"Эмбеддинги API: `{settings.embedding_model}`")
    st.write(f"База: `{settings.qdrant_url}`")
    st.write(f"Коллекция Qdrant: `{settings.collection}`")
    st.caption("Для смены модели измените .env, затем заново выполните индексацию с --reset.")
    if st.button("Очистить историю"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Найденные фрагменты"):
                for source in msg["sources"]:
                    st.markdown(source)

if question := st.chat_input("Например: когда заканчивается приём документов в 2026 году?"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Ищу подтверждение в документах..."):
                result = ask(question, settings)
            st.markdown(result.text)
            source_labels = []
            for i, hit in enumerate(result.sources, 1):
                label = (
                    f"**[{i}] {hit.title}** — [Оригинал]({hit.source_url})"
                    f"{f', стр. {hit.page}' if hit.page else ''} "
                    f"(кампания {hit.campaign_year}; score {hit.score:.3f})\n\n"
                    f"> {hit.text[:300].replace(chr(10), ' ')}…"
                )
                source_labels.append(label)
            if source_labels:
                with st.expander("Найденные фрагменты и ссылки"):
                    for label in source_labels:
                        st.markdown(label)
            st.session_state.messages.append({
                "role": "assistant", "content": result.text, "sources": source_labels,
            })
        except Exception as error:
            st.error(f"Не удалось выполнить запрос: {error}")
            st.info("Проверьте .env и то, что индекс создан: python -m src.ingest --reset")
