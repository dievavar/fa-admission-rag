# MVP: RAG-ассистент абитуриента Финансового университета

Учебный проект для курсовой: **GigaChat через API** генерирует ответ, **EmbeddingsGigaR через API** преобразует тексты и вопросы в векторы, **Qdrant в Docker** хранит и находит фрагменты, **Streamlit** показывает чат. **Локальные языковые и embedding-модели не устанавливаются**. Qdrant разворачивается только как база данных на Mac.

> Сервис не является официальным приложением Финансового университета. В `data/raw/` находятся **четыре учебных конспекта**, составленных по открытым разделам fa.ru. Это не полные нормативные документы. Они относятся к кампании **2026 года**; не воспринимайте их как правила поступления в 2027 году.

## 1. Первый запуск на macOS

Условия: запущен Docker Desktop, установлен Python 3.11 или 3.12, есть интернет и **ключ авторизации GigaChat API** с доступом к генеративной и embedding-моделям. Для GigaChat API следует проверить сертификаты НУЦ Минцифры согласно официальной документации (не отключать SSL-проверку без необходимости).

Откройте Terminal в папке проекта и выполните:

```bash
# 1. Запуск Qdrant

docker compose up -d
docker compose ps

# 2. Подготовка Python
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Создание персональной конфигурации
cp .env.example .env
open -a TextEdit .env
```

В файле `.env` укажите **свой** `GIGACHAT_CREDENTIALS`. Это *ключ авторизации*, а не временный access token; **не присылайте его в чат и не публикуйте в GitHub**. По умолчанию используется `GIGACHAT_SCOPE=GIGACHAT_API_PERS`. Если ваш личный кабинет указывает другое значение scope, замените его. Названия моделей (`GigaChat-2-Pro` и `EmbeddingsGigaR`) должны быть доступны вашему тарифу.

После сохранения `.env`:

```bash
# Убедиться, что Qdrant доступен, а GigaChat выдаёт эмбеддинги
# ВНИМАНИЕ: в этом шаге делается один запрос к API
python -m src.check_setup

# Загрузить 4 демонстрационных документа в Qdrant
python -m src.ingest --reset

# Запустить чат
streamlit run app.py
```

В браузере откройте адрес, показанный Streamlit (обычно `http://localhost:8501`). Вопрос для первого теста: **«Какие способы подачи документов были в 2026 году?»**

## 2. Проверка Qdrant

- Адрес API: `http://localhost:6333`
- Панель коллекций: **http://localhost:6333/dashboard**
- Проверка сервера: `curl http://localhost:6333`
- Остановить контейнер: `docker compose stop`
- Запустить снова: `docker compose start`
- Удалить контейнер, но сохранить данные: `docker compose down`

В `docker-compose.yml` используется именованный том `qdrant_data`. **Не используйте `docker compose down -v`**: флаг `-v` удаляет том и данные индекса. Доступ ограничен `127.0.0.1`, база не открыта в интернет. Не публикуйте порт без TLS и аутентификации.

## 3. Где хранится информация

`data/raw/` — исходные файлы PDF/MD/TXT, `data/raw/manifest.json` — название, ссылка на оригинал и год кампании. Скрипт `src/ingest.py` извлекает текст, разбивает на фрагменты и направляет их на **EmbeddingsGigaR API**. В Qdrant сохраняются векторы и текст каждого фрагмента вместе с метаданными. Запрос абитуриента также отправляется в EmbeddingsGigaR, затем Qdrant находит похожие фрагменты, после чего **GigaChat API** формирует ответ. Учтите, что при такой схеме тексты документов и пользовательские вопросы передаются во внешний API. Не загружайте туда персональные данные.

## 4. Как добавить 1–3 настоящих PDF

1. Скачайте PDF с **официального fa.ru**; предпочтительно документы с выделяемым текстом, а не сканы.
2. Поместите файлы в `data/raw/`, например `pravila_2026.pdf`.
3. Откройте `data/raw/manifest.json` и добавьте запись **внутрь JSON-объекта** (через запятую после предыдущей записи):

```json
"pravila_2026.pdf": {
  "title": "Правила приёма — 2026 (проверьте точное название)",
  "source_url": "https://www.fa.ru/укажите-реальную-ссылку-на-pdf",
  "campaign_year": 2026
}
```

4. Запустите `python -m src.ingest --reset`; данные из файлов того же `ADMISSION_YEAR` проиндексируются заново.
5. Перейдите в Qdrant Dashboard → Collections → `fa_admission_2026`, чтобы увидеть точки и payload.

Для другой кампании установите `ADMISSION_YEAR=2027`, `QDRANT_COLLECTION=fa_admission_2027`; загрузите документы 2027 года и повторите индексацию. **Не используйте 2026 материалы как актуальные для 2027 без проверки.**

**Ограничения:** обычный `pypdf` не распознаёт сканы; табличные PDF (ЕГЭ, места, стоимость) иногда извлекаются с нарушением структуры. Перед индексацией таких таблиц требуется контроль извлечённого текста.

## 5. Если позже переносить Qdrant на Amvera

Создайте в Amvera преднастроенный сервис Qdrant. В соответствии с документацией Amvera включите аутентификацию сервиса, задав `QDRANT__SERVICE__API_KEY`; настройте внешний HTTPS-домен, если подключение с Mac, и убедитесь, что доступ разрешён.

В `.env` замените параметры:

```dotenv
QDRANT_URL=https://ваш-qdrant.amvera.io
QDRANT_API_KEY=ваш_ключ_к_qdrant
```

Это **другой ключ**, не `GIGACHAT_CREDENTIALS`. Запустите `python -m src.ingest --reset` для создания индекса в новом экземпляре. Остальной Python-код менять не нужно. За размещение и сетевой трафик у облачного провайдера может взиматься плата.

## 6. Что делает код

```text
app.py                     Streamlit-чат
src/config.py              загрузка настроек .env
src/check_setup.py         проверка Qdrant + GigaChat Embeddings API
src/documents.py           чтение PDF/MD/TXT, ссылка и год
src/chunking.py            деление на фрагменты
src/embeddings.py          только EmbeddingsGigaR (либо иная модель GigaChat API)
src/gigachat_service.py    официальный SDK GigaChat
src/vector_store.py        подключение Qdrant по URL, запись, поиск
src/ingest.py              индексация документов
src/rag.py                 retrieval + генерация ответа
src/evaluate_retrieval.py  тест Hit@k на тестовых вопросах
tests/                     локальные модульные тесты
```

Запуск тестов: `python -m pytest -q` (не требует ключа и Docker). Оценка поиска после индексации: `python -m src.evaluate_retrieval` (требует API). Перед реальной демонстрацией проверьте, что ссылки в ответах подтверждают утверждения. 4 конспекта не дают достаточного охвата для полноценной оценки качества; для курсовой понадобится размеченный набор вопросов.

## 7. Почему не ru-en-RoSBERTa для текущей версии

[`ai-forever/ru-en-RoSBERTa`](https://huggingface.co/ai-forever/ru-en-RoSBERTa) — опубликованная на Hugging Face модель **для получения эмбеддингов**, в том числе семантического поиска на русском. Пример со страницы модели загружает веса через `SentenceTransformer(...)`. Чтобы пользоваться именно RoSBERTa *по ключу*, нужен **отдельный hosted inference endpoint**, а не только ссылка на страницу модели. Для MVP выбран готовый API EmbeddingsGigaR, поэтому установка PyTorch / SentenceTransformers не нужна. Если позже решите сравнивать модели, используйте отдельные коллекции или полностью переиндексируйте одну коллекцию.

## Официальная документация

- [GigaChat: API и ключи](https://developers.sber.ru/docs/ru/gigachat/api/reference/rest/gigachat-api)
- [GigaChat: эмбеддинги и retrieval-инструкции](https://developers.sber.ru/docs/ru/gigachat/guides/embeddings)
- [GigaChat: Python SDK](https://developers.sber.ru/docs/ru/gigachain/tools/python/gigachat)
- [Qdrant: локальная установка в Docker](https://qdrant.tech/documentation/quickstart/)
- [Amvera: Qdrant](https://docs.amvera.ru/databases/Qdrant.html)
