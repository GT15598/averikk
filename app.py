import csv
import hashlib
import io
import json
import os
import random
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

try:
    import fitz
except ImportError:
    fitz = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


ROOT = Path(__file__).parent
DB_PATH = ROOT / "questionforge.sqlite3"

LEVELS = {
    1: ("Очень легкий", "Определение или простой факт"),
    2: ("Легкий", "Объяснение смысла и связей"),
    3: ("Средний", "Применение знаний"),
    4: ("Сложный", "Анализ ситуации или ошибки"),
    5: ("Очень сложный", "Нестандартное решение с обоснованием"),
}
STOP_WORDS = {
    "который", "которая", "которые", "это", "также", "может", "можно", "нужно",
    "быть", "более", "менее", "если", "при", "для", "что", "как", "или", "так",
    "его", "ее", "они", "из", "на", "по", "с", "в", "и", "а", "но", "не", "до",
    "за", "от", "к", "у", "the", "this", "that", "with", "from", "and", "for",
}


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                explanation TEXT NOT NULL,
                topic TEXT NOT NULL,
                level INTEGER NOT NULL,
                kind TEXT NOT NULL,
                options TEXT NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        connection.commit()


def db_save_questions(document: str, questions: list[dict[str, Any]]) -> None:
    with sqlite3.connect(DB_PATH) as connection:
        connection.executemany(
            """INSERT INTO questions
            (document, question, answer, explanation, topic, level, kind, options, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    document,
                    item["question"],
                    item["answer"],
                    item["explanation"],
                    item["topic"],
                    item["level"],
                    item["kind"],
                    json.dumps(item.get("options", []), ensure_ascii=False),
                    item.get("source", ""),
                )
                for item in questions
            ],
        )
        connection.commit()


def db_load_questions(document: str) -> list[dict[str, Any]]:
    with sqlite3.connect(DB_PATH) as connection:
        rows = connection.execute(
            """SELECT question, answer, explanation, topic, level, kind, options, source
               FROM questions WHERE document = ? ORDER BY id""",
            (document,),
        ).fetchall()
    return [
        {
            "question": row[0],
            "answer": row[1],
            "explanation": row[2],
            "topic": row[3],
            "level": row[4],
            "kind": row[5],
            "options": json.loads(row[6] or "[]"),
            "source": row[7],
        }
        for row in rows
    ]


def db_documents() -> list[str]:
    with sqlite3.connect(DB_PATH) as connection:
        rows = connection.execute(
            "SELECT document, COUNT(*) FROM questions GROUP BY document ORDER BY MAX(id) DESC"
        ).fetchall()
    return [f"{row[0]} · {row[1]} вопросов" for row in rows]


def evaluate_open_answer(user_answer: str, expected_answer: str) -> tuple[str, int]:
    user_words = {
        word for word in re.findall(r"[А-Яа-яЁёA-Za-z]{4,}", user_answer.lower())
        if word not in STOP_WORDS
    }
    expected_words = {
        word for word in re.findall(r"[А-Яа-яЁёA-Za-z]{4,}", expected_answer.lower())
        if word not in STOP_WORDS
    }
    if not user_words:
        return "Ответ не заполнен.", 0
    overlap = len(user_words & expected_words)
    score = round(overlap / max(len(expected_words), 1) * 100)
    if score >= 55:
        return "Ответ содержит основные идеи правильного решения.", score
    if score >= 25:
        return "Ответ частично совпадает с эталоном. Добавьте ключевые понятия и объяснение.", score
    return "Ответ сильно отличается от эталона. Повторите материал и попробуйте снова.", score


def generate_full_set(
    topic: str,
    text: str,
    pages: list[dict[str, Any]],
    kind: str,
    per_level: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for level in LEVELS:
        result.extend(ai_generate(topic, per_level, level, kind, text, pages))
    return result


def extract_document(uploaded_file: Any) -> tuple[str, list[dict[str, Any]]]:
    data = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix.lower()
    pages: list[dict[str, Any]] = []
    if suffix == ".txt":
        text = data.decode("utf-8-sig", errors="replace").strip()
        pages.append({"page": 1, "text": text})
    elif suffix == ".pdf":
        if fitz is None:
            raise RuntimeError("Установите зависимости из requirements.txt.")
        document = fitz.open(stream=data, filetype="pdf")
        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text("text").strip()
            if page_text:
                pages.append({"page": page_number, "text": page_text})
        text = "\n\n".join(f"[Страница {item['page']}]\n{item['text']}" for item in pages)
    else:
        raise ValueError("Можно загрузить только PDF или TXT.")
    if len(text.strip()) < 40:
        raise ValueError("В документе слишком мало распознаваемого текста.")
    return text, pages


def sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if len(part.strip()) >= 20]


def detect_topics(text: str) -> list[str]:
    headings = re.findall(
        r"(?m)^\s*(?:\d+(?:\.\d+)*[.)]?\s+)?([А-ЯA-ZЁ][^.!?\n]{3,80})\s*$",
        text,
    )
    topics: list[str] = []
    for heading in headings:
        value = re.sub(r"\s+", " ", heading).strip(" -:;")
        if 3 <= len(value.split()) <= 10 and value.lower() not in {item.lower() for item in topics}:
            topics.append(value)
    words = re.findall(r"[А-Яа-яЁёA-Za-z]{5,}", text.lower())
    counts = Counter(word for word in words if word not in STOP_WORDS)
    for word, _ in counts.most_common(30):
        label = word.capitalize()
        if not any(word in topic.lower() for topic in topics):
            topics.append(label)
        if len(topics) >= 8:
            break
    return topics[:8] or ["Основные понятия"]


def source_for_topic(topic: str, text: str) -> str:
    matches = [item for item in sentences(text) if topic.lower() in item.lower()]
    source = matches[0] if matches else (sentences(text) or [text])[0]
    return source[:700].strip()


def page_for_source(source: str, pages: list[dict[str, Any]]) -> str:
    for item in pages:
        if source[:80].lower() in item["text"].lower():
            return f"Страница {item['page']}"
    return "Раздел документа"


def normalize_item(item: dict[str, Any], topic: str, level: int, kind: str, source: str) -> dict[str, Any]:
    options = item.get("options", [])
    if not isinstance(options, list):
        options = []
    return {
        "question": str(item.get("question", "")).strip(),
        "answer": str(item.get("answer", "")).strip(),
        "explanation": str(item.get("explanation", "")).strip(),
        "topic": topic,
        "level": level,
        "kind": kind,
        "options": [str(option).strip() for option in options[:4]],
        "source": str(item.get("source", "")).strip() or source,
    }


def local_generate(topic: str, amount: int, level: int, kind: str, text: str, pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    all_sentences = sentences(text)
    topic_sentences = [sentence for sentence in all_sentences if topic.lower() in sentence.lower()]
    source_pool = topic_sentences + [sentence for sentence in all_sentences if sentence not in topic_sentences]
    source_pool = source_pool or [f"Материал по теме «{topic}»."]
    result = []
    question_starts = {
        1: [
            f"Какой факт о теме «{topic}» приведен в материале?",
            f"Что утверждается в тексте о теме «{topic}»?",
            f"Как определяется тема «{topic}» в учебном материале?",
        ],
        2: [
            f"Как объяснить связь темы «{topic}» с изучаемым материалом?",
            f"Почему утверждение о теме «{topic}» важно?",
            f"Как пересказать основную мысль о теме «{topic}»?",
        ],
        3: [
            f"Как применить принцип темы «{topic}» в практической ситуации?",
            f"Какой результат получится при использовании идеи «{topic}»?",
            f"Как решить задачу, используя правило из темы «{topic}»?",
        ],
        4: [
            f"Какая ошибка возможна при использовании темы «{topic}»?",
            f"Как проверить правильность решения по теме «{topic}»?",
            f"Почему неверное применение темы «{topic}» приведет к ошибке?",
        ],
        5: [
            f"Как предложить нестандартное решение задачи по теме «{topic}»?",
            f"Как объединить идею «{topic}» с другими знаниями из материала?",
            f"Как обосновать сложное решение, связанное с темой «{topic}»?",
        ],
    }
    explanations = {
        1: "Проверяется знание факта или определения из документа.",
        2: "Проверяется понимание смысла и связей между понятиями.",
        3: "Проверяется применение изученного принципа.",
        4: "Проверяется анализ ситуации и поиск причины ошибки.",
        5: "Проверяется самостоятельное решение и аргументация.",
    }
    for index in range(amount):
        source = source_pool[(index + level - 1) % len(source_pool)]
        short = source.rstrip(" .,:;")
        question = question_starts[level][index % len(question_starts[level])]
        if index >= len(question_starts[level]):
            question = f"{question} Приведите отдельное обоснование для аспекта {index + 1}."
        answer = short + "."
        if level == 4:
            answer = f"Ошибка возникает, если нарушить условие из источника: {short}. Нужно проверить это условие перед применением."
        elif level == 5:
            answer = f"Решение должно опираться на содержание источника: {short}. Выбранный подход необходимо объяснить и проверить на примере."
        item = {
            "question": question,
            "answer": answer,
            "explanation": explanations[level],
            "topic": topic,
            "level": level,
            "kind": kind,
            "options": [],
            "source": page_for_source(source, pages),
        }
        if kind == "Тестовый":
            distractors = [
                sentence.rstrip(" .,:;") + "."
                for sentence in source_pool
                if sentence != source
            ]
            distractors.extend([
                f"Это утверждение относится к другой теме, а не к «{topic}».",
                "В документе такого утверждения нет.",
                "Это предположение не подтверждается выбранным материалом.",
            ])
            item["options"] = [answer] + distractors[:3]
            random.shuffle(item["options"])
        result.append(item)
    return result


def ai_generate(topic: str, amount: int, level: int, kind: str, text: str, pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or OpenAI is None:
        return local_generate(topic, amount, level, kind, text, pages)
    source = source_for_topic(topic, text)
    prompt = f"""Создай {amount} учебных вопросов по материалу ниже.
Тема: {topic}
Уровень: {level} — {LEVELS[level][0]}: {LEVELS[level][1]}
Тип: {kind}
Не добавляй факты, которых нет в материале.
Верни только JSON-объект вида {{"questions": [{{"question": "...", "answer": "...", "explanation": "...", "options": ["..."], "source": "..."}}]}}.
Для типа «Тестовый» сделай ровно 4 варианта options, включая правильный ответ.
Для типа «Открытый» options должен быть пустым массивом.
Материал:
{source}
{text[:14000]}"""
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.35,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Ты методист. Соблюдай JSON и опирайся только на исходный материал."},
            {"role": "user", "content": prompt},
        ],
    )
    payload = json.loads(response.choices[0].message.content or "{}")
    raw_questions = payload.get("questions", []) if isinstance(payload, dict) else []
    normalized = [normalize_item(item, topic, level, kind, source) for item in raw_questions[:amount]]
    return normalized if len(normalized) == amount else local_generate(topic, amount, level, kind, text, pages)


def quality_issues(item: dict[str, Any], all_questions: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    question = item.get("question", "").strip()
    answer = item.get("answer", "").strip()
    if len(question) < 15:
        issues.append("короткий вопрос")
    if len(answer) < 10:
        issues.append("нет полноценного ответа")
    if item.get("kind") == "Тестовый" and len(item.get("options", [])) != 4:
        issues.append("нужно ровно 4 варианта")
    if item.get("kind") == "Тестовый" and answer not in item.get("options", []):
        issues.append("правильного ответа нет среди вариантов")
    same = sum(other.get("question", "").strip().lower() == question.lower() for other in all_questions)
    if same > 1:
        issues.append("повторяющийся вопрос")
    return issues


def export_csv(items: list[dict[str, Any]]) -> str:
    output = io.StringIO()
    fields = ["question", "answer", "explanation", "topic", "level", "kind", "options", "source"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for item in items:
        row = dict(item)
        row["options"] = " | ".join(row.get("options", []))
        writer.writerow({field: row.get(field, "") for field in fields})
    return output.getvalue()


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        :root { --ink:#172033; --muted:#667085; --line:#e6eaf0; --violet:#5b4bdb; }
        .stApp { background:#f7f8fc; color:var(--ink); }
        .block-container { max-width:1360px; padding:2.4rem 3.2rem 4rem; }
        header[data-testid="stHeader"] { background:transparent; }
        [data-testid="stSidebar"] { background:#ffffff; border-right:1px solid var(--line); }
        [data-testid="stSidebar"] > div:first-child { padding:2rem 1.35rem; }
        [data-testid="stSidebar"] * { color:var(--ink); }
        [data-testid="stSidebar"] hr { border-color:var(--line); }
        [data-testid="stSidebar"] label { font-size:13px; font-weight:700; color:#344054; }
        [data-testid="stSidebar"] .stButton button { border-radius:12px; min-height:44px; }
        .brand { display:flex; align-items:center; gap:14px; margin:0 0 28px; }
        .brand-mark { width:50px; height:50px; border-radius:16px; background:linear-gradient(135deg,#6657e8,#8b5cf6); display:flex; align-items:center; justify-content:center; font-size:25px; box-shadow:0 8px 20px rgba(91,75,219,.22); }
        .brand h1 { margin:0; font-size:29px; letter-spacing:-.7px; color:var(--ink); }
        .brand p { margin:4px 0 0; color:var(--muted); font-size:14px; }
        .hero { position:relative; overflow:hidden; border-radius:26px; padding:42px 46px; color:white; background:linear-gradient(120deg,#24205d 0%,#4d3bd1 55%,#7758e9 100%); box-shadow:0 18px 42px rgba(76,60,190,.2); margin-bottom:28px; }
        .hero:after { content:""; position:absolute; width:290px; height:290px; border-radius:50%; right:-90px; top:-140px; background:rgba(255,255,255,.1); }
        .hero h2 { margin:0; max-width:760px; font-size:36px; line-height:1.12; letter-spacing:-1px; }
        .hero p { margin:14px 0 0; max-width:700px; opacity:.86; font-size:16px; line-height:1.6; }
        .eyebrow { text-transform:uppercase; letter-spacing:1.5px; font-weight:800; font-size:11px; opacity:.78; margin-bottom:12px; }
        .step { border:1px solid var(--line); border-radius:18px; padding:22px; min-height:130px; background:#fff; box-shadow:0 5px 16px rgba(16,24,40,.035); }
        .step b { display:block; margin-bottom:8px; color:var(--ink); font-size:16px; }
        .step span { color:var(--muted) !important; font-size:14px; line-height:1.45; }
        .section-title { margin:4px 0 14px; color:var(--ink); font-size:21px; font-weight:800; }
        .upload-card { border:1px dashed #b9b3f3; border-radius:18px; padding:18px; background:#f8f7ff; margin-top:14px; }
        div[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:16px; padding:15px 16px; box-shadow:0 5px 16px rgba(16,24,40,.03); }
        div[data-testid="stMetric"] label { color:var(--muted); }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] { color:var(--ink); }
        div[data-testid="stExpander"] { border-radius:16px; border-color:var(--line); background:#fff; }
        .source { background:#f1f5ff; border-left:4px solid #6475e8; padding:11px 14px; border-radius:9px; color:#3446a3; font-size:13px; }
        .mini-note { color:var(--muted); font-size:13px; line-height:1.5; }
        .stButton button, .stDownloadButton button { border-radius:11px; font-weight:700; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    init_db()
    st.set_page_config(page_title="QuestionForge", page_icon="🧠", layout="wide")
    apply_styles()
    st.markdown(
        '<div class="brand"><div class="brand-mark">🧠</div><div><h1>QuestionForge</h1><p>Учебный материал превращается в понятную проверку знаний</p></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero"><div class="eyebrow">AI STUDY WORKSPACE</div><h2>Превратите учебный материал<br>в готовую проверку знаний</h2><p>Загрузите PDF или TXT — система найдёт темы, создаст вопросы пяти уровней и подготовит ответы с указанием источника.</p></div>',
        unsafe_allow_html=True,
    )
    for key, default in {
        "document_name": "",
        "text": "",
        "pages": [],
        "topics": [],
        "questions": [],
        "quiz_result": None,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    with st.sidebar:
        st.markdown("### 01 · Материал")
        uploaded = st.file_uploader("Загрузите PDF или TXT", type=["pdf", "txt"])
        if uploaded is not None and uploaded.name != st.session_state.document_name:
            try:
                text, pages = extract_document(uploaded)
                st.session_state.document_name = uploaded.name
                st.session_state.text = text
                st.session_state.pages = pages
                st.session_state.topics = detect_topics(text)
                st.session_state.questions = []
                st.session_state.quiz_result = None
                st.success("Материал обработан")
            except (ValueError, RuntimeError) as error:
                st.error(str(error))
        if st.session_state.text:
            st.caption(f"Файл: {st.session_state.document_name}")
            st.caption(f"Символов: {len(st.session_state.text):,} · страниц: {len(st.session_state.pages)}")
        st.divider()
        st.markdown("### 02 · Генерация")
        if not st.session_state.text:
            st.info("Сначала загрузите учебный материал.")
        else:
            selected_topic = st.selectbox("Тема", st.session_state.topics)
            amount = st.slider("Количество вопросов", 1, 10, 5)
            level = st.select_slider(
                "Сложность",
                options=list(LEVELS),
                value=3,
                format_func=lambda value: f"{value} · {LEVELS[value][0]}",
            )
            kind = st.radio("Тип", ["Тестовый", "Открытый"], horizontal=True)
            if st.button("Создать вопросы", type="primary", use_container_width=True):
                with st.spinner("Анализируем материал и создаем вопросы..."):
                    try:
                        new_items = ai_generate(selected_topic, amount, level, kind, st.session_state.text, st.session_state.pages)
                        st.session_state.questions.extend(new_items)
                        st.toast(f"Создано вопросов: {len(new_items)}")
                    except Exception as error:
                        st.error(f"Ошибка генерации: {error}")
            if st.button("Создать набор всех 5 уровней", use_container_width=True):
                with st.spinner("Создаем вопросы каждого уровня..."):
                    try:
                        full_set = generate_full_set(
                            selected_topic,
                            st.session_state.text,
                            st.session_state.pages,
                            kind,
                            max(1, min(amount, 5)),
                        )
                        st.session_state.questions.extend(full_set)
                        st.toast(f"Добавлен набор из {len(full_set)} вопросов")
                    except Exception as error:
                        st.error(f"Ошибка генерации набора: {error}")
            if st.button("Очистить банк вопросов", use_container_width=True):
                st.session_state.questions = []
                st.session_state.quiz_result = None
                st.rerun()
        saved_sets = db_documents()
        if saved_sets:
            st.divider()
            st.markdown("### 03 · Сохраненные наборы")
            saved_choice = st.selectbox("Документ", ["Не загружать"] + saved_sets)
            if st.button("Загрузить из SQLite", use_container_width=True) and saved_choice != "Не загружать":
                saved_document = saved_choice.rsplit(" · ", 1)[0]
                st.session_state.questions = db_load_questions(saved_document)
                st.toast(f"Загружено вопросов: {len(st.session_state.questions)}")

    if not st.session_state.text:
        st.markdown(
            '<div class="upload-card"><b style="font-size:16px;color:#312e81">Начните с учебного материала</b><div class="mini-note">Загрузите файл в левой панели. После обработки здесь появятся темы и рабочее пространство проекта.</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-title">Всё необходимое в одном рабочем пространстве</div>', unsafe_allow_html=True)
        cards = st.columns(4)
        for column, title, description in zip(
            cards,
            ["01 · Темы", "02 · Уровни", "03 · Редактор", "04 · Проверка"],
            ["Находит основные темы в вашем документе", "От простого определения до сложной задачи", "Изменяйте, удаляйте и сохраняйте вопросы", "Показывает повторы и недостающие данные"],
        ):
            with column:
                st.markdown(f'<div class="step"><b>{title}</b><span>{description}</span></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Как это работает</div>', unsafe_allow_html=True)
        flow = st.columns(4)
        for column, number, title in zip(flow, ["01", "02", "03", "04"], ["Загрузите", "Выберите", "Создайте", "Проверьте"]):
            with column:
                st.markdown(f'<div class="step"><b style="color:#5b4bdb">{number}</b><span>{title}</span></div>', unsafe_allow_html=True)
        return

    metrics = st.columns(4)
    metrics[0].metric("Вопросов", len(st.session_state.questions))
    metrics[1].metric("Тем", len(st.session_state.topics))
    metrics[2].metric("Тестовых", sum(item["kind"] == "Тестовый" for item in st.session_state.questions))
    bad_count = sum(bool(quality_issues(item, st.session_state.questions)) for item in st.session_state.questions)
    metrics[3].metric("Нужна проверка", bad_count)

    tab_material, tab_questions, tab_quality, tab_quiz = st.tabs(["📚 Материал и темы", "✍️ Банк вопросов", "✅ Качество", "🎯 Мини-тест"])
    with tab_material:
        left, right = st.columns([1, 1.4])
        with left:
            st.subheader("Основные темы")
            for index, topic in enumerate(st.session_state.topics, start=1):
                st.markdown(f"**{index}.** {topic}")
        with right:
            st.subheader(st.session_state.document_name)
            with st.expander("Показать извлеченный текст"):
                st.text(st.session_state.text[:12000])
    with tab_questions:
        if not st.session_state.questions:
            st.info("Выберите параметры слева и создайте первый набор вопросов.")
        else:
            controls = st.columns([1, 2, 1])
            with controls[0]:
                filter_level = st.selectbox("Уровень", ["Все"] + [str(value) for value in LEVELS])
            with controls[1]:
                search = st.text_input("Поиск", placeholder="Поиск по вопросу или теме")
            with controls[2]:
                save_to_db = st.button("Сохранить в SQLite")
            if save_to_db:
                db_save_questions(st.session_state.document_name, st.session_state.questions)
                st.success("Вопросы сохранены в SQLite.")
            visible: list[tuple[int, dict[str, Any]]] = []
            for index, item in enumerate(st.session_state.questions):
                matches_level = filter_level == "Все" or str(item["level"]) == filter_level
                matches_search = not search or search.lower() in f"{item['question']} {item['topic']}".lower()
                if matches_level and matches_search:
                    visible.append((index, item))
            st.caption(f"Показано: {len(visible)} из {len(st.session_state.questions)}")
            for index, item in visible:
                issues = quality_issues(item, st.session_state.questions)
                icon = "⚠️" if issues else "✅"
                with st.expander(f"{icon} {index + 1}. {item['topic']} · уровень {item['level']} · {item['kind']}"):
                    item["question"] = st.text_area("Вопрос", item["question"], key=f"q_{index}")
                    if item["options"]:
                        updated_options = []
                        for option_index, option in enumerate(item["options"]):
                            updated_options.append(st.text_input(f"Вариант {option_index + 1}", option, key=f"o_{index}_{option_index}"))
                        item["options"] = updated_options
                    item["answer"] = st.text_area("Правильный ответ", item["answer"], key=f"a_{index}")
                    item["explanation"] = st.text_area("Объяснение", item["explanation"], key=f"e_{index}")
                    st.markdown(f'<div class="source"><b>Источник:</b> {item["source"]}</div>', unsafe_allow_html=True)
                    if issues:
                        st.warning(" · ".join(issues))
                    if st.button("Удалить", key=f"delete_{index}"):
                        st.session_state.questions.pop(index)
                        st.rerun()
            export_col1, export_col2 = st.columns(2)
            payload = json.dumps(st.session_state.questions, ensure_ascii=False, indent=2)
            export_col1.download_button("Скачать JSON", payload, "questions.json", "application/json", use_container_width=True)
            export_col2.download_button("Скачать CSV", export_csv(st.session_state.questions), "questions.csv", "text/csv", use_container_width=True)
    with tab_quality:
        if not st.session_state.questions:
            st.info("Сначала создайте вопросы.")
        elif bad_count == 0:
            st.success("Все вопросы прошли базовую проверку.")
        else:
            st.warning(f"Найдено вопросов для проверки: {bad_count}")
            for index, item in enumerate(st.session_state.questions):
                issues = quality_issues(item, st.session_state.questions)
                if issues:
                    st.write(f"**{index + 1}. {item['question']}** — {'; '.join(issues)}")
    with tab_quiz:
        quiz_items = st.session_state.questions
        if not quiz_items:
            st.info("Сначала создайте вопросы, чтобы пройти мини-тест.")
        else:
            with st.form("quiz"):
                answers: list[str] = []
                for index, item in enumerate(quiz_items):
                    st.markdown(f"**{index + 1}. {item['question']}**")
                    if item["kind"] == "Тестовый" and len(item["options"]) == 4:
                        answer = st.radio("Выберите ответ", item["options"], key=f"quiz_{index}", label_visibility="collapsed")
                    else:
                        answer = st.text_area(
                            "Ваш ответ",
                            key=f"quiz_{index}",
                            placeholder="Напишите ответ своими словами...",
                            label_visibility="collapsed",
                        )
                    answers.append(answer or "")
                submitted = st.form_submit_button("Проверить ответы", type="primary")
            if submitted:
                result_items = []
                score = 0
                max_score = 0
                for answer, item in zip(answers, quiz_items):
                    if item["kind"] == "Тестовый" and len(item["options"]) == 4:
                        correct = answer == item["answer"]
                        score += int(correct)
                        max_score += 1
                        result_items.append(("Тестовый", correct, "Правильно" if correct else f"Правильный ответ: {item['answer']}"))
                    else:
                        feedback, points = evaluate_open_answer(answer, item["answer"])
                        score += points
                        max_score += 100
                        result_items.append(("Открытый", points >= 55, f"{feedback} Баллы: {points}/100"))
                st.session_state.quiz_result = {"score": score, "max_score": max_score, "items": result_items}
            if st.session_state.quiz_result:
                result = st.session_state.quiz_result
                percent = result["score"] / max(result["max_score"], 1) * 100
                st.success(f"Результат: {percent:.0f}%")
                for index, (_, passed, feedback) in enumerate(result["items"], start=1):
                    if passed:
                        st.success(f"{index}. {feedback}")
                    else:
                        st.warning(f"{index}. {feedback}")


if __name__ == "__main__":
    main()
