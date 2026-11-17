import csv
import io
import secrets
import re
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

from app import (
    LEVELS,
    ai_generate,
    db_save_questions,
    detect_topics,
    evaluate_open_answer,
    extract_document,
    generate_full_set,
    quality_issues,
    source_for_topic,
)


app = Flask(__name__, template_folder="templates", static_folder="static")
projects: dict[str, dict] = {}


class UploadAdapter:
    def __init__(self, uploaded):
        self.name = uploaded.filename
        self._data = uploaded.read()

    def getvalue(self):
        return self._data


def project_or_404(project_id: str) -> dict:
    project = projects.get(project_id)
    if project is None:
        raise KeyError("Проект не найден.")
    return project


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/upload")
def upload():
    uploaded = request.files.get("document")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "Выберите PDF или TXT-файл."}), 400
    try:
        text, pages = extract_document(UploadAdapter(uploaded))
    except (ValueError, RuntimeError) as error:
        return jsonify({"error": str(error)}), 400
    project_id = secrets.token_urlsafe(12)
    project = {
        "id": project_id,
        "name": uploaded.filename,
        "text": text,
        "pages": pages,
        "topics": detect_topics(text),
        "questions": [],
        "attempts": [],
    }
    projects[project_id] = project
    return jsonify({
        "id": project_id,
        "name": project["name"],
        "topics": project["topics"],
        "characters": len(text),
        "pages": len(pages),
    })


@app.get("/api/analysis/<project_id>")
def analysis(project_id: str):
    try:
        project = project_or_404(project_id)
    except KeyError as error:
        return jsonify({"error": str(error)}), 404
    sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", project["text"]).strip())
    useful = [sentence.strip() for sentence in sentences if len(sentence.strip()) > 45]
    definitions = [sentence for sentence in useful if any(mark in sentence.lower() for mark in ("это", "означает", "является", "называется"))][:5]
    terms = [topic for topic in project["topics"][:6]]
    topic_cards = [
        {
            "name": topic,
            "description": source_for_topic(topic, project["text"])[:180],
            "questions": sum(item["topic"] == topic for item in project["questions"]),
            "mastery": 0,
            "accuracy": 0,
        }
        for topic in project["topics"]
    ]
    nodes = [{"id": topic, "group": "main"} for topic in project["topics"]]
    edges = [{"source": project["topics"][index], "target": project["topics"][index + 1]} for index in range(len(project["topics"]) - 1)]
    return jsonify({
        "overview": {
            "name": project["name"],
            "pages": len(project["pages"]),
            "words": len(re.findall(r"\w+", project["text"], flags=re.UNICODE)),
            "topics": len(project["topics"]),
            "status": "Ready",
        },
        "summary": {
            "main_ideas": useful[:3],
            "definitions": definitions,
            "terms": terms,
            "chapter_summary": useful[:8],
        },
        "topics": topic_cards,
        "graph": {"nodes": nodes, "edges": edges},
    })


@app.post("/api/chat")
def chat():
    data = request.get_json(force=True)
    try:
        project = project_or_404(data["project_id"])
    except KeyError as error:
        return jsonify({"error": str(error)}), 404
    question = str(data.get("question", "")).strip()
    if not question:
        return jsonify({"error": "Введите вопрос."}), 400
    terms = [word.lower() for word in re.findall(r"[А-Яа-яЁёA-Za-z]{4,}", question)]
    chunks = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", project["text"]).strip())
    matches = [chunk for chunk in chunks if any(term in chunk.lower() for term in terms)]
    source = " ".join(matches[:3]) if matches else ""
    if source:
        answer = f"В загруженном документе сказано: {source[:900]}"
        return jsonify({"answer": answer, "source": "Фрагмент загруженного документа"})
    return jsonify({"answer": "В загруженном документе не найдено информации по этому вопросу.", "source": "Источник не найден"})


@app.post("/api/flashcards/<project_id>")
def flashcards(project_id: str):
    try:
        project = project_or_404(project_id)
    except KeyError as error:
        return jsonify({"error": str(error)}), 404
    cards = []
    for topic in project["topics"][:10]:
        source = source_for_topic(topic, project["text"])
        cards.append({
            "front": topic,
            "back": source,
            "source": "Загруженный документ",
            "status": "Нужно повторить",
        })
    return jsonify({"cards": cards})


@app.post("/api/quiz/submit")
def submit_quiz():
    data = request.get_json(force=True)
    try:
        project = project_or_404(data["project_id"])
    except KeyError as error:
        return jsonify({"error": str(error)}), 404
    answers = data.get("answers", [])
    items = data.get("questions", [])
    correct = sum(answer == item.get("answer") for answer, item in zip(answers, items))
    total = len(items)
    attempt = {"score": correct, "total": total, "accuracy": round(correct / total * 100) if total else 0}
    project["attempts"].append(attempt)
    return jsonify(attempt)


@app.post("/api/generate")
def generate():
    data = request.get_json(force=True)
    try:
        project = project_or_404(data["project_id"])
        topic = str(data["topic"])
        amount = max(1, min(int(data.get("amount", 5)), 10))
        level = max(1, min(int(data.get("level", 3)), 5))
        kind = data.get("kind", "Открытый")
        if data.get("full_set"):
            questions = generate_full_set(topic, project["text"], project["pages"], kind, min(amount, 5))
        else:
            questions = ai_generate(topic, amount, level, kind, project["text"], project["pages"])
        project["questions"].extend(questions)
        return jsonify({"questions": project["questions"]})
    except (KeyError, ValueError, TypeError) as error:
        return jsonify({"error": str(error)}), 400
    except Exception as error:
        return jsonify({"error": f"Не удалось создать вопросы: {error}"}), 500


@app.patch("/api/questions/<project_id>/<int:index>")
def update_question(project_id: str, index: int):
    try:
        project = project_or_404(project_id)
        item = project["questions"][index]
        item.update(request.get_json(force=True))
        return jsonify(item)
    except (KeyError, IndexError) as error:
        return jsonify({"error": str(error)}), 404


@app.delete("/api/questions/<project_id>/<int:index>")
def delete_question(project_id: str, index: int):
    try:
        project = project_or_404(project_id)
        project["questions"].pop(index)
        return jsonify({"questions": project["questions"]})
    except (KeyError, IndexError) as error:
        return jsonify({"error": str(error)}), 404


@app.post("/api/save/<project_id>")
def save_questions(project_id: str):
    try:
        project = project_or_404(project_id)
        db_save_questions(project["name"], project["questions"])
        return jsonify({"saved": len(project["questions"])})
    except KeyError as error:
        return jsonify({"error": str(error)}), 404


@app.post("/api/check")
def check_answer():
    data = request.get_json(force=True)
    feedback, score = evaluate_open_answer(data.get("answer", ""), data.get("expected", ""))
    return jsonify({"feedback": feedback, "score": score})


@app.get("/api/export/<project_id>/<format_name>")
def export_questions(project_id: str, format_name: str):
    try:
        project = project_or_404(project_id)
    except KeyError as error:
        return jsonify({"error": str(error)}), 404
    if format_name == "json":
        import json
        data = json.dumps(project["questions"], ensure_ascii=False, indent=2).encode("utf-8")
        return send_file(io.BytesIO(data), as_attachment=True, download_name="questions.json", mimetype="application/json")
    if format_name == "csv":
        output = io.StringIO()
        fields = ["question", "answer", "explanation", "topic", "level", "kind", "options", "source"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for item in project["questions"]:
            row = dict(item)
            row["options"] = " | ".join(row.get("options", []))
            writer.writerow({field: row.get(field, "") for field in fields})
        return send_file(
            io.BytesIO(output.getvalue().encode("utf-8-sig")),
            as_attachment=True,
            download_name="questions.csv",
            mimetype="text/csv",
        )
    return jsonify({"error": "Неизвестный формат."}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
