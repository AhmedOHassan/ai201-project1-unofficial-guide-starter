"""
Milestone 5 — Query interface (Gradio web UI).

A user types a plain-language question about NC State off-campus housing; the
app retrieves relevant student-review chunks, generates a grounded answer with
Groq, and shows the answer alongside the source threads it drew from and the
raw retrieved chunks (so a viewer can see the grounding).

Run:
    python app.py
Then open http://localhost:7860
"""

import gradio as gr

from query import ask

EXAMPLES = [
    "Which apartment complex do students say tows or damages cars?",
    "What are the hidden fees and parking costs at the budget complexes?",
    "Is there a safety concern near Avent Ferry and Socket Dr?",
    "What's the window/fire-safety problem at The Standard?",
    "What infrastructure problems do reviewers report at Valentine Commons?",
]


def handle_query(question: str):
    question = (question or "").strip()
    if not question:
        return "Please enter a question.", "", ""

    result = ask(question)

    answer = result["answer"]
    sources = (
        "\n".join(f"• {s['source']}  ({s['url']})" for s in result["sources"])
        or "— (no sources — the documents didn't cover this)"
    )
    retrieved = "\n\n".join(
        f"[{i}] {r.source}  (distance {r.distance:.3f})\n"
        f"{r.text.strip()[:400]}{'…' if len(r.text) > 400 else ''}"
        for i, r in enumerate(result["results"], 1)
    )
    return answer, sources, retrieved


with gr.Blocks(title="The Unofficial Guide — NC State Housing") as demo:
    gr.Markdown(
        "# The Unofficial Guide — NC State Off-Campus Housing\n"
        "Ask about apartments near NCSU. Answers are grounded **only** in real "
        "r/NCSU review threads, with the sources shown."
    )
    question = gr.Textbox(
        label="Your question",
        placeholder="e.g. Which complex tows cars? What does parking cost?",
    )
    ask_btn = gr.Button("Ask", variant="primary")
    answer = gr.Textbox(label="Answer", lines=6)
    sources = gr.Textbox(label="Sources (retrieved threads)", lines=4)
    with gr.Accordion("Retrieved chunks (the grounding)", open=False):
        retrieved = gr.Textbox(label="Top retrieved chunks", lines=12)

    gr.Examples(examples=EXAMPLES, inputs=question)

    ask_btn.click(handle_query, inputs=question,
                  outputs=[answer, sources, retrieved])
    question.submit(handle_query, inputs=question,
                    outputs=[answer, sources, retrieved])


if __name__ == "__main__":
    demo.launch()
