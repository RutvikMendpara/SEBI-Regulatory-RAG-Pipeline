# ============================================================
# RegSage - LangGraph version ("corrective RAG")
#
# Graph:
#   START -> retrieve -> [grade: are the chunks relevant?]
#                          yes (or tried twice) -> generate -> END
#                          no                   -> rewrite query -> retrieve (loop)
#
# Why a graph? A plain chain runs once, top to bottom. A graph can branch and
# loop, keeps state between steps, and caps the loop (max 2 tries) so it can't run forever.
#
# Run:  python graph.py
# ============================================================

from typing import TypedDict

from langgraph.graph import StateGraph, START, END

# Reuse everything from rag.py: the index, the LLM, and the answer function.
from rag import llm, retrieve, format_context, answerer, SYSTEM, print_answer, Answer

MAX_TRIES = 2


# 1. STATE: the shared "notebook" every node reads from and writes to.
class State(TypedDict, total=False):
    question: str        # current search question (may get rewritten)
    original: str        # what the user actually asked
    docs: list           # retrieved chunks
    tries: int           # how many times we've retrieved
    answer: Answer       # final answer


# 2. NODE: retrieve chunks for the current question.
def retrieve_node(state: State) -> dict:
    tries = state.get("tries", 0) + 1
    print(f"  [retrieve] try {tries}: {state['question']!r}")
    return {"docs": retrieve(state["question"]), "tries": tries}


# 3. ROUTER: ask the LLM whether the chunks can answer the question.
#    Returns the NAME of the next node.
def grade_route(state: State) -> str:
    passages = "\n---\n".join(d.page_content[:500] for d in state["docs"])
    verdict = llm.invoke(
        f"Question: {state['original']}\n\nPassages:\n{passages}\n\n"
        "Do these passages contain the information needed to answer the question? "
        "Reply with only yes or no."
    ).content
    relevant = str(verdict).strip().lower().startswith("yes")
    print(f"  [grade] relevant={relevant}")
    if relevant or state["tries"] >= MAX_TRIES:
        return "generate"
    return "rewrite"


# 4. NODE: rewrite a vague question into a better search query.
def rewrite_node(state: State) -> dict:
    new_q = llm.invoke(
        "Rewrite this as a precise search query for SEBI regulation documents "
        "(use regulatory terms). Return only the query.\n\n"
        f"Question: {state['original']}"
    ).content
    return {"question": str(new_q).strip()}


# 5. NODE: generate the cited answer from the retrieved chunks.
#    Note: we answer the ORIGINAL question, using chunks found with the rewritten one.
def generate_node(state: State) -> dict:
    context = format_context(state["docs"])
    result = answerer.invoke([
        ("system", SYSTEM),
        ("human", f"<context>\n{context}\n</context>\n\nQuestion: {state['original']}"),
    ])
    return {"answer": result}


# 6. WIRE THE GRAPH: nodes + edges.
builder = StateGraph(State)
builder.add_node("retrieve", retrieve_node)
builder.add_node("rewrite", rewrite_node)
builder.add_node("generate", generate_node)

builder.add_edge(START, "retrieve")
builder.add_conditional_edges("retrieve", grade_route, {"generate": "generate", "rewrite": "rewrite"})
builder.add_edge("rewrite", "retrieve")   # the loop
builder.add_edge("generate", END)

app = builder.compile()


# 7. Chat loop. Try a vague question like "insider rules for staff" to see the rewrite loop.
if __name__ == "__main__":
    while True:
        q = input("\nAsk (graph) about SEBI regs (or 'exit'): ").strip()
        if not q:
            continue
        if q.lower() == "exit":
            break
        final = app.invoke({"question": q, "original": q})
        print_answer(final["answer"])