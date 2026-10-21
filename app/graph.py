# import os
# from typing import TypedDict, List
# from langgraph.graph import StateGraph, END
# from langchain_pinecone import PineconeVectorStore
# from tavily import TavilyClient
# from app.llm import get_llm
# from app.ingest import get_embeddings, INDEX

# llm = get_llm()
# vs = PineconeVectorStore(index_name=INDEX, embedding=get_embeddings())
# tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# class State(TypedDict):
#     question: str
#     context: List[str]
#     sources: List[str]
#     relevant: bool
#     answer: str
#     route: str

# def retrieve(state: State):
#     docs = vs.similarity_search(state["question"], k=4)
#     return {
#         "context": [d.page_content for d in docs],
#         "sources": [f"Gale Encyclopedia, p.{d.metadata.get('page', '?')}" for d in docs],
#         "route": "document",
#     }

# def grade(state: State):
#     prompt = (
#         "Does the context below contain information that helps answer the question? "
#         "Reply only YES or NO.\n\n"
#         f"Question: {state['question']}\n\nContext:\n" + "\n---\n".join(state["context"])
#     )
#     verdict = llm.invoke(prompt).content.strip().upper()
#     return {"relevant": verdict.startswith("YES")}

# def web_search(state: State):
#     res = tavily.search(state["question"], max_results=3)["results"]
#     return {
#         "context": [r["content"] for r in res],
#         "sources": [r["url"] for r in res],
#          "route": "web",
#     }

# def generate(state: State):
#     system = (
#         "You are MedGuide AI, an educational medical assistant. Answer ONLY from the "
#         "provided context. Be clear and concise. You are not a doctor: do not diagnose, "
#         "and advise seeing a healthcare professional for personal medical concerns. "
#         "For emergencies (chest pain, stroke signs, severe bleeding, etc.), tell the "
#         "user to call local emergency services immediately. If the context is "
#         "insufficient, say so."
#     )
#     prompt = f"{system}\n\nContext:\n" + "\n---\n".join(state["context"]) + \
#              f"\n\nQuestion: {state['question']}"
#     return {"answer": llm.invoke(prompt).content}

# def route(state: State):
#     return "generate" if state["relevant"] else "web_search"

# g = StateGraph(State)
# g.add_node("retrieve", retrieve)
# g.add_node("grade", grade)
# g.add_node("web_search", web_search)
# g.add_node("generate", generate)

# g.set_entry_point("retrieve")
# g.add_edge("retrieve", "grade")
# g.add_conditional_edges("grade", route, {"generate": "generate", "web_search": "web_search"})
# g.add_edge("web_search", "generate")
# g.add_edge("generate", END)

# agent = g.compile()



import os
from typing import TypedDict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_pinecone import PineconeVectorStore
from tavily import TavilyClient
from app.llm import get_llm
from app.ingest import get_embeddings, INDEX

load_dotenv()

# Cosine similarity thresholds for the all-MiniLM-L6-v2 embeddings.
# Below MIN_SCORE -> go to the web. Above STRONG_SCORE -> trust the PDF.
# In between -> ask the LLM. Tune them with check_index.py.
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.30"))
STRONG_SCORE = float(os.getenv("STRONG_SCORE", "0.50"))

llm = get_llm()
vs = PineconeVectorStore(index_name=INDEX, embedding=get_embeddings())
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


class State(TypedDict):
    question: str
    context: List[str]
    sources: List[str]
    relevant: bool
    answer: str
    route: str
    best_score: float


def retrieve(state: State):
    results = vs.similarity_search_with_score(state["question"], k=5)
    context, sources, best = [], [], 0.0
    for doc, score in results:
        best = max(best, score)
        context.append(doc.page_content)
        book = os.path.basename(str(doc.metadata.get("source", "Medical book")))
        page = doc.metadata.get("page")
        sources.append(f"{book}, p.{int(page) + 1}" if page is not None else book)
    print(f"[retrieve] chunks={len(results)} best_score={best:.3f}")
    return {
        "context": context,
        "sources": sources,
        "route": "document",
        "best_score": best,
    }


def grade(state: State):
    best = state["best_score"]
    if not state["context"] or best < MIN_SCORE:
        relevant = False
    elif best >= STRONG_SCORE:
        relevant = True
    else:
        prompt = (
            "Does the context below contain information that helps answer the question? "
            "Reply with the single word YES or NO.\n\n"
            f"Question: {state['question']}\n\nContext:\n"
            + "\n---\n".join(state["context"])[:3000]
        )
        verdict = llm.invoke(prompt).content.upper()
        relevant = "YES" in verdict
    print(f"[grade] best_score={best:.3f} relevant={relevant}")
    return {"relevant": relevant}


def web_search(state: State):
    try:
        res = tavily.search(state["question"], max_results=3)["results"]
    except Exception as e:
        print(f"[web_search] failed: {e}")
        return {}
    print(f"[web_search] {len(res)} results")
    return {
        "context": [r["content"] for r in res],
        "sources": [r["url"] for r in res],
        "route": "web",
    }


def generate(state: State):
    system = (
        "You are MedGuide AI, an educational medical assistant. Answer ONLY from the "
        "provided context. Be clear and concise. You are not a doctor: do not diagnose, "
        "and advise seeing a healthcare professional for personal medical concerns. "
        "For emergencies (chest pain, stroke signs, severe bleeding, etc.), tell the "
        "user to call local emergency services immediately. If the context is "
        "insufficient, say so."
    )
    prompt = (
        f"{system}\n\nContext:\n" + "\n---\n".join(state["context"])
        + f"\n\nQuestion: {state['question']}"
    )
    return {"answer": llm.invoke(prompt).content}


def route(state: State):
    return "generate" if state["relevant"] else "web_search"


g = StateGraph(State)
g.add_node("retrieve", retrieve)
g.add_node("grade", grade)
g.add_node("web_search", web_search)
g.add_node("generate", generate)

g.set_entry_point("retrieve")
g.add_edge("retrieve", "grade")
g.add_conditional_edges("grade", route, {"generate": "generate", "web_search": "web_search"})
g.add_edge("web_search", "generate")
g.add_edge("generate", END)

agent = g.compile()