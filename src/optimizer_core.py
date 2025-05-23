import pdfplumber
import docx
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph


llm = ChatOpenAI(
    openai_api_base="https://chatapi.akash.network/api/v1",
    openai_api_key="sk-wsSXBy9BTQgVlWzdba2HDw",
    model="Meta-Llama-4-Maverick-17B-128E-Instruct-FP8",
)


def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return "\n".join([page.extract_text() or "" for page in pdf.pages])


def extract_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])


def extract_bullets(text):
    return "\n".join(
        [line for line in text.splitlines() if line.strip().startswith("-")]
    )


def extract_intro(text):
    return "\n".join(
        [line for line in text.splitlines() if "objective" in line.lower()]
    )


# --- LangChain Nodes ---
def parse_resume(text):
    return {
        "bullet_points": extract_bullets(text),
        "intro_objectives": extract_intro(text),
    }


analyze_bullets = (
    PromptTemplate.from_template(
        "Evaluate and enhance the following resume bullet points:\n{bullets}"
    )
    | llm
)

analyze_intro = (
    PromptTemplate.from_template(
        "Evaluate and improve this resume introduction and career objectives:\n{intro}"
    )
    | llm
)

summarize_bullets = (
    PromptTemplate.from_template(
        "Summarize the key improvements made to the bullet points:\n{analysis}"
    )
    | llm
)

summarize_intro = (
    PromptTemplate.from_template(
        "Summarize the enhancements made to the introduction and objectives:\n{analysis}"
    )
    | llm
)

generate_resume = (
    PromptTemplate.from_template(
        """Create a professional resume using the following improved content:

Bullet Points:
{summary_bullets}

Introduction & Objectives:
{summary_intro}
"""
    )
    | llm
)


def build_optimizer_graph():
    builder = StateGraph()
    builder.add_node("parse", RunnableLambda(parse_resume))
    builder.add_node(
        "analyze_bullets",
        RunnableLambda(
            lambda d: analyze_bullets.invoke({"bullets": d["bullet_points"]})
        ),
    )
    builder.add_node(
        "analyze_intro",
        RunnableLambda(
            lambda d: analyze_intro.invoke({"intro": d["intro_objectives"]})
        ),
    )
    builder.add_node(
        "summarize_bullets",
        RunnableLambda(
            lambda d: summarize_bullets.invoke({"analysis": d["analyze_bullets"]})
        ),
    )
    builder.add_node(
        "summarize_intro",
        RunnableLambda(
            lambda d: summarize_intro.invoke({"analysis": d["analyze_intro"]})
        ),
    )
    builder.add_node(
        "generate_resume",
        RunnableLambda(
            lambda d: generate_resume.invoke(
                {
                    "summary_bullets": d["summarize_bullets"],
                    "summary_intro": d["summarize_intro"],
                }
            )
        ),
    )

    builder.set_entry_point("parse")
    builder.connect("parse", "analyze_bullets")
    builder.connect("parse", "analyze_intro")
    builder.connect("analyze_bullets", "summarize_bullets")
    builder.connect("analyze_intro", "summarize_intro")
    builder.connect("summarize_bullets", "generate_resume")
    builder.connect("summarize_intro", "generate_resume")
    builder.set_finish_point("generate_resume")

    return builder.compile()
