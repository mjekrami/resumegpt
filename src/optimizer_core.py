from typing_extensions import TypedDict, Annotated
from operator import add
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END


class ResumeState(TypedDict):
    bullet_points: Annotated[str, add]
    intro_objectives: Annotated[str, add]
    analyze_bullets: Annotated[str, add]
    analyze_intro: Annotated[str, add]
    summarize_bullets: Annotated[str, add]
    summarize_intro: Annotated[str, add]
    generate_resume: Annotated[str, add]


llm = ChatOpenAI(
    openai_api_base="https://chatapi.akash.network/api/v1",
    openai_api_key="sk-wsSXBy9BTQgVlWzdba2HDw",
    model="Meta-Llama-4-Maverick-17B-128E-Instruct-FP8",
)


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
    builder = StateGraph(ResumeState)
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
    builder.add_edge("parse", "analyze_bullets")
    builder.add_edge("parse", "analyze_intro")
    builder.add_edge("analyze_bullets", "summarize_bullets")
    builder.add_edge("analyze_intro", "summarize_intro")
    builder.add_edge("summarize_bullets", "generate_resume")
    builder.add_edge("summarize_intro", "generate_resume")
    builder.add_edge("generate_resume", END)

    return builder.compile()
