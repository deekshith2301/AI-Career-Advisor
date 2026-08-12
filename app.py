import os
import requests
import streamlit as st

from dotenv import load_dotenv
from pypdf import PdfReader
from typing import TypedDict, Annotated

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found")
    st.stop()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

st.set_page_config(
    page_title="AI Career Advisor",
    page_icon="🤖",
    layout="wide"
)

st.title("AI Career Advisor")

uploaded_file = st.file_uploader(
    "Upload Resume",
    type=["pdf"]
)

resume_text = ""

if uploaded_file:

    reader = PdfReader(uploaded_file)

    for page in reader.pages:
        resume_text += page.extract_text() or ""

    st.success("Resume uploaded successfully")

target_role = st.text_input(
    "Target Role",
    placeholder="AI Engineer"
)

github_username = st.text_input(
    "GitHub Username",
    placeholder="deekshith2301"
)


def content_to_text(content):

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        text = []

        for item in content:

            if isinstance(item, dict):

                if item.get("type") == "text":
                    text.append(item["text"])

        return "\n".join(text)

    return str(content)


class AgentState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]
    resume_text: str
    target_role: str
    github_username: str
    project_ideas: str


@tool
def analyze_resume(resume_text, target_role):
    """Analyze the candidate's resume."""

    prompt = f"""
You are an expert AI Career Advisor.

Analyze the resume for the target role.

Resume:
{resume_text}

Target Role:
{target_role}

Provide:

1. Candidate Summary
2. Technical Skills
3. Projects
4. Strengths
5. Weaknesses
6. Suitability Score (/10)

Return the answer in Markdown.
"""
    result = llm.invoke(prompt)
    return content_to_text(result.content)

@tool
def skill_gap_analysis(resume_text, target_role):
    """Analyze the candidate's skill gaps."""

    prompt = f"""
Compare this resume with the target role.

Resume:
{resume_text}

Target Role:
{target_role}

Provide:

1. Existing Skills
2. Missing Skills
3. Areas to Improve
4. Learning Roadmap
"""

    result = llm.invoke(prompt)
    return content_to_text(result.content)

@tool
def github_analysis(github_username, target_role):
    """Analyze the candidate's GitHub profile."""

    if github_username.strip() == "":
        return "GitHub username not provided."

    url = f"https://api.github.com/users/{github_username}/repos"

    try:
        response = requests.get(url, timeout=10)
    except Exception as e:
        return f"GitHub Error: {e}"

    if response.status_code != 200:
        return "GitHub profile not found."

    repos = response.json()

    if not repos:
        return "No public repositories found."

    repo_info = []

    for repo in repos:
        repo_info.append({
            "name": repo.get("name"),
            "description": repo.get("description"),
            "language": repo.get("language"),
            "stars": repo.get("stargazers_count"),
            "forks": repo.get("forks_count")
        })

    prompt = f"""
Analyze this GitHub profile.

GitHub:
{github_username}

Target Role:
{target_role}

Repositories:
{repo_info}

Provide:

1. Technologies Used
2. Strengths
3. Weaknesses
4. Suggestions
"""

    result = llm.invoke(prompt)
    return content_to_text(result.content)
@tool
def project_ideas(resume_text, target_role):
    """Suggest projects based on the candidate's resume."""

    prompt = f"""
Suggest 5 practical projects.

Resume:
{resume_text}

Target Role:
{target_role}

For each project provide:

1. Project Name
2. Description
3. Technologies
4. Features
5. Difficulty
"""

    result = llm.invoke(prompt)
    return content_to_text(result.content)

tools = [
    analyze_resume,
    skill_gap_analysis,
    github_analysis
]

llm_with_tools = llm.bind_tools(tools)

tool_node = ToolNode(tools)


def agent_node(state):

    response = llm_with_tools.invoke(state["messages"])

    return {
        "messages": [response]
    }


def project_ideas_node(state):

    result = project_ideas.invoke({
        "resume_text": state["resume_text"],
        "target_role": state["target_role"]
    })

    return {
        "project_ideas": result
    }


def final_synthesis_node(state):

    tool_output = []

    for message in state["messages"]:

        if type(message).__name__ == "ToolMessage":
            tool_output.append(content_to_text(message.content))

    prompt = f"""

You are an expert AI Career Advisor.

Create one final report.

Resume

{state["resume_text"]}

Target Role

{state["target_role"]}

Tool Analysis

{"".join(tool_output)}

Recommended Projects

{state["project_ideas"]}

Generate a professional report with:

1. Candidate Summary

2. Resume Evaluation

3. Technical Skills

4. Skill Gap Analysis

5. GitHub Evaluation

6. Recommended Projects

7. Overall Score (/10)

8. Learning Roadmap

9. Final Suggestions

Return everything in Markdown.

"""

    response = llm.invoke(prompt)

    return {
        "messages": [response]
    }


graph_builder = StateGraph(AgentState)

graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("project_ideas", project_ideas_node)
graph_builder.add_node("final_synthesis", final_synthesis_node)

graph_builder.add_edge(START, "agent")

graph_builder.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "tools",
        END: "project_ideas"
    }
)

graph_builder.add_edge("tools", "agent")
graph_builder.add_edge("project_ideas", "final_synthesis")
graph_builder.add_edge("final_synthesis", END)

graph = graph_builder.compile()
if st.button("Analyze Candidate"):

    if not uploaded_file:
        st.warning("Please upload your resume.")
        st.stop()

    if target_role.strip() == "":
        st.warning("Please enter the target role.")
        st.stop()

    with st.spinner("Analyzing Resume..."):

        initial_state = {
            "messages": [
                HumanMessage(
                    content=f"""
Analyze this candidate.

Resume

{resume_text}

Target Role

{target_role}

GitHub Username

{github_username}

Use the available tools to complete the analysis.
"""
                )
            ],
            "resume_text": resume_text,
            "target_role": target_role,
            "github_username": github_username,
            "project_ideas": ""
        }

        result = graph.invoke(initial_state)

        final_report = content_to_text(
            result["messages"][-1].content
        )

    st.success("Analysis Completed")

    st.markdown(final_report)