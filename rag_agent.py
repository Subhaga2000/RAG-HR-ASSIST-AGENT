from dotenv import load_dotenv
import os
import smtplib
from email.mime.text import MIMEText
from typing import TypedDict, Annotated, Sequence, Optional, Dict
from operator import add as add_messages
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool


load_dotenv()

# -----------------------------
# GLOBAL EMAIL DRAFT STORAGE
# -----------------------------
pending_email_draft: Optional[Dict[str, str]] = None

# -----------------------------
# LLM
# -----------------------------
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7
)

# -----------------------------
# EMBEDDINGS
# -----------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# -----------------------------
# PDF PATH
# -----------------------------
pdf_path = "C:/Users/subha/OneDrive/Desktop/RAG-NEW-AGENT/hr_manual.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF not found at {pdf_path}")

# -----------------------------
# LOAD PDF
# -----------------------------
loader = PyPDFLoader(pdf_path)
pages = loader.load()
print(f"PDF loaded successfully with {len(pages)} pages.")

# -----------------------------
# TEXT SPLITTING
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=300
)

docs = text_splitter.split_documents(pages)

# -----------------------------
# VECTOR DB
# -----------------------------
persist_directory = "C:/Users/subha/OneDrive/Desktop/RAG-NEW-AGENT/vector_db"
collection_name = "hr_manual"

if os.path.exists(persist_directory) and os.listdir(persist_directory):
    print("Loading existing vector database...")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name
    )
else:
    print("Creating new vector database...")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )

print("Vector database ready.")

# -----------------------------
# RETRIEVER
# -----------------------------
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 8}
)

# -----------------------------
# TOOL 1: RETRIEVER
# -----------------------------
@tool
def retriever_tool(query: str) -> str:
    """Search HR manual for relevant information."""

    q = query.lower()
    expanded_query = query

    # Improve retrieval for onboarding / welcome / new employee prompts
    if any(word in q for word in ["welcome", "onboarding", "new employee", "selected employee", "joined", "joining"]):
        expanded_query += " induction orientation commencement of employment pre-employment formalities personal data dependents official address probation"

    docs_found = retriever.invoke(expanded_query)

    if not docs_found:
        return "No relevant information found in the HR manual."

    results = []
    for doc in docs_found:
        page = doc.metadata.get("page", "unknown")
        results.append(f"[HR Manual - Page {page}]\n{doc.page_content}")

    return "\n\n".join(results)

# -----------------------------
# TOOL 2: CREATE EMAIL DRAFT
# -----------------------------
@tool
def create_email_draft_tool(employee_email: str, subject: str, body: str) -> str:
    """
    Create an email draft and hold it for HR Head approval before sending.
    """
    global pending_email_draft

    pending_email_draft = {
        "employee_email": employee_email,
        "subject": subject,
        "body": body
    }

    return f"Email draft created for {employee_email} and is waiting for HR Head approval."

# -----------------------------
# REAL EMAIL SENDER
# NOT EXPOSED TO LLM
# -----------------------------
def send_email_now(employee_email: str, subject: str, body: str) -> str:
    sender_email = (os.getenv("EMAIL_ADDRESS") or "").strip()
    sender_password = (os.getenv("EMAIL_PASSWORD") or "").strip().replace(" ", "")

    if not sender_email or not sender_password:
        return "Email credentials not configured in .env file."

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = employee_email

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, employee_email, msg.as_string())
        return f"Email sent successfully to {employee_email}"

    except smtplib.SMTPAuthenticationError:
        return (
            "Gmail authentication failed. Use the Gmail account's 16-digit App Password "
            "in EMAIL_PASSWORD, make sure 2-Step Verification is enabled, and regenerate "
            "the App Password if you changed the Google password."
        )
    except Exception as e:
        return f"Error sending email: {str(e)}"

print("EMAIL_ADDRESS =", os.getenv("EMAIL_ADDRESS"))
print("EMAIL_PASSWORD loaded =", bool(os.getenv("EMAIL_PASSWORD")))

# -----------------------------
# ONLY THESE TOOLS ARE GIVEN TO LLM
# -----------------------------
tools = [retriever_tool, create_email_draft_tool]
llm = llm.bind_tools(tools)

tools_dict = {tool.name: tool for tool in tools}

# -----------------------------
# AGENT STATE
# -----------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# -----------------------------
# SHOULD CONTINUE
# -----------------------------
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    return hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0

# -----------------------------
# SYSTEM PROMPT
# -----------------------------
system_prompt = """
You are an HR assistant that answers questions using ONLY the HR manual.

Rules:
1. Always call retriever_tool before answering HR questions.
2. Use ONLY the information returned from the HR manual.
3. Do not add policies that are not present in the retrieved text.
4. If the user asks for an email, first retrieve the relevant HR policy and then create a professional email draft.
5. For email requests, call create_email_draft_tool with:
   - employee_email
   - subject
   - body
6. Never claim that the email has been sent.
7. The email will only be sent after HR Head approval.
8. Cite HR manual page numbers in the final answer where relevant.
9. If the information is not found, say:
   "This information is not available in the HR manual."
"""

# -----------------------------
# LLM NODE
# -----------------------------
def call_llm(state: AgentState) -> AgentState:
    messages = list(state["messages"])
    messages = [SystemMessage(content=system_prompt)] + messages
    response = llm.invoke(messages)
    return {"messages": [response]}

# -----------------------------
# TOOL EXECUTION NODE
# -----------------------------
def take_action(state: AgentState) -> AgentState:
    tool_calls = state["messages"][-1].tool_calls
    tool_results = []

    for t in tool_calls:
        tool_name = t["name"]
        tool_args = t["args"]

        print(f"\nCalling Tool: {tool_name}")
        print(f"Arguments: {tool_args}")

        if tool_name not in tools_dict:
            tool_output = "Invalid tool."
        else:
            tool_output = tools_dict[tool_name].invoke(tool_args)

        tool_results.append(
            ToolMessage(
                tool_call_id=t["id"],
                content=str(tool_output)
            )
        )

    print("Tool execution complete.")
    return {"messages": tool_results}

# -----------------------------
# BUILD GRAPH
# -----------------------------
graph = StateGraph(AgentState)

graph.add_node("llm", call_llm)
graph.add_node("tool_node", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "tool_node", False: END}
)

graph.add_edge("tool_node", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()

# -----------------------------
# APPROVAL + SEND PROCESS
# -----------------------------
def handle_hr_approval():
    global pending_email_draft

    if not pending_email_draft:
        return

    print("\n====== EMAIL DRAFT (HR HEAD APPROVAL REQUIRED) ======")
    print(f"To      : {pending_email_draft['employee_email']}")
    print(f"Subject : {pending_email_draft['subject']}")
    print("\nBody:")
    print(pending_email_draft["body"])
    print("=====================================================")

    approval = input("\nHR Head approval - send this email? (yes/no): ").strip().lower()

    if approval in ["yes", "y"]:
        send_result = send_email_now(
            employee_email=pending_email_draft["employee_email"],
            subject=pending_email_draft["subject"],
            body=pending_email_draft["body"]
        )
        print("\n====== EMAIL STATUS ======")
        print(send_result)
    else:
        print("\nEmail sending cancelled by HR Head.")

    pending_email_draft = None

# -----------------------------
# RUN AGENT
# -----------------------------
def running_agent():
    print("\n==== RAG AGENT STARTED ====")

    while True:
        user_input = input("\nYour Question: ")

        if user_input.lower() in ["exit", "quit"]:
            break

        messages = [HumanMessage(content=user_input)]
        result = rag_agent.invoke({"messages": messages})

        print("\n====== ANSWER ======")
        print(result["messages"][-1].content)

        # Ask HR Head approval if email draft exists
        handle_hr_approval()

running_agent()