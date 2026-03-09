from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
#libries for Add an Email Sending Tool
import smtplib
from email.mime.text import MIMEText
from langchain_core.tools import tool

load_dotenv()

# LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7
)

# Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# PDF path
pdf_path = "C:/Users/subha/OneDrive/Desktop/RAG-NEW-AGENT/hr_manual.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF not found at {pdf_path}")

# Load PDF
loader = PyPDFLoader(pdf_path)
pages = loader.load()
print(f"PDF loaded successfully with {len(pages)} pages.")

# Text splitting (improved)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=300
)

docs = text_splitter.split_documents(pages)

# Vector DB
persist_directory = "C:/Users/subha/OneDrive/Desktop/RAG-NEW-AGENT/vector_db"
collection_name = "hr_manual"

if os.path.exists(persist_directory):
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name
    )
else:
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )

print("Vector database ready.")

# Retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k":8}
)

# Tool
@tool
def retriever_tool(query: str) -> str:
    """Search HR manual for relevant information"""

    docs = retriever.invoke(query)

    if not docs:
        return "No relevant information found in the HR manual."

    results = []

    for doc in docs:
        page = doc.metadata.get("page", "unknown")

        results.append(
            f"[HR Manual - Page {page}]\n{doc.page_content}"
        )

    return "\n\n".join(results)

@tool
def send_email_tool(employee_email: str, subject: str, body: str) -> str:
    """
    Send an email to the employee automatically.
    """

    sender_email = os.getenv("EMAIL_ADDRESS")
    sender_password = os.getenv("EMAIL_PASSWORD")

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = employee_email

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, employee_email, msg.as_string())

        return f"Email sent successfully to {employee_email}"

    except Exception as e:
        return f"Error sending email: {str(e)}"


tools = [retriever_tool, send_email_tool]
llm = llm.bind_tools(tools)

# Agent State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Continue decision
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    return hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0


system_prompt = """
You are an HR assistant that answers questions using ONLY the HR manual.

Rules:
1. Always call the retriever_tool before answering.
2. Use ONLY the information returned from the HR manual.
3. Do not add policies that are not present in the retrieved text.
4. When generating emails or guidance, base them strictly on the retrieved HR policies.
5. Cite the HR manual page numbers in your answer.
6. If the information is not found, say:
   "This information is not available in the HR manual."
7.Generate professional HR emails.
8. Use send_email_tool to send the email if employee email is provided.
"""

# Tool dictionary
tools_dict = {tool.name: tool for tool in tools}

# LLM node
def call_llm(state: AgentState) -> AgentState:

    messages = list(state["messages"])
    messages = [SystemMessage(content=system_prompt)] + messages

    response = llm.invoke(messages)

    return {"messages": [response]}

# Tool execution node
def take_action(state: AgentState) -> AgentState:

    tool_calls = state["messages"][-1].tool_calls
    tool_results = []

    for t in tool_calls:

        tool_name = t["name"]
        tool_args = t["args"]

        print(f"\nCalling Tool: {tool_name}")
        print(f"Query: {tool_args.get('query','')}")

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

# Graph
graph = StateGraph(AgentState)

graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END}
)

graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()

# Run agent
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

running_agent()