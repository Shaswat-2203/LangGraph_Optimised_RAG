# streamlit_app.py
import streamlit as st
from collections import Counter
from itertools import chain
from typing_extensions import TypedDict

import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import StateGraph, START, END

# -----------------------------------------------------------------------------
# 🗄️  State schema
# -----------------------------------------------------------------------------
class State(TypedDict):
    user_query: str
    subquery_generated: list[str]
    queryAnswers: list[str]
    queryAnswerLinks: list[str]
    filterAnswer: str
    hiteshSirAnswer: str

# -----------------------------------------------------------------------------
# 🧩  LangGraph node functions  (⚠️ prompts inside **MUST NOT change**)
# -----------------------------------------------------------------------------

def queryEnhancer(state: State):
    query = state["user_query"]
    SYSTEM_PROMPT = """
    You are a Query Decomposition Expert.
    
    Task:
    1. Read the user’s natural-language query.
    2. Derive exactly **three** distinct, meaningful subqueries that together cover the core information needs.
    3. **Respond with nothing except a JSON/array literal containing those three subqueries** (strictly using comma-separated, no brackets and extra spaces).  
       • No keys, labels, numbering, or extra commentary.  
       • Example format: subquery 1, subquery 2, subquery 3
    
    Obey the format strictly every time.
    
    Few-Shot Examples (in the required list-of-strings format)
    User Query	->  Assistant Output
    How can I reduce the size of my Docker images and improve build speed?   ->	What techniques shrink Docker image layers and remove unused files?, How do multi-stage builds optimize image size in Docker?, Which base image choices and caching strategies accelerate Docker build times?
    Why does my Node.js application show high CPU usage under heavy load?    ->	How can I identify CPU-bound code paths in a Node.js application?, What tools and methods profile event-loop latency and CPU use in Node.js?, Which optimization strategies reduce CPU overhead in Node.js when handling concurrent requests?
    Is MongoDB a good choice for real-time analytics compared to PostgreSQL? ->	How does MongoDB’s aggregation framework perform for real-time analytics workloads?, What are the scalability and sharding considerations for analytics in MongoDB versus PostgreSQL?, How do indexing strategies differ between MongoDB and PostgreSQL for analytical queries?
    """

    chat_model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config={"temperature": 0},
        system_instruction=SYSTEM_PROMPT,
    )
    response = chat_model.start_chat(history=[]).send_message(query)
    state["subquery_generated"] = [s.strip() for s in response.text.split(",")]
    return state


def searchRelevantPages(state: State):
    SYSTEM_PROMPT = (
        "You are a helpful AI assistant. "
        "Answer the user's question **only** from the supplied context "
        "and point them to the most relevant web page(s) for more info."
    )

    chat_model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config={"temperature": 0},
        system_instruction=SYSTEM_PROMPT,
    )

    state["queryAnswers"], state["queryAnswerLinks"] = [], []
    subqueries = [state["user_query"]] + state["subquery_generated"]

    for subquery in subqueries:
        results = st.session_state["VECTOR_DB"].similarity_search(subquery, k=5)
        context = "\n\n".join(f"Page Content: {r.page_content}" for r in results)
        links = [r.metadata.get("source") for r in results]

        user_prompt = f"Context:\n{context}\n\nQuestion:\n{subquery}\n\nAnswer:"
        response = chat_model.generate_content(user_prompt)

        state["queryAnswers"].append(response.text)
        state["queryAnswerLinks"].append(links)
    return state


def filterPages(state: State):
    user_query = state["user_query"]
    totalContext = "\n\n".join(state["queryAnswers"])
    SYSTEM_PROMPT =f"""
        You are an AI expert in understanding and finding relevant detailed information from given context so that it matches with user query the most.
        Return the most relevant answer. 
        Total Context:\n{totalContext}\n\n
    """

    chat_model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config={"temperature": 0},
        system_instruction=SYSTEM_PROMPT,
    )
    response = chat_model.generate_content(user_query)

    flat_links = list(chain.from_iterable(state["queryAnswerLinks"]))
    top_links = ", ".join(link for link, _ in Counter(flat_links).most_common(2))
    state["filterAnswer"] = response.text + f"\n\n**Relevant Links:** {top_links}"
    return state


def hiteshSirPersona(state: State):
    user_query = state["user_query"]
    answer = state["filterAnswer"]  # noqa: F841  # (kept because prompt references it internally)
    SYSTEM_PROMPT = f"""
    You are now emulating Hitesh Choudhary — a passionate Indian tech educator, software engineer, and YouTuber known for his crystal-clear teaching, love for chai, no-nonsense attitude, and Hinglish style.
    
    You are speaking directly to curious learners who might be beginners or intermediate coders. You are friendly, motivating, and break down topics step-by-step.
    
    Here’s how you behave:
    - You use a lot of Hinglish like “thoda patience rakho”, “ab yeh samajhna zaroori hai”, “mast kaam kiya!”, “ek chai ho jaaye”.
    - You encourage learners to code, not just consume content. “Sirf dekhne se kuch nahi hoga bhai, code bhi karo.”
    - You’re brutally honest: “Agar lagta hai easy hoga, toh galat soch rahe ho.”
    - You keep things practical: “Yeh sab theory ki baat nahi, chalo ek real-life example lete hain.”
    - You use analogies: “Socho function ek chai machine hai... input doge, output milega.”
    - You often begin with a hook: “Ab tum soch rahe hoge... ki iska kya fayda?”
    - You’re fun and inspiring: “Maza tab aayega jab khud build karoge.”
    - You often end tips with motivation: “Aage badho, seekhte raho, chill maaro.”
    - You sometimes mention your love for tea: “Main toh chai ke bina code hi nahi karta.”
    - You want learners to build projects: “Project banao, warna sab bhool jaoge.”
    
    **Example Hitesh-style responses:**
    
    1. “Chalo bhai, sabse pehle samajhte hain ki API hoti kya hai. Socho ek waiter restaurant mein — woh tumhari request kitchen tak le jaata hai.”
    2. “Agar recursion samajhna hai, toh bas soch lo mirror mein khud ko dekh rahe ho — har level pe ek aur tum ho!”
    3. “Yeh error ka matlab hai ki compiler tumse gussa hai 😅. Ab dekhte hain usko kaise shaant karte hain.”
    4. “Mujhe yaad hai pehli baar Git seekha tha — sab kuch uda diya tha repo se. But seekhna wahi se start hota hai.”
    5. “React ek masaledaar framework hai — pehle useState try karo, fir useEffect ka tadka lagao.”
    6. “Chalo isko divide karte hain 3 simple steps mein — aasan ho jaayega.”
    7. “Interview ke liye DSA zaroori hai, par project banana usse bhi zyada important hai.”
    8. “Socho Node.js ek waiter hai jo har table ko handle karta hai bina busy hue. That’s event loop for you.”
    9. “CSS tricky hai, lekin jab samajh aa jaye na... toh maza aa jata hai.”
    10. “Kabhi kabhi failure se zyada seekhne ko milta hai. Main bhi crash course banate waqt bahut kuch barbaad kar chuka hoon.”
    
    **Your goal:
    ** Emulate Hitesh Choudhary's mind — break down user query's problem, motivate learners, and guide them step-by-step in Hinglish with energy, humor, and chai-fueled wisdom.
    ** You should understand user query and the given answer(scrapped from chai-docs which is your own website) below and answer it in your way of speaking and also give them the web page.
    
    User query: {user_query}
    Answer: {answer}
    
    Speak like a friend who genuinely wants the learner to succeed.
    
    Let’s go!
    """

    chat_model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config={"temperature": 0},
        system_instruction=SYSTEM_PROMPT,
    )
    response = chat_model.generate_content(user_query)
    state["hiteshSirAnswer"] = response.text
    return state


# -----------------------------------------------------------------------------
# 🔗  Build (un‑compiled) LangGraph so we can call node functions manually
# -----------------------------------------------------------------------------

def build_graph():
    builder = StateGraph(State)
    builder.add_node("queryEnhancer", queryEnhancer)
    builder.add_node("searchRelevantPages", searchRelevantPages)
    builder.add_node("filterPages", filterPages)
    builder.add_node("hiteshSirPersona", hiteshSirPersona)

    builder.add_edge(START, "queryEnhancer")
    builder.add_edge("queryEnhancer", "searchRelevantPages")
    builder.add_edge("searchRelevantPages", "filterPages")
    builder.add_edge("filterPages", "hiteshSirPersona")
    builder.add_edge("hiteshSirPersona", END)
    return builder.compile()


# -----------------------------------------------------------------------------
# 🎨  Streamlit UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Hitesh AI Chat", page_icon="🤖", layout="centered")

with st.sidebar:
    st.title("🧠 Gemini Chat")
    api_key = st.text_input("Google Generative AI Key", type="password")
    if st.button("Reset Chat"):
        st.session_state.clear()

# ---- Ensure resources are initialised once the key is provided ----
if api_key:
    genai.configure(api_key=api_key)

    if "VECTOR_DB" not in st.session_state:
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        st.session_state["VECTOR_DB"] = QdrantVectorStore.from_existing_collection(
            url="https://e59dfb81-bb98-4eb6-9806-f172c977a89f.us-east-1-0.aws.cloud.qdrant.io:6333",
            api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.EhU0hmKfZ9p-LYubvLHcF7aQg-piIYam2L1qK7CdAnE",
            collection_name="chai_docs",
            embedding=embedding_model,
        )

    if "GRAPH" not in st.session_state:
        st.session_state["GRAPH"] = build_graph()

    # ---- Chat area ----
    st.title("💬 Ask like a Pro")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display chat history
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).markdown(msg["content"])

    # ---- Input ----
    user_input = st.chat_input("Type your tech query…")

    if user_input:
        # Save & echo user message
        st.session_state["messages"].append({"role": "user", "content": user_input})
        st.chat_message("user").markdown(user_input)

        # ------- 👀 Step‑by‑step node transition UI -------
        status_placeholder = st.empty()
        steps = [
            ("Thinking", queryEnhancer),
            ("Observing", searchRelevantPages),
            ("Filtering", filterPages),
            ("Hitesh Sir is connecting", hiteshSirPersona),
        ]
        progress = {label: False for label, _ in steps}

        def render_status():
            lines = []
            for label, _ in steps:
                icon = "✅" if progress[label] else "⏳"
                lines.append(f"{icon} **{label}**")
            status_placeholder.markdown("\n".join(lines))

        # Initial render (all pending)
        render_status()

        # Initial state object
        state: State = {
            "user_query": user_input,
            "subquery_generated": [],
            "queryAnswers": [],
            "queryAnswerLinks": [],
            "filterAnswer": "",
            "hiteshSirAnswer": "",
        }

        # Execute each node & update UI
        for label, fn in steps:
            state = fn(state)
            progress[label] = True
            render_status()

        # Hide progress once complete
        status_placeholder.empty()

        # Show final answer
        assistant_answer = state["hiteshSirAnswer"]
        st.session_state["messages"].append({"role": "assistant", "content": assistant_answer})
        st.chat_message("assistant").markdown(assistant_answer)
else:
    st.warning("Please provide your Gemini API key in the sidebar.")
