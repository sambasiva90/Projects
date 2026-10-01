import os
import json

import streamlit as st
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_google_genai import ChatGoogleGenerativeAI
import faiss

from products import PRODUCTS

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Shopping Assistant", page_icon="🛍️", layout="wide")
st.title("🛍️ E-commerce Product Assistant (RAG-powered)")
st.caption(
    "Ask in plain language — e.g. \"lightweight laptop under $1000 with good battery for travel\". "
    "Powered by hybrid retrieval (semantic + filters) + Gemini."
)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

FILTER_EXTRACTION_PROMPT = """You extract structured shopping filters from a user's message for a \
product search system. You are given the filters already established from earlier in the \
conversation (if any) and the user's latest message. Update the filters based on the latest \
message — keep prior filters unless the user overrides or clears them (e.g. "actually show me \
phones instead" replaces category; "cheaper" lowers price_max; "never mind the brand" clears brand).

Previously established filters:
{prev_filters}

User's latest message:
{query}

Respond with ONLY a valid JSON object (no markdown fences, no commentary) with exactly these keys:
{{
  "category": <string or null, one of: Laptop, Headphones, Jacket, Shoes, Phone, Backpack, or null if not specified>,
  "brand": <string or null>,
  "price_min": <number or null>,
  "price_max": <number or null>,
  "search_text": "<freeform text capturing what qualities/features/use-case the user wants, for semantic search>"
}}
"""

RECOMMENDATION_PROMPT = """You are a helpful shopping assistant. Recommend products to the user based \
ONLY on the catalog items provided below — never invent products or specs not listed.

User's request: {query}

Matching catalog items (JSON):
{products}

Write a short, friendly recommendation (under 150 words). Mention 2-4 specific products by name, \
explain briefly why each fits (or a trade-off if relevant, e.g. price vs. battery life). If the \
catalog items don't fit well, say so honestly rather than forcing a recommendation.
"""

# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
defaults = {"filters": {}, "chat_history": []}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ----------------------------------------------------------------------------
# Catalog + vector store (built once, cached)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


@st.cache_resource(show_spinner=False)
def build_catalog_store():
    embeddings = get_embeddings()
    index = faiss.IndexFlatL2(EMBED_DIM)
    store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    docs = []
    for p in PRODUCTS:
        text = (
            f"{p['title']} by {p['brand']}. Category: {p['category']}. Price: ${p['price']}. "
            f"Rating: {p['rating']}/5. {p['description']} Features: {p['features']}."
        )
        docs.append(Document(page_content=text, metadata=p))
    store.add_documents(docs)
    return store


def get_llm(api_key: str, model_name: str, temperature: float = 0.0):
    return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=temperature)


def parse_json_response(raw: str):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        if raw.lower().startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


def extract_filters(query, prev_filters, api_key, model_name):
    llm = get_llm(api_key, model_name, temperature=0.0)
    prompt = FILTER_EXTRACTION_PROMPT.format(prev_filters=json.dumps(prev_filters), query=query)
    raw = llm.invoke(prompt).content
    try:
        return parse_json_response(raw)
    except (json.JSONDecodeError, ValueError):
        # Fall back to just using the raw query for semantic search
        merged = dict(prev_filters)
        merged["search_text"] = query
        return merged


def retrieve_products(store, filters, k=20, top_n=5):
    search_text = filters.get("search_text") or ""
    if search_text.strip():
        docs = store.similarity_search(search_text, k=k)
    else:
        # No semantic query yet (e.g. pure filter request) — pull broadly via a generic query
        docs = store.similarity_search("product", k=k)

    def matches(meta):
        if filters.get("category") and filters["category"].lower() not in meta["category"].lower():
            return False
        if filters.get("brand") and filters["brand"].lower() not in meta["brand"].lower():
            return False
        if filters.get("price_min") is not None and meta["price"] < filters["price_min"]:
            return False
        if filters.get("price_max") is not None and meta["price"] > filters["price_max"]:
            return False
        return True

    filtered = [d for d in docs if matches(d.metadata)]
    fallback_used = False
    if not filtered:
        # Broaden: drop category/brand filters but keep price constraints
        def price_only_match(meta):
            if filters.get("price_min") is not None and meta["price"] < filters["price_min"]:
                return False
            if filters.get("price_max") is not None and meta["price"] > filters["price_max"]:
                return False
            return True

        filtered = [d for d in docs if price_only_match(d.metadata)]
        fallback_used = True

    return filtered[:top_n], fallback_used


def generate_recommendation(query, products, api_key, model_name):
    llm = get_llm(api_key, model_name, temperature=0.3)
    products_json = json.dumps([d.metadata for d in products], indent=2)
    prompt = RECOMMENDATION_PROMPT.format(query=query, products=products_json)
    return llm.invoke(prompt).content


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input(
        "Google API Key",
        type="password",
        value=os.environ.get("GOOGLE_API_KEY", ""),
        help="Get a key from https://aistudio.google.com/apikey.",
    )
    model_name = st.selectbox(
        "Gemini model", options=["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"], index=0
    )

    st.divider()
    st.header("📦 Catalog")
    st.caption(f"{len(PRODUCTS)} demo products loaded across "
               f"{len(set(p['category'] for p in PRODUCTS))} categories.")
    with st.expander("View catalog"):
        for p in PRODUCTS:
            st.caption(f"**{p['title']}** — {p['brand']} — ${p['price']} — ⭐{p['rating']}")

    if st.session_state.filters:
        st.divider()
        st.header("🔎 Active filters")
        st.json(st.session_state.filters)

    if st.button("Reset conversation"):
        st.session_state.filters = {}
        st.session_state.chat_history = []
        st.rerun()

# ----------------------------------------------------------------------------
# Main — chat interface
# ----------------------------------------------------------------------------
store = build_catalog_store()

for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(entry["query"])
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        if entry["products"]:
            cols = st.columns(len(entry["products"]))
            for col, meta in zip(cols, entry["products"]):
                with col:
                    st.markdown(f"**{meta['title']}**")
                    st.caption(f"{meta['brand']} · {meta['category']}")
                    st.markdown(f"💲 **{meta['price']}** · ⭐ {meta['rating']}")
                    st.caption(meta["description"])

query = st.chat_input('Try: "lightweight laptop under $1000 with good battery for travel"')

if query:
    if not api_key:
        st.error("Please enter your Google API key in the sidebar first.")
    else:
        with st.chat_message("user"):
            st.write(query)
        with st.chat_message("assistant"):
            with st.spinner("Understanding your request..."):
                filters = extract_filters(query, st.session_state.filters, api_key, model_name)
                st.session_state.filters = filters
            with st.spinner("Searching catalog..."):
                results, fallback_used = retrieve_products(store, filters)
            with st.spinner("Preparing recommendation..."):
                if results:
                    answer = generate_recommendation(query, results, api_key, model_name)
                else:
                    answer = "I couldn't find anything in the catalog matching that — try loosening your constraints."
                if fallback_used and results:
                    answer = "*(No exact category/brand match — showing closest options.)*\n\n" + answer

            st.write(answer)
            metas = [d.metadata for d in results]
            if metas:
                cols = st.columns(len(metas))
                for col, meta in zip(cols, metas):
                    with col:
                        st.markdown(f"**{meta['title']}**")
                        st.caption(f"{meta['brand']} · {meta['category']}")
                        st.markdown(f"💲 **{meta['price']}** · ⭐ {meta['rating']}")
                        st.caption(meta["description"])

            st.session_state.chat_history.append({"query": query, "answer": answer, "products": metas})
