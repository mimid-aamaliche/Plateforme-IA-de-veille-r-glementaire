"""
Legal RAG Engine
================
Core RAG pipeline with a custom JSON messaging protocol.

Message Protocol
----------------
Incoming payload:
    {
        "user_prompt": str,
        "system_prompt": str,        # optional override
        "context": list[dict]        # conversation history [{role, content}, ...]
    }

LLM response (structured JSON):
    {
        "response_to": "user" | "system",
        "response":    str,          # non-empty when response_to == "user"
        "query":       str           # non-empty when response_to == "system" → triggers DB retrieval
    }

Flow
----
1. User sends payload → LLM decides if it needs DB context or can answer directly.
2. If response_to == "system"  → engine queries the vector DB with `query`,
   appends retrieved chunks to context, calls LLM again (loop).
3. If response_to == "user"    → final answer is returned to the caller.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from itertools import groupby
from dotenv import load_dotenv


load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env" )

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    DirectoryLoader, PyMuPDFLoader, TextLoader, CSVLoader, Docx2txtLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ── provider imports (loaded lazily so missing packages don't crash the other) ──
def _get_openai_llm(model: str, temperature: float, api_key: str | None):
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    llm = ChatOpenAI(model=model, temperature=temperature,
                     openai_api_key=api_key or os.getenv("OPENAI_API_KEY"))
    embeddings = OpenAIEmbeddings(openai_api_key=api_key or os.getenv("OPENAI_API_KEY"))
    return llm, embeddings


def _get_ollama_llm(model: str, temperature: float, base_url: str):
    from langchain_community.llms import Ollama
    from langchain_community.embeddings import OllamaEmbeddings
    llm = Ollama(model=model, temperature=temperature, base_url=base_url)
    embeddings = OllamaEmbeddings(model=model, base_url=base_url)
    return llm, embeddings


def _get_gemini_llm(model: str, temperature: float, api_key: str | None, ollama_base_url: str = "http://localhost:11434"):
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_community.embeddings import OllamaEmbeddings
    key = api_key or os.getenv("GOOGLE_API_KEY")
    print(key)
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=key,
        convert_system_message_to_human=True,
    )
    # Free local embeddings via Ollama — no API key needed
    embeddings = OllamaEmbeddings(
        model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        base_url=ollama_base_url,
    )
    return llm, embeddings


# ── Default system prompt (Moroccan legal assistant) ─────────────────────────
DEFAULT_SYSTEM_PROMPT = """You are a friendly Moroccan legal advisor who explains the law in plain, everyday language — like a knowledgeable friend, not a lawyer in court.

You have access to a knowledge base containing the Code Pénal, Code de commerce, Moudawwana, company bylaws, and other Moroccan legal documents.

IMPORTANT — You MUST respond ONLY with a valid JSON object in this exact format:
{
  "response_to": "user" | "system",
  "response": "<your answer, or empty string>",
  "query": "<a precise search query to retrieve relevant legal articles, or empty string>"
}

ROUTING RULES:
- Need to look up laws first? → response_to="system", query="...", response=""
- Ready to answer? → response_to="user", response="...", query=""
- Never fill both response and query at the same time.

RESPONSE RULES:
1. Be concise — short paragraphs, no filler. Say more with fewer words.
2. Use simple, everyday language. Avoid legal jargon. If a legal term is necessary, explain it in one sentence.
3. NEVER cite article numbers or law names unless the user explicitly asks "which law" or "what article". Use the law to inform your answer, don't recite it.
4. For scenario/situation questions ("what if...", "what happens when...", "can my employer...", etc.):
   - Acknowledge the situation briefly
   - Give a clear, direct answer: what the person can do, what their rights are, what risks they face
   - End with a concrete next step or practical advice
5. Match the user's language: Arabic → reply in Arabic, French → reply in French.
6. Keep responses under 150 words unless the situation genuinely requires more detail.
7. When writing in Arabic, avoid embedding French acronyms or terms inline. 
  Either translate them, or place them at the end in parentheses.
"""



class LegalRAGEngine:
    """
    Stateless RAG engine.  The caller owns the conversation history.
    """

    MAX_RETRIEVAL_LOOPS = 10   # safety limit to prevent infinite DB-query loops

    def __init__(
        self,
        provider: str = "gemini",          # "gemini" | "openai" | "ollama"
        model: str = "gemini-2.5-flash",
        temperature: float = 0.1,
        api_key: str | None = None,        # OpenAI or Google API key
        ollama_base_url: str = "http://localhost:11434",
        vector_store_dir: str | None = None,
        k_results: int = 4,
        system_prompt: str | None = None,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.ollama_base_url = ollama_base_url
        self.vector_store_dir = Path(vector_store_dir or "./data/vector_stores")
        self.k_results = k_results
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

        self.llm, self.embeddings = self._init_provider()
        self.vectorstore: Chroma | None = self._load_vectorstore()

    # ── Initialisation ────────────────────────────────────────────────────────
    def _init_provider(self):
        if self.provider == "openai":
            return _get_openai_llm(self.model, self.temperature, self.api_key)
        elif self.provider == "ollama":
            return _get_ollama_llm(self.model, self.temperature, self.ollama_base_url)
        elif self.provider == "gemini":
            return _get_gemini_llm(self.model, self.temperature, self.api_key, self.ollama_base_url)
        else:
            raise ValueError(f"Unknown provider: {self.provider!r}. Use 'openai', 'gemini', or 'ollama'.")

    def _load_vectorstore(self) -> Chroma | None:
        if self.vector_store_dir.exists() and any(self.vector_store_dir.iterdir()):
            try:
                vs = Chroma(
                    persist_directory=str(self.vector_store_dir),
                    embedding_function=self.embeddings,
                )
                print(f"[RAG] Vector store loaded from {self.vector_store_dir}")
                return vs
            except Exception as e:
                print(f"[RAG] Could not load vector store: {e}")
        return None

    # ── Document ingestion ───────────────────────────────────────────────────
    def ingest_documents(self, source_dir: str | Path, chunk_size: int = 1200, chunk_overlap: int = 300):
        """Load, split and embed documents from source_dir into the vector store."""
        source_dir = Path(source_dir)
        raw_documents = []

        # 1. Define loaders with PyMuPDF for better PDF handling
        loaders = [
            DirectoryLoader(
                str(source_dir), glob="**/*.pdf", loader_cls=PyMuPDFLoader, 
                show_progress=True
            ),
            DirectoryLoader(
                str(source_dir), glob="**/*.txt", loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf8"}, show_progress=True
            ),
            DirectoryLoader(
                str(source_dir), glob="**/*.csv", loader_cls=CSVLoader,
                loader_kwargs={"encoding": "utf8"}, show_progress=True
            ),
            DirectoryLoader(
                str(source_dir), glob="**/*.docx", loader_cls=Docx2txtLoader, 
                show_progress=True
            ),
        ]

        # 2. Load all pages/files
        for loader in loaders:
            try:
                raw_documents.extend(loader.load())
            except Exception as e:
                print(f"[RAG] Loader warning: {e}")

        if not raw_documents:
            raise ValueError(f"No documents found in {source_dir}")

        # 3. MERGE LOGIC: Group pages by source to reconstruct full documents
        # This ensures Article separators work across page boundaries.
        processed_docs = []
        # Sort by source for the groupby function
        raw_documents.sort(key=lambda x: x.metadata.get("source", ""))
        
        for source, group in groupby(raw_documents, key=lambda x: x.metadata.get("source", "")):
            pages = list(group)
            # Join content with a clear page break marker for the LLM
            full_content = "\n\n".join([p.page_content for p in pages])
            
            # Preserve original metadata from the first page
            combined_metadata = pages[0].metadata
            combined_metadata["filename"] = Path(source).name
            
            processed_docs.append(Document(page_content=full_content, metadata=combined_metadata))

        # 4. Split the merged documents
        splitter = RecursiveCharacterTextSplitter(
            # We prioritize Article boundaries in French and Arabic
            separators=["\nArticle ", "\nالمادة ", "\n\n", "\n", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            keep_separator=True,
        )
        
        chunks = splitter.split_documents(processed_docs)
        
        # Ensure every chunk has a clean filename for citations
        for chunk in chunks:
            if "source" in chunk.metadata:
                chunk.metadata["filename"] = Path(chunk.metadata["source"]).name

        print(f"[RAG] Ingested {len(processed_docs)} documents → {len(chunks)} chunks")

        # 5. Save to Vector Store
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=str(self.vector_store_dir),
        )
        self.vectorstore.persist()
        print(f"[RAG] Vector store saved to {self.vector_store_dir}")
        
        return len(chunks)



    # ── Vector DB retrieval ──────────────────────────────────────────────────
    def _retrieve(self, query: str) -> str:
        if self.vectorstore is None:
            return "[No vector store available. Please ingest documents first.]"
        docs = self.vectorstore.similarity_search(query, k=self.k_results)
        if not docs:
            return "[No relevant documents found.]"
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "")
            ref = f"{Path(source).name}" + (f" p.{page}" if page != "" else "")
            parts.append(f"--- [Source {i}: {ref}] ---\n{doc.page_content}")
        return "\n\n".join(parts)

    # ── LLM call ─────────────────────────────────────────────────────────────
    def _call_llm(self, messages: list[dict]) -> dict:
        """
        Call the LLM and parse the structured JSON response.
        Returns a dict with keys: response_to, response, query
        """
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

        lc_messages = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))

        raw = self.llm.invoke(lc_messages)
        text = raw.content if hasattr(raw, "content") else str(raw)

        # Strip markdown fences if the model wrapped its JSON
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Graceful fallback: treat the whole output as a user response
            parsed = {"response_to": "user", "response": text, "query": ""}

        # Normalise keys
        return {
            "response_to": parsed.get("response_to", "user"),
            "response":    parsed.get("response", ""),
            "query":       parsed.get("query", ""),
        }



    # ── Main public method ───────────────────────────────────────────────────
    def chat(self, payload: dict) -> dict:
        """
        Process one turn of the conversation.

        Args:
            payload: {
                "user_prompt":   str,
                "system_prompt": str (optional),
                "context":       list[{role, content}]  (optional, conversation history)
            }

        Returns:
            {
                "response_to": "user",
                "response":    str,          # final answer
                "query":       "",
                "context":     list[dict]    # updated history to pass on next turn
            }
        """

        user_prompt   = payload.get("user_prompt", "")
        # we should consider in the future to make some changes on the query 
        context_text= self._retrieve(user_prompt)
        system_prompt = self.system_prompt + f"""
                        \n\n[INITIAL LEGAL CONTEXT]\n
                        {context_text}
                        \n\n
                        """
        context       = list(payload.get("context", []))  # mutable copy

        if not user_prompt.strip():
            return {
                "response_to": "user",
                "response": "Please enter a question.",
                "query": "",
                "context": context,
            }

        # Build the message list for this turn
        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        messages.extend(context)
        messages.append({"role": "user", "content": user_prompt})

        # Add user turn to history
        context.append({"role": "user", "content": user_prompt})

        for loop in range(self.MAX_RETRIEVAL_LOOPS):
            llm_response = self._call_llm(messages)

            if llm_response["response_to"] == "user":
                # Done — return final answer
                context.append({"role": "assistant", "content": llm_response["response"]})
                return {
                    "response_to": "user",
                    "response":    llm_response["response"],
                    "query":       "",
                    "context":     context,
                }

            elif llm_response["response_to"] == "system":
                # The LLM wants DB context — retrieve and inject
                query = llm_response.get("query", "").strip()
                if not query:
                    # Malformed system response: fall back
                    context.append({"role": "assistant", "content": llm_response.get("response", "")})
                    return {**llm_response, "context": context}

                retrieved = self._retrieve(query)
                retrieval_message = (
                    f"[SYSTEM — DB results for query: \"{query}\"]\n\n{retrieved}\n\n"
                    "Review this additional information and provide your final answer. "
                    "Respond with JSON as instructed."
                )
                # Append the retrieval result as a synthetic assistant turn + new user nudge
                messages.append({"role": "assistant", "content": json.dumps(llm_response)})
                messages.append({"role": "user", "content": retrieval_message})

            else:
                # Unknown response_to value — treat as user response
                context.append({"role": "assistant", "content": llm_response.get("response", "")})
                return {**llm_response, "context": context}

        # Loop exhausted without a user response (shouldn't happen in practice)
        fallback = "I was unable to retrieve enough context to answer. Please rephrase your question."
        context.append({"role": "assistant", "content": fallback})
        return {"response_to": "user", "response": fallback, "query": "", "context": context}