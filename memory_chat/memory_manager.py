# memory_manager.py (可以创建一个新文件)
import os
import json
import chromadb
from datetime import datetime
from typing import Dict, List, Any

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings  # 假设你继续使用这个
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory


class MemoryManager:
    def __init__(self, medium_term_path="./memory_logs/", long_term_path="./vector_db/"):
        # --- 短期记忆 ---
        self.short_term_memory: Dict[str, ChatMessageHistory] = {}

        # --- 中期记忆 ---
        self.medium_term_path = medium_term_path
        os.makedirs(self.medium_term_path, exist_ok=True)

        # --- 长期记忆 (RAG) ---
        self.long_term_path = long_term_path
        self.embedding_function = DashScopeEmbeddings(model="text-embedding-v2")  # 使用和之前一致的
        self.vector_store = Chroma(
            persist_directory=self.long_term_path,
            embedding_function=self.embedding_function
        )
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    # --- 短期记忆方法 ---
    def get_short_term_history(self, session_id: str) -> ChatMessageHistory:
        if session_id not in self.short_term_memory:
            self.short_term_memory[session_id] = ChatMessageHistory()
        return self.short_term_memory[session_id]

    # --- 中期记忆方法 ---
    def _get_medium_term_filepath(self, file_id: str) -> str:
        """根据文件ID生成记忆日志的文件路径。"""
        return os.path.join(self.medium_term_path, f"{os.path.basename(file_id)}.mem.json")

    def save_edit_to_medium_term(self, file_id: str, edit_details: Dict[str, Any]):
        """将一次编辑操作存入中期记忆。"""
        filepath = self._get_medium_term_filepath(file_id)
        history = self.load_medium_term_history(file_id)

        # 添加时间戳
        edit_details['timestamp'] = datetime.now().isoformat()
        history.append(edit_details)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=4, ensure_ascii=False)

    def load_medium_term_history(self, file_id: str) -> List[Dict[str, Any]]:
        """加载特定文件的编辑历史。"""
        filepath = self._get_medium_term_filepath(file_id)
        if not os.path.exists(filepath):
            return []
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    # --- 长期记忆方法 ---
    def retrieve_from_long_term(self, query: str, k: int = 3) -> str:
        """从向量数据库中检索相关知识。"""
        docs = self.vector_store.similarity_search(query, k=k)
        return "\n\n".join([doc.page_content for doc in docs])

    def ingest_document_to_long_term(self, file_path: str):
        """将一个文档的内容消化并存入长期记忆。"""
        print(f"Ingesting document '{file_path}' into long-term memory...")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        chunks = self.text_splitter.split_text(text)
        self.vector_store.add_texts(
            texts=chunks,
            # 添加元数据以便追溯来源
            metadatas=[{"source": file_path} for _ in chunks]
        )
        self.vector_store.persist()
        print("Ingestion complete.")