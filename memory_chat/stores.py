import uuid
from typing import Sequence, Tuple, Any, List, Optional
from langgraph.store.base import BaseStore  # 👈 我们只导入 BaseStore 基类
from langchain_core.vectorstores import VectorStore
import os
import asyncio
from dotenv import load_dotenv

# ✨ 关键修正：在类定义中移除了 [str, Any] 这个泛型部分
class VectorStoreWrapperForLangMem(BaseStore):
    """
    An adapter that wraps a LangChain VectorStore to make it compatible
    with the langgraph.store.base.BaseStore interface.
    """

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def _get_namespace_and_metadata(self, key: Tuple[str, ...]) -> Tuple[str, dict]:
        """Helper to extract namespace and metadata from the key tuple."""
        # This logic depends on the vector store. For stores like Pinecone/Redis
        # that use a single string namespace, we can use the collection part.
        namespace = key[2] if len(key) > 2 else "default_collection"
        metadata = {
            "app_id": key[0] if len(key) > 0 else "N/A",
            "user_id": key[1] if len(key) > 1 else "N/A",
            "collection": namespace
        }
        return namespace, metadata

    # --- 实现写入方法 ---
    async def aput(self, items_or_namespace, key=None, value=None) -> None:
        # Handle both calling patterns:
        # 1. aput(items: Sequence[Tuple[Any, Any]]) - from our test
        # 2. aput(namespace, key=..., value=...) - from langmem tool
        
        if key is not None and value is not None:
            # Pattern 2: langmem tool calling aput(namespace, key=..., value=...)
            print(f"--- 🔌 Storing memory via langmem tool ---")
            # Convert to our expected format
            items = [(items_or_namespace, value)]
        else:
            # Pattern 1: our test calling aput(items)
            print(f"--- 🔌 Custom Adapter 'aput' called with items pattern ---")
            items = items_or_namespace
            print(f"  - Items received: {items}")
        
        texts_to_add = []
        metadatas = []
        ids = []

        # ✨ 关键：我们现在只处理 langmem 传入的标准格式
        for item_key, item_value in items:
            # 1. 我们【期望】value 是一个字典
            if not isinstance(item_value, dict):
                print(f"⚠️ WARNING: Skipping item because value is not a dictionary: {item_value}")
                continue

            # 2. 我们从字典中【提取】纯字符串 content
            content = item_value.get("content")

            if content and isinstance(content, str):
                # 3. 准备【扁平的】数据给 aadd_texts
                namespace, metadata = self._get_namespace_and_metadata(item_key)
                texts_to_add.append(content)
                metadatas.append(metadata)
                ids.append(str(uuid.uuid4()))
                print(f"  - Storing: '{content[:50]}...' in collection '{namespace}'")

        if texts_to_add:
            # 4. 调用 aadd_texts，现在它接收的是简单的、非嵌套的数据
            try:
                # Use synchronous add_texts instead of async to avoid thread issues
                self.vector_store.add_texts(
                    texts=texts_to_add,
                    metadatas=metadatas,
                    ids=ids
                )
                print(f"  - ✅ Successfully stored {len(texts_to_add)} memory(ies)")
            except Exception as e:
                print(f"  - ❌ Error storing memories: {e}")
                import traceback
                traceback.print_exc()
                raise
        else:
            print(f"  - ⚠️ No texts to add after processing items")

    # --- 实现所有必需的抽象方法 (最小化实现) ---
    def put(self, items: Sequence[Tuple[str, Any]]) -> None:
        raise NotImplementedError("Sync operations not implemented. Use async versions.")

    async def abatch(self, items: Sequence[Tuple[str, Any]]) -> None:
        # 批量操作可以直接调用 aput
        await self.aput(items)

    def batch(self, items: Sequence[Tuple[str, Any]]) -> None:
        raise NotImplementedError("Sync operations not implemented. Use async versions.")

    async def aget(self, keys: Sequence[str]) -> List[Optional[Any]]:
        print(f"⚠️ WARNING: .aget() is not implemented for semantic search. Returning None for {len(keys)} keys.")
        return [None] * len(keys)

    def get(self, keys: Sequence[str]) -> List[Optional[Any]]:
        raise NotImplementedError("Sync operations not implemented. Use async versions.")

    async def adelete(self, keys: Sequence[str]) -> None:
        print("⚠️ WARNING: .adelete() is not implemented in this adapter.")
        pass

    def delete(self, keys: Sequence[str]) -> None:
        raise NotImplementedError("Sync operations not implemented. Use async versions.")


# --- 导入基础设施 ---
from langchain_community.vectorstores import SQLiteVSS
from langchain_community.embeddings import DashScopeEmbeddings


# ==============================================================================
# Main Debug Logic
# ==============================================================================
async def main():
    print("--- 🔬 Starting Adapter Direct Invocation Test ---")
    load_dotenv()

    # 1. 初始化一个真实的 VectorStore (SQLiteVSS)
    embedding_model = DashScopeEmbeddings(model="text-embedding-v2")
    db_file = "debug_semantic_memory.db"
    if os.path.exists(db_file):
        os.remove(db_file)  # 确保每次测试都是全新的
    underlying_vector_store = SQLiteVSS.from_texts(
        texts=[],
        embedding=embedding_model,
        db_file=db_file,
        table_name="debug_collection"
    )
    print("✅ Underlying VectorStore (SQLiteVSS) is ready.")

    # 2. 用我们的适配器包装它
    persistent_store = VectorStoreWrapperForLangMem(vector_store=underlying_vector_store)

    # 3. 模拟 langmem 工具的调用
    # langmem 会传入一个元组列表，每个元组是 (namespace_tuple, value_dict)
    namespace = ("edit_mem_assistant", "debug_user", "debug_collection")
    value = {"content": "This is a test of the adapter."}
    items_to_put = [(namespace, value)]

    print(f"\n--- 📞 Attempting to call adapter.aput() with items: {items_to_put} ---")
    try:
        # ✨ 直接调用我们适配器的 .aput 方法
        await persistent_store.aput(items_to_put)

        print("\n--- ✅ Adapter .aput() call Succeeded without exception ---")

        # ✨ 验证：尝试从底层的 vector store 中搜索数据
        try:
            results = underlying_vector_store.similarity_search("test adapter")
            print("\n--- 🔍 Verification Search Results ---")
            if results:
                print("  - SUCCESS! Found stored data:")
                for doc in results:
                    print(f"    - Content: {doc.page_content}, Metadata: {doc.metadata}")
            else:
                print("  - FAILURE! No data found in the vector store after .aput() call.")
        except Exception as e:
            print(f"\n--- ⚠️ Verification search failed: {e} ---")
            print("  - But the .aput() operation succeeded, so the data should be stored.")

    except Exception as e:
        print("\n--- ❌ Adapter .aput() call FAILED ---")
        import traceback
        print("Error Type:", type(e))
        print("Error Message:", e)
        print("--- Traceback ---")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())