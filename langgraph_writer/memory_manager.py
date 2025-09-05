import os
import asyncio
from typing import List, Dict, Optional
from langchain_community.embeddings import DashScopeEmbeddings  
from langchain_community.vectorstores import SQLiteVSS
from memory_chat.stores import VectorStoreWrapperForLangMem
from langmem import create_search_memory_tool
from langgraph.store.base import BaseStore


class WritingMemoryManager:
    """统一的写作记忆管理器，用于检索相关的编辑经验和写作指导"""
    
    def __init__(self, db_file: str = "semantic_memory.db", table_name: str = "memory_collection"):
        """
        初始化记忆管理器
        
        Args:
            db_file: SQLite数据库文件路径
            table_name: 向量存储表名
        """
        self.db_file = db_file
        self.table_name = table_name
        
        # 初始化embedding模型
        self.embedding_model = DashScopeEmbeddings(model="text-embedding-v2")
        
        # 初始化向量存储
        self.vector_store = SQLiteVSS.from_texts(
            texts=[],
            embedding=self.embedding_model,
            db_file=self.db_file,
            table_name=self.table_name
        )
        
        # 包装为BaseStore兼容接口
        self.persistent_store = VectorStoreWrapperForLangMem(
            vector_store=self.vector_store
        )
        
        # 创建搜索工具
        self.search_memory_tool = create_search_memory_tool(
            namespace=("edit_mem_assistant", "{langgraph_user_id}", "{collection_name}")
        )
    
    def search_relevant_memories(self, query: str, user_id: str = "dengjuan", 
                                collection: str = "intro", max_results: int = 3) -> List[str]:
        """
        搜索相关的记忆内容
        
        Args:
            query: 搜索查询
            user_id: 用户ID
            collection: 记忆集合名称（intro, leaf等）
            max_results: 最大返回结果数
            
        Returns:
            相关记忆内容的列表
        """
        try:
            # 直接使用向量存储的相似度搜索
            docs = self.vector_store.similarity_search(
                query=query, 
                k=max_results,
                filter={"collection": collection}
            )
            
            memories = []
            for doc in docs:
                # 提取记忆内容，去除格式化前缀
                content = doc.page_content
                if content.startswith("修改的说理性文字："):
                    content = content.replace("修改的说理性文字：", "")
                elif content.startswith("记录修改的说理性文字："):
                    content = content.replace("记录修改的说理性文字：", "")
                elif content.startswith("记录文本修改说明："):
                    content = content.replace("记录文本修改说明：", "")
                    
                memories.append(content.strip())
            
            return memories
            
        except Exception as e:
            print(f"⚠️ 记忆检索失败: {e}")
            return []
    
    def get_writing_guidance(self, section_type: str) -> str:
        """
        获取写作指导，基于历史编辑经验（按section_type分类）
        
        Args:
            section_type: 章节类型（leaf, intro, conclusion）
            
        Returns:
            格式化的写作指导文本
        """
        print(f"\n🧠 [记忆检索] 开始为 {section_type} 类型章节检索写作指导...")
        
        # 根据section_type确定搜索的collection
        collection = section_type if section_type in ["intro", "leaf"] else "intro"
        print(f"   → 目标collection: {collection}")
        
        # 简单的通用搜索查询，获取该类型的所有写作经验
        query = f"写作 修改 编辑"
        print(f"   → 搜索查询: '{query}'")
        
        # 搜索相关记忆
        memories = self.search_relevant_memories(
            query=query, 
            collection=collection,
            max_results=5  # 获取更多的写作原则
        )
        
        print(f"   → 检索到 {len(memories)} 条相关记忆")
        
        if not memories:
            print("   ⚠️ 未找到相关记忆，跳过写作指导")
            return ""
        
        print("   ✅ 检索到的写作法则:")
        for i, memory in enumerate(memories, 1):
            print(f"      {i}. {memory[:80]}{'...' if len(memory) > 80 else ''}")
        
        # 格式化记忆为指导文本
        guidance_text = f"\n\n## 📝 {section_type.upper()}类型章节的写作指导原则:\n"
        for i, memory in enumerate(memories, 1):
            guidance_text += f"{i}. {memory}\n"
        
        print(f"   ✅ 已生成写作指导文本 ({len(guidance_text)} 字符)")
        return guidance_text
    
    def get_edit_principles(self, edit_context: str) -> str:
        """
        获取编辑原则，用于指导内容修改
        
        Args:
            edit_context: 编辑上下文
            
        Returns:
            格式化的编辑原则文本
        """
        # 搜索编辑相关的记忆
        query = f"修改 编辑 {edit_context}"
        
        # 搜索所有集合中的相关记忆
        intro_memories = self.search_relevant_memories(query, collection="intro", max_results=2)
        leaf_memories = self.search_relevant_memories(query, collection="leaf", max_results=2)
        
        all_memories = intro_memories + leaf_memories
        
        if not all_memories:
            return ""
        
        # 格式化为编辑原则
        principles_text = "\n\n## 🎯 基于历史经验的编辑原则:\n"
        for i, memory in enumerate(all_memories, 1):
            principles_text += f"- {memory}\n"
        
        return principles_text

    def get_reflection_checklist(self, section_type: str) -> str:
        """
        获取反思检查清单，用于检验内容是否遵循历史修改建议
        
        Args:
            section_type: 章节类型（leaf, intro, conclusion）
            
        Returns:
            格式化的检查清单文本
        """
        print(f"\n🔍 [记忆检索] 开始为 {section_type} 类型章节生成反思检查清单...")
        
        # 获取该类型的所有编辑经验
        collection = section_type if section_type in ["intro", "leaf"] else "intro"
        print(f"   → 目标collection: {collection}")
        
        # 搜索所有相关的修改建议
        query = f"修改 删除 统一 避免 保持"
        print(f"   → 搜索查询: '{query}'")
        
        memories = self.search_relevant_memories(
            query=query,
            collection=collection,
            max_results=8  # 获取更多检查项
        )
        
        print(f"   → 检索到 {len(memories)} 条修改建议")
        
        if not memories:
            print("   ⚠️ 未找到相关修改建议，跳过检查清单")
            return ""
        
        print("   ✅ 检索到的修改法则:")
        for i, memory in enumerate(memories, 1):
            print(f"      ✓ 检查项{i}: {memory[:60]}{'...' if len(memory) > 60 else ''}")
        
        # 格式化为检查清单
        checklist_text = f"\n\n## 🔍 {section_type.upper()}类型章节反思检查清单:\n"
        checklist_text += "请仔细检查以下每一项是否得到遵循：\n\n"
        
        for i, memory in enumerate(memories, 1):
            # 将记忆转换为检查项格式
            checklist_text += f"✓ 检查项{i}: {memory}\n"
        
        checklist_text += "\n请逐项检查上述原则，如果发现问题请给出具体的修改建议。"
        
        print(f"   ✅ 已生成检查清单 ({len(memories)} 项检查, {len(checklist_text)} 字符)")
        return checklist_text


def add_memory_context_to_prompt(original_prompt: str, memory_manager: WritingMemoryManager, 
                                section_type: str, content_context: str) -> str:
    """
    将记忆上下文添加到原始prompt中
    
    Args:
        original_prompt: 原始prompt
        memory_manager: 记忆管理器实例
        section_type: 章节类型
        content_context: 内容上下文
        
    Returns:
        增强的prompt
    """
    guidance = memory_manager.get_writing_guidance(section_type, content_context)
    
    if guidance:
        enhanced_prompt = original_prompt + guidance + "\n\n---\n以上指导基于历史编辑经验，请在写作时参考这些原则。\n"
        return enhanced_prompt
    
    return original_prompt