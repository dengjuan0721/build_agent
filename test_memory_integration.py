#!/usr/bin/env python3
"""
测试长期记忆集成功能
验证WritingMemoryManager是否能正确检索和格式化记忆内容
"""
import asyncio
from dotenv import load_dotenv

# 加载环境变量
load_dotenv(dotenv_path="/Users/dengjuan1/build_agent/.env", override=True)

from langgraph_writer.memory_manager import WritingMemoryManager

async def test_memory_integration():
    """测试记忆管理器的基本功能"""
    print("=== 测试长期记忆集成功能 ===\n")
    
    try:
        # 1. 初始化记忆管理器
        print("1. 初始化记忆管理器...")
        memory_manager = WritingMemoryManager()
        print("✅ 记忆管理器初始化成功\n")
        
        # 2. 测试写作指导检索
        print("2. 测试写作指导检索...")
        
        for section_type in ["leaf", "intro"]:
            print(f"\n--- 测试 {section_type} 类型 ---")
            guidance = memory_manager.get_writing_guidance(section_type)
            
            if guidance:
                print(f"✅ 成功检索到{section_type}类型的写作指导:")
                print(guidance[:200] + "..." if len(guidance) > 200 else guidance)
            else:
                print(f"⚠️ 未检索到{section_type}类型的写作指导")
        
        # 3. 测试反思检查清单
        print("\n3. 测试反思检查清单...")
        
        for section_type in ["leaf", "intro"]:
            print(f"\n--- 测试 {section_type} 类型反思检查 ---")
            checklist = memory_manager.get_reflection_checklist(section_type)
            
            if checklist:
                print(f"✅ 成功生成{section_type}类型的检查清单:")
                print(checklist[:300] + "..." if len(checklist) > 300 else checklist)
            else:
                print(f"⚠️ 未生成{section_type}类型的检查清单")
        
        # 4. 测试编辑原则检索
        print("\n4. 测试编辑原则检索...")
        edit_principles = memory_manager.get_edit_principles("文本修改")
        
        if edit_principles:
            print("✅ 成功检索到编辑原则:")
            print(edit_principles[:300] + "..." if len(edit_principles) > 300 else edit_principles)
        else:
            print("⚠️ 未检索到编辑原则")
        
        print("\n=== 记忆集成测试完成 ===")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_graph_initialization():
    """测试图组件的记忆管理器初始化"""
    print("\n=== 测试图组件初始化 ===\n")
    
    try:
        # 测试 SectionWriterGraph
        print("1. 测试 SectionWriterGraph...")
        from langgraph_writer import SectionWriterGraph
        
        with SectionWriterGraph() as writer:
            if hasattr(writer, 'memory_manager') and writer.memory_manager:
                print("✅ SectionWriterGraph 记忆管理器初始化成功")
            else:
                print("⚠️ SectionWriterGraph 记忆管理器未正确初始化")
        
        # 测试 ConnectorWriterGraph  
        print("2. 测试 ConnectorWriterGraph...")
        from langgraph_writer import ConnectorWriterGraph
        
        with ConnectorWriterGraph() as connector:
            if hasattr(connector, 'memory_manager') and connector.memory_manager:
                print("✅ ConnectorWriterGraph 记忆管理器初始化成功")
            else:
                print("⚠️ ConnectorWriterGraph 记忆管理器未正确初始化")
                
        print("\n=== 图组件初始化测试完成 ===")
        
    except Exception as e:
        print(f"❌ 图组件测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 运行异步测试
    asyncio.run(test_memory_integration())
    
    # 运行同步测试
    test_graph_initialization()