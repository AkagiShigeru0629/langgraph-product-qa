# -*- coding: utf-8 -*-
"""
LangGraph 工作流集成示例
=========================
演示如何在 app.py 中集成 qa_workflow

使用方法：
    python src/qa_workflow_integration.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import os

# ========== 路径配置 ==========
import sys
from pathlib import Path

# 将项目根目录添加到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ========== 导入 LangGraph 工作流 ==========
from demo1.qa_workflow import build_qa_workflow, create_initial_state, ask_question

# ========== LLM 配置 ==========
try:
    from langchain_ollama import OllamaLLM
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("⚠️ langchain_ollama 未安装，将使用降级模式")


# ============================================================================
# 方式1：Streamlit 集成（推荐）
# ============================================================================

def create_streamlit_app():
    """
    创建 Streamlit 应用的示例代码
    
    将此代码添加到 app.py 中即可
    """
    
    # ===================== 页面配置 =====================
    st.set_page_config(
        page_title="多产品参数问答系统（LangGraph版）",
        page_icon="🚁",
        layout="wide"
    )
    
    # ===================== 标题 =====================
    st.title("🚀 多产品参数问答系统（LangGraph版）")
    st.markdown("#### 基于 LangGraph 工作流的智能问答系统")
    
    # ===================== 工作流初始化 =====================
    @st.cache_resource
    def get_qa_app():
        """缓存工作流实例，避免重复构建"""
        if LLM_AVAILABLE:
            llm = OllamaLLM(model="qwen2.5")
        else:
            llm = None
        return build_qa_workflow(llm)
    
    qa_app = get_qa_app()
    
    # ===================== 侧边栏 =====================
    with st.sidebar:
        st.title("📚 使用说明")
        st.markdown("""
        **支持的问题类型：**
        - 产品参数查询（如：续航、重量）
        - 产品介绍（如：功能、特点）
        - 产品对比
        - 闲聊问候
        
        **示例问题：**
        - 星途Pro的续航是多少？
        - 网关有哪些参数？
        - 星途Pro无人机怎么样？
        """)
        
        st.markdown("---")
        st.markdown("### 🔧 工作流状态")
        if LLM_AVAILABLE:
            st.success("✅ LLM 已连接")
        else:
            st.warning("⚠️ LLM 未连接，使用规则模式")
    
    # ===================== 主界面 =====================
    question = st.text_input(
        "💬 请输入您的问题：",
        placeholder="例如：星途Pro的续航是多少？"
    )
    
    if question:
        with st.spinner("🤔 正在思考..."):
            # 创建初始状态
            initial_state = create_initial_state(question)
            
            # 执行工作流
            result = qa_app.invoke(initial_state)
        
        # ===================== 显示结果 =====================
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("### 💬 回答")
            st.info(result["answer"])
        
        with col2:
            st.markdown("### 📊 信息")
            st.write(f"**意图：** {result['intent']}")
            st.write(f"**置信度：** {result['confidence']}")
            st.write(f"**相关性：** {result['relevance_score']:.2f}")
            
            if result["sources"]:
                st.markdown("**来源：**")
                for source in result["sources"]:
                    st.caption(source)


# ============================================================================
# 方式2：独立脚本使用
# ============================================================================

def run_standalone_demo():
    """
    独立运行演示（无 Streamlit）
    """
    print("="*60)
    print("🤖 LangGraph 问答系统演示")
    print("="*60)
    
    # 测试问题
    test_questions = [
        "星途Pro的续航是多少？",
        "网关有哪些参数？",
        "你好",
        "星途Pro无人机怎么样？"
    ]
    
    for q in test_questions:
        print("\n" + "="*60)
        result = ask_question(q, llm=None, verbose=True)
        print("="*60)


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--streamlit":
        # 启动 Streamlit 应用
        sys.argv = [sys.argv[0], "server", "--reload"]
        create_streamlit_app()
    else:
        # 运行独立演示
        run_standalone_demo()
