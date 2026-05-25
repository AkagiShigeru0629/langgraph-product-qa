"""
多产品参数问答系统 - RAG增强版
==============================
基于 LLM 语义归一化 + AI Agent + 规则引擎 + RAG向量检索
"""
import streamlit as st
import os
import sys
from qa_workflow import build_qa_workflow, create_initial_state
from pathlib import Path
from dotenv import load_dotenv
from memory import AgentMemory
from langchain_ollama import OllamaLLM

# ========== Streamlit兼容的UTF-8编码配置（无冲突） ==========
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['STREAMLIT_SERVER_CHARSET'] = 'utf-8'
# ========== 加载环境变量 ==========
load_dotenv()

# ========== RAG模块导入 ==========
sys.path.append(str(Path(__file__).parent / "demo1"))
try:
    from rag_retriever import RAGRetriever, create_hybrid_answer_func

    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("⚠️ rag_retriever模块未找到，将使用纯规则模式")

# ========== Utils导入 ==========
from utils import (
    load_product_param_db,
    hybrid_answer,
    list_knowledge_files,
    PRODUCT_KEYS,
    PARAM_KEYS,
    load_full_doc_content
)


# ===================== AI Agent核心功能（新增） =====================

def agent_tool_calling(question, param_db, llm_mode):
    from utils import match_product, normalize_question
    product_name = match_product(question, param_db)
    param_names = normalize_question(question, mode=llm_mode)

    results = []
    if product_name and param_names:
        prod_params = param_db.get(product_name, {})
        for param in param_names:
            if param in prod_params:
                results.append(f"{param}：{prod_params[param]}")

    if results:
        return f"{product_name} - {'; '.join(results)}"
    return "未找到相关参数"


def agent_task_planning(task, param_db, full_doc, mode):
    from utils import llm_router, extract_product_intro
    prompt = f"""
    你是任务规划Agent，仅输出JSON数组，无其他文字！
    把复杂任务拆解成3-5个可执行的子任务，子任务要具体、可落地。
    复杂任务：{task}
    示例输入：生成星途Pro产品分析报告
    示例输出：["提取星途Pro核心参数","提取星途Pro产品概述","生成产品分析报告"]
    """
    task_list_str = llm_router(prompt, mode=mode)
    try:
        task_list = eval(task_list_str)
        if not isinstance(task_list, list):
            task_list = ["提取星途Pro核心参数", "提取星途Pro产品概述", "生成产品分析报告"]
    except:
        task_list = ["提取星途Pro核心参数", "提取星途Pro产品概述", "生成产品分析报告"]

    results = []
    for sub_task in task_list:
        if "提取参数" in sub_task:
            param_result = agent_tool_calling("星途Pro的核心参数：电池容量、最大续航、图传距离", param_db, mode)
            results.append(f"【子任务：{sub_task}】\n{param_result}")
        elif "产品概述" in sub_task:
            intro_result = extract_product_intro("星途 Pro 专业级航拍无人机", full_doc)
            results.append(f"【子任务：{sub_task}】\n{intro_result}")
        elif "生成报告" in sub_task:
            report_prompt = f"""
            基于以下信息生成专业的产品分析报告，结构清晰、语言简洁：
            {chr(10).join(results)}
            """
            report_result = llm_router(report_prompt, mode=mode)
            results.append(f"【子任务：{sub_task}】\n{report_result}")
        else:
            sub_result = llm_router(f"完成子任务：{sub_task}", mode=mode)
            results.append(f"【子任务：{sub_task}】\n{sub_result}")

    # 强制UTF-8编码，去除乱码字符（加errors='ignore'避免编码错误）
    final_result = "\n".join([str(res).encode('utf-8', errors='ignore').decode('utf-8') for res in results])
    return final_result


def agent_safe_execution(func, *args, **kwargs):
    import time
    import logging
    # 日志文件指定UTF-8编码
    logging.basicConfig(filename="agent_log.log", level=logging.INFO, encoding='utf-8')

    try:
        start_time = time.time()
        if time.time() - start_time > 30:
            raise TimeoutError("Agent执行超时")
        result = func(*args, **kwargs)
        # 结果也强制编码（加容错）
        clean_result = str(result).encode('utf-8', errors='ignore').decode('utf-8')
        logging.info(f"Agent执行成功：{func.__name__}，耗时：{time.time() - start_time:.2f}s")
        return clean_result
    except TimeoutError as e:
        logging.error(f"Agent执行超时：{str(e)}")
        return "❌ 任务执行超时，请重试"
    except Exception as e:
        logging.error(f"Agent执行失败：{str(e)}")
        return f"❌ 任务执行失败：{str(e)}"


# ===================== 页面配置 =====================
st.set_page_config(
    page_title="多产品参数问答系统（RAG增强版）",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 强制页面UTF-8编码（Streamlit兼容版）
st.markdown("""
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
    <style>
        * {
            font-family: "Microsoft YaHei", "SimHei", "Arial Unicode MS", sans-serif !important;
        }
    </style>
""", unsafe_allow_html=True)

# ===================== 侧边栏 =====================
with st.sidebar:
    st.title("📚 产品知识库")
    llm_mode = "local"
    # 当前配置
    st.markdown("### ⚙️ 当前配置")
    st.info("本地模式 · Ollama qwen:7b")

    st.markdown("---")

    # 向量检索开关
    if st.session_state.get("rag_enabled", False):
        st.markdown("### 🔍 向量检索")
        rag_toggle = st.toggle(
            "启用RAG增强",
            value=True,
            help="启用后可进行语义问答"
        )

    st.markdown("---")

    # 文档状态
    st.markdown("### 📄 文档状态")
    st.success("✅ 已加载知识库文档")

    st.markdown("---")

    # 支持参数
    st.markdown("### 📊 支持参数")
    st.caption("续航、图传距离、重量、尺寸、电池、相机、避障、通信协议等")

    st.markdown("---")

    # 支持产品
    st.markdown("### 📦 支持产品")
    st.caption("星途 Pro 无人机 | 智联云盒 Pro 网关 | 清氧净护 空气净化器")

# ===================== 主界面 =====================
st.title("🚀 多产品参数问答系统（RAG增强版）")
st.markdown("#### 基于 LLM 语义归一化 + AI Agent + 规则引擎 + RAG向量检索")

# 初始化数据
if "param_db" not in st.session_state:
    with st.spinner("🔍 正在加载产品参数库..."):
        st.session_state.param_db = load_product_param_db()
        st.session_state.full_doc = load_full_doc_content()

if "qa_app" not in st.session_state:
    llm = OllamaLLM(model="qwen:7b")
    st.session_state.qa_app = build_qa_workflow(llm=llm)



# 初始化AI Agent
if "agent_memory" not in st.session_state:
    st.session_state.agent_memory = AgentMemory()

# 显示加载状态
prod_count = len(st.session_state.param_db)
st.info(f"✅ 已加载 {prod_count} 款产品 | 当前模式：{llm_mode}")

# 聊天交互区
st.markdown("---")
user_input = st.text_input(
    "💬 请输入你的问题（例如：介绍星途Pro | 星途Pro的电池容量）",
    placeholder="输入问题后按回车提问...",
    key="user_input"
)

# ==================== 问答逻辑（LangGraph工作流版） ====================
if user_input:
    with st.spinner(f"🤖 正在{llm_mode}模式下处理..."):
        clean_input = str(user_input).encode('utf-8', errors='ignore').decode('utf-8')
        st.session_state.agent_memory.add_memory("user", clean_input)

        # 代词消解：把"它"替换成具体产品名
        resolved_input = st.session_state.agent_memory.resolve_pronouns(clean_input)

        # 使用LangGraph工作流
        state = create_initial_state(resolved_input)
        result = st.session_state.qa_app.invoke(state)
        answer = result["answer"]

        # 记录到记忆
        st.session_state.agent_memory.add_memory("assistant", str(answer))

        # 显示结果
        st.markdown("---")
        st.markdown(f"**📝 回答（{result['intent']} | 相关性：{result['relevance_score']:.2f}）**")
        st.success(str(answer))


# 示例问题
st.markdown("---")
st.markdown("### 💡 示例问题：")
example_questions = [
    "无人机的续航时间是多少",
    "星途Pro的避障功能",
    "智能网关支持哪些网络",
    "空气净化器的适用面积"
]


for idx, q in enumerate(example_questions):
    if st.button(q, key=f"btn_{idx}"):
        with st.spinner(f"🤖 正在处理..."):
            clean_q = str(q).encode('utf-8', errors='ignore').decode('utf-8')

            state = create_initial_state(clean_q)
            result = st.session_state.qa_app.invoke(state)

            st.session_state.agent_memory.add_memory("user", clean_q)
            st.session_state.agent_memory.add_memory("assistant", result["answer"])

            st.markdown("### 📝 回答：")
            st.success(result["answer"])

# ===================== AI Agent复杂任务（彻底修复乱码） =====================
st.markdown("---")
st.markdown("### 🚀 AI Agent 复杂任务")

if st.button("📊 生成星途Pro产品分析报告", key="agent_report"):
    with st.spinner("🤖 Agent 正在规划并执行任务..."):
        task = "生成星途Pro产品分析报告，包含核心参数和产品概述"
        clean_task = str(task).encode('utf-8', errors='ignore').decode('utf-8')
        # 使用新的Agent处理复杂任务
        if "product_agent" not in st.session_state:
            from utils import ProductParamAgent

            st.session_state.product_agent = ProductParamAgent(mode=llm_mode)

        report = agent_safe_execution(
            st.session_state.product_agent.answer_question,
            clean_task,
            st.session_state.param_db,
            st.session_state.full_doc
        )

        clean_report = str(report).encode('utf-8', errors='ignore').decode('utf-8')
        st.markdown("### 📝 产品分析报告")
        st.success(clean_report)

# 对话记忆（纯文本渲染，无乱码）
st.markdown("---")
st.markdown("### 📜 对话记忆（Agent上下文）")
if st.button("清空记忆", key="clear_memory"):
    st.session_state.agent_memory.clear_memory()
    st.success("✅ 已清空 Agent 记忆！")

memory_history = st.session_state.agent_memory.get_memory()
if memory_history:
    st.subheader("历史对话：")
    for idx, msg in enumerate(memory_history):
        st.write(f"**第{idx + 1}条 - {msg['role']}：**")
        st.text(msg['content'])  # 强制纯文本渲染，避免乱码
        st.divider()
else:
    st.info("暂无对话记忆")

# 页脚
st.markdown("---")
st.markdown("© 2026 多产品参数问答系统 | RAG增强版 · 企业级架构")
