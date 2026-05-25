# -*- coding: utf-8 -*-
"""
LangGraph 问答工作流
====================
用 LangGraph 组织的 RAG 问答流程

工作流：用户提问 → 意图识别 → RAG检索 → 评估相关性 → 生成回答

使用方法：
    from qa_workflow import build_qa_workflow, create_initial_state

    # 1. 构建工作流
    llm = OllamaLLM(model="qwen:7b")
    app = build_qa_workflow(llm)

    # 2. 创建初始状态
    state = create_initial_state("星途Pro的续航是多少？")

    # 3. 执行工作流
    result = app.invoke(state)

    # 4. 获取结果
    print(result["answer"])
    print(result["confidence"])
"""
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import warnings
import operator
from typing import Annotated, Sequence, Literal, TypedDict, List, Dict, Any, Optional

# ========== 警告过滤（LangChain 兼容性） ==========
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="langchain")
warnings.filterwarnings("ignore", module="langchain_core")

# ========== LangGraph 核心导入 ==========
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# ========== LLM 导入 ==========
try:
    from langchain_ollama import OllamaLLM

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️ langchain_ollama 未安装，将使用模拟模式")

# ========== 项目内部导入 ==========
import sys
from pathlib import Path

# 将 src 目录添加到路径
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

# 尝试导入 RAG 检索器
try:
    from demo1.rag_retriever import RAGRetriever

    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("⚠️ rag_retriever 未找到，RAG 功能不可用")

# 尝试导入工具函数
try:
    from utils import match_product, PARAM_KEYS, PRODUCT_KEYS, INTRO_KEYWORDS

    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    PARAM_KEYS = ["续航", "重量", "尺寸", "电池", "图传", "像素"]
    PRODUCT_KEYS = ["星途", "无人机", "网关", "净化器"]
    INTRO_KEYWORDS = ["介绍", "概述", "简介", "功能", "特点"]

# ========== 全局统一参数关键词字典（所有函数共用，避免遗漏） ==========
PARAM_SYNONYMS = {
    "续航": ["续航", "续航时间", "飞行时间", "工作时间", "电池续航", "能飞多久", "能飞多长时间", "飞多久"],
    "重量": ["重量", "起飞重量", "净重"],
    "尺寸": ["尺寸", "展开尺寸", "折叠尺寸", "大小"],
    "电池": ["电池", "电池容量", "mAh"],
    "图传": ["图传", "图传距离", "传输距离", "信号距离"],
    "像素": ["像素", "分辨率", "摄像头", "相机", "CMOS", "传感器"],
    "速度": ["速度", "飞行速度", "最大速度"],
    "距离": ["距离", "图传距离", "飞行距离", "传输距离"],
    "避障": ["避障", "避障系统", "视觉避障", "红外避障"],
    "网络": ["网络", "5G", "4G", "Wi-Fi", "wifi", "通信", "连接", "蜂窝", "LTE"],
    "协议": ["协议", "通信协议", "Modbus", "MQTT", "OPC", "HTTP"],
    "噪音": ["噪音", "噪音水平", "分贝", "dB", "静音"],
    "面积": ["面积", "适用面积", "覆盖面积"],
    "温度": ["温度", "工作温度"],
    "防水": ["防水", "防水等级", "IP"],
    "功率": ["功率", "额定功率", "功耗"],
    "滤芯": ["滤芯", "HEPA", "过滤网", "滤网"],
    "接口": ["接口", "USB", "RS485", "以太网", "串口"],
    "GPS": ["GPS", "定位", "RTK", "GNSS"],
    "价格": ["价格", "售价", "多少钱"],
    "CADR": ["CADR", "cadr", "洁净空气量"],
    "CCM": ["CCM", "ccm", "累计净化量"],
    "甲醛": ["甲醛", "HCHO", "除甲醛"],
    "PM2.5": ["PM2.5", "pm2.5", "颗粒物"],
    "适用面积": ["适用面积", "覆盖面积", "适用空间"],
}


# ============================================================================
# 第一部分：State 定义
# ============================================================================

class QAState(TypedDict):
    """
    问答工作流的统一状态

    这个类定义了工作流中所有节点共享的数据结构。
    每个节点可以读取这些数据，也可以返回更新的数据。
    """

    # ========== 输入 ==========
    question: str  # 用户原始问题

    # ========== 中间状态（节点之间传递的数据）==========
    intent: str  # 识别的意图类型
    intent_confidence: float  # 意图置信度 (0-1)
    retrieved_context: str  # RAG 检索到的上下文
    relevance_score: float  # 相关性评分 (0-1)

    # ========== 输出 ==========
    answer: str  # 最终回答
    confidence: str  # 回答置信度：高/中/低
    sources: List[str]  # 引用的知识来源

    # ========== 对话历史（用于多轮对话扩展）==========
    messages: Annotated[list, add_messages]  # 对话消息历史


# ============================================================================
# 第二部分：辅助函数
# ============================================================================

def get_rag_retriever() -> Optional[RAGRetriever]:
    """
    获取 RAG 检索器实例

    Returns:
        RAGRetriever 实例，如果不可用则返回 None
    """
    if not RAG_AVAILABLE:
        return None

    try:
        rag = RAGRetriever()
        # 确保索引已构建
        rag.build_index()
        return rag
    except Exception as e:
        print(f"⚠️ RAG 初始化失败: {e}")
        return None


def create_initial_state(question: str) -> Dict[str, Any]:
    """
    创建问答工作流的初始状态

    Args:
        question: 用户问题

    Returns:
        初始状态字典
    """
    return {
        "question": question,
        "intent": "",
        "intent_confidence": 0.0,
        "retrieved_context": "",
        "relevance_score": 0.0,
        "answer": "",
        "confidence": "中",
        "sources": [],
        "messages": []
    }


# ============================================================================
# 第三部分：核心节点实现
# ============================================================================

# -------------------- 节点1：意图分类 --------------------

def intent_classifier(state: QAState) -> Dict[str, Any]:
    """
    意图分类节点

    分析用户问题，判断用户想要做什么

    支持的意图类型：
    - product_param_query: 产品参数查询
    - product_intro: 产品介绍查询
    - comparison: 产品对比
    - general_chat: 闲聊/问候
    - unclear: 无法识别

    Args:
        state: 当前状态，包含 question

    Returns:
        更新后的状态，包含 intent 和 intent_confidence
    """
    question = state["question"]
    question_lower = question.lower()

    # 1. 检查是否闲聊/问候
    greeting_keywords = ["你好", "hello", "hi", "嗨", "在吗", "帮忙", "谢谢", "请问"]
    if any(kw in question_lower for kw in greeting_keywords) and len(question) < 10:
        return {
            "intent": "general_chat",
            "intent_confidence": 0.95,
            "messages": [{"role": "user", "content": f"识别到闲聊意图: {question}"}]
        }

    # 2. 检查是否为产品介绍类查询
    intro_keywords = ["介绍", "概述", "简介", "产品介绍", "详情", "功能", "特点", "亮点", "优势"]
    if any(kw in question_lower for kw in intro_keywords):
        return {
            "intent": "product_intro",
            "intent_confidence": 0.85,
            "messages": [{"role": "user", "content": f"识别到产品介绍意图: {question}"}]
        }

    # 3. 检查是否为产品对比
    comparison_keywords = ["对比", "比较", "区别", "哪个好", "差异", "对比", "差异"]
    if any(kw in question_lower for kw in comparison_keywords):
        return {
            "intent": "comparison",
            "intent_confidence": 0.80,
            "messages": [{"role": "user", "content": f"识别到产品对比意图: {question}"}]
        }

    # 4. 检查是否为产品参数查询（最常见）
    # 从全局字典提取所有关键词（扁平化）
    param_keywords = [kw for synonyms in PARAM_SYNONYMS.values() for kw in synonyms]
    product_keywords = ["星途", "无人机", "网关", "净化器", "智联云盒", "清氧"]

    has_param = any(kw in question_lower for kw in param_keywords)
    has_product = any(kw in question_lower for kw in product_keywords)

    if has_param and has_product:
        return {
            "intent": "product_param_query",
            "intent_confidence": 0.90,
            "messages": [{"role": "user", "content": f"识别到参数查询意图: {question}"}]
        }

    if has_product:
        return {
            "intent": "product_intro",
            "intent_confidence": 0.70,
            "messages": [{"role": "user", "content": f"识别到产品相关意图: {question}"}]
        }

    # 5. 无法识别

    return {
        "intent": "unclear",
        "intent_confidence": 0.50,
        "messages": [{"role": "user", "content": f"无法识别意图: {question}"}]
    }


# -------------------- 节点2：RAG 检索 --------------------

def rag_retriever_node(state: QAState) -> Dict[str, Any]:
    """
    RAG 检索节点

    使用向量检索从知识库中获取与问题相关的内容

    Args:
        state: 当前状态，包含 question

    Returns:
        更新后的状态，包含 retrieved_context
    """
    question = state["question"]
    intent = state["intent"]

    # 如果是闲聊，不需要 RAG 检索
    if intent == "general_chat":
        return {"retrieved_context": ""}

    # 获取 RAG 检索器
    rag = get_rag_retriever()

    if rag is None:
        # RAG 不可用时的降级处理
        print("⚠️ RAG 不可用，使用降级模式")
        return {"retrieved_context": ""}

    try:
        # 执行检索
        context = rag.retrieve(question, top_k=10)
        print(f"✅ RAG 检索完成，获取到 {len(context)} 字符的上下文")
        return {"retrieved_context": context}
    except Exception as e:
        print(f"⚠️ RAG 检索失败: {e}")
        return {"retrieved_context": ""}


# -------------------- 节点3：相关性评估（优化版）--------------------

def relevance_evaluator(state: QAState) -> Dict[str, Any]:
    """
    相关性评估节点 - 优化版

    改进点：
    1. 增加参数关键词的权重
    2. 增加语义相似度判断（同义词识别）

    评分标准：
    - score >= 0.5: 高相关
    - 0.3 <= score < 0.5: 中相关
    - score < 0.3: 低相关
    """
    question = state["question"]
    context = state["retrieved_context"]
    intent = state["intent"]

    if intent == "general_chat":
        return {"relevance_score": 1.0}

    if not context:
        return {"relevance_score": 0.0}

    question_lower = question.lower()
    context_lower = context.lower()

    # ===== 1. 参数关键词匹配（核心改进） =====
    param_synonyms = PARAM_SYNONYMS

    matched_params = []

    for param, synonyms in param_synonyms.items():
        question_has_param = any(syn in question_lower for syn in synonyms)
        context_has_param = any(syn in context_lower for syn in synonyms)

        if question_has_param and context_has_param:
            matched_params.append(param)

    # 匹配到参数就给高分，1个=0.7，2个=0.9，3个+=1.0
    if len(matched_params) == 1:
        param_match_score = 0.7
    elif len(matched_params) == 2:
        param_match_score = 0.9
    elif len(matched_params) >= 3:
        param_match_score = 1.0
    else:
        param_match_score = 0.0

    if matched_params:
        print(f"📊 匹配到参数关键词: {matched_params}")

    # ===== 2. 通用关键词覆盖率 =====
    # 中文不用空格分词，改用2字滑窗
    question_keywords = list(set(question_lower[i:i + 2] for i in range(len(question_lower) - 1)))
    matched_keywords = sum(1 for kw in question_keywords if kw in context_lower)
    keyword_score = matched_keywords / max(len(question_keywords), 1)

    # ===== 3. 上下文长度 =====
    length_score = min(len(context) / 200, 1.0)

    # ===== 4. 综合评分 =====
    relevance_score = param_match_score * 0.5 + keyword_score * 0.3 + length_score * 0.2

    print(
        f"📊 相关性评估: 参数匹配={param_match_score:.2f}, 关键词={keyword_score:.2f}, 长度={length_score:.2f}, 综合={relevance_score:.2f}")

    return {"relevance_score": relevance_score}


# -------------------- 节点4：回答生成 --------------------

def answer_generator(state: QAState, llm=None) -> Dict[str, Any]:
    """
    回答生成节点

    基于意图和上下文生成最终回答

    Args:
        state: 当前状态
        llm: 语言模型实例

    Returns:
        更新后的状态，包含 answer 和 confidence
    """
    question = state["question"]
    intent = state["intent"]
    context = state["retrieved_context"]
    relevance = state["relevance_score"]
    question_lower = question.lower()

    # 参数关键词提取
    all_param_keywords = PARAM_SYNONYMS
    target_params = []
    for param, keywords in all_param_keywords.items():
        if any(kw in question_lower for kw in keywords):
            target_params.append(param)

    known_products = ["无人机", "星途", "网关", "智联", "净化器", "清氧"]
    mentions_known = any(kw in question_lower for kw in known_products)
    outside_products = ["特斯拉", "苹果", "华为", "小米", "手机", "汽车"]
    mentions_outside = any(kw in question_lower for kw in outside_products)

    # ========== 分意图处理 ==========

    # 1. 闲聊回复
    if intent == "general_chat":
        chat_responses = {
            "你好": "您好！我是智能产品问答助手，可以帮您查询产品参数和介绍。有什么可以帮您的吗？",
            "hi": "Hi! I'm your product Q&A assistant. How can I help you today?",
            "hello": "Hello! 有什么产品问题我可以帮您解答吗？",
        }
        for keyword, response in chat_responses.items():
            if keyword in question.lower():
                return {
                    "answer": response,
                    "confidence": "高",
                    "sources": [],
                    "messages": [{"role": "assistant", "content": "闲聊回复"}]
                }
        return {
            "answer": "您好！我是智能产品问答助手，支持查询产品参数、功能介绍等信息。请问有什么可以帮您？",
            "confidence": "高",
            "sources": [],
            "messages": [{"role": "assistant", "content": "默认闲聊回复"}]
        }

    # 2. 无关产品拒绝
    if mentions_outside and not mentions_known:
        return {
            "answer": "抱歉，知识库中没有该产品的相关信息。我目前支持查询：星途Pro无人机、智联云盒网关、清氧净护净化器。",
            "confidence": "低",
            "sources": [],
            "messages": [{"role": "assistant", "content": "无关产品拒绝"}]
        }

    # 3. 无上下文时的降级处理
    if not context:
        return {
            "answer": "抱歉，知识库中暂时没有找到与您问题相关的内容。\n\n您可以尝试：\n1. 换个方式描述您的问题\n2. 咨询具体的产品参数名称\n3. 稍后再试",
            "confidence": "低",
            "sources": [],
            "messages": [{"role": "assistant", "content": "无上下文降级回复"}]
        }

    # 3.5 只提产品没提参数 → 返回产品简介
    intro_words = ["介绍", "功能", "有什么", "特点", "概述", "简介"]
    is_intro_query = any(kw in question_lower for kw in intro_words)

    if mentions_known and not target_params:
        for line in context.split('\n'):
            line_clean = line.replace('[产品说明-无人机.txt]', '').replace('[产品说明-智能网关.txt]', '').replace(
                '[产品说明-空气净化器.txt]', '').replace('---', '').strip()
            if any(kw in line_clean for kw in ['是一款', '是面向', '是一款集']):
                product_name = ""
                if any(kw in question_lower for kw in ["无人机", "星途"]):
                    product_name = "星途Pro无人机"
                elif any(kw in question_lower for kw in ["网关", "智联"]):
                    product_name = "智联云盒网关"
                elif any(kw in question_lower for kw in ["净化器", "清氧"]):
                    product_name = "清氧净护净化器"
                return {
                    "answer": f"{product_name}：{line_clean}",
                    "confidence": "中",
                    "sources": extract_sources(context),
                    "messages": [{"role": "assistant", "content": "产品介绍"}]
                }

    # 3.6 只提参数没提产品 → 引导指定产品
    if target_params and not mentions_known:
        param_name = '、'.join(target_params)
        return {
            "answer": f"您想了解哪个产品的{param_name}？请指定产品名称，例如：\n- 星途Pro无人机的{target_params[0]}\n- 智联云盒网关的{target_params[0]}\n- 清氧净护净化器的{target_params[0]}",
            "confidence": "中",
            "sources": [],
            "messages": [{"role": "assistant", "content": "缺产品提示"}]
        }

    # 3.7 低相关时的混合回答
    if relevance < 0.4:
        fallback_response = generate_fallback_answer(question, context, llm)
        return fallback_response

    # 4. 正常回答生成（高/中相关）
    # 先过滤出与问题相关的上下文
    filtered_context = filter_relevant_context(question, context)

    prompt = f"""基于以下产品知识回答用户问题。

    【产品知识库】
    {filtered_context}

    【用户问题】
    {question}

    【回答要求】
    1. 直接回答用户问题，不要输出无关信息
    2. 如果问题问的是具体参数，只回答该参数值
    3. 如果知识库中没有相关信息，请明确告知
    4. 如果多个检索结果包含相同信息，只保留最完整的一条，不要重复罗列
    
    【回答】"""

    # 如果有 LLM，使用 LLM 生成
    print(f"🔍 DEBUG - llm是否为None: {llm is None}, target_params: {target_params}, mentions_known: {mentions_known}")
    if llm is not None:
        try:
            answer = llm.invoke(prompt)
            if isinstance(answer, str):
                answer = answer.strip()
            else:
                answer = str(answer)
        except Exception as e:
            print(f"⚠️ LLM 生成失败: {e}")
            answer = generate_simple_answer(question, context)
    else:
        answer = generate_simple_answer(question, context)

    # 清理文档标记
    for tag in ['[产品说明-无人机.txt]', '[产品说明-智能网关.txt]', '[产品说明-空气净化器.txt]', '---']:
        answer = answer.replace(tag, '')
    answer = answer.strip()

    # 判断置信度
    confidence = "高" if relevance >= 0.7 else "中"

    # 提取来源
    sources = extract_sources(context)

    return {
        "answer": answer,
        "confidence": confidence,
        "sources": sources,
        "messages": [{"role": "assistant", "content": f"生成{confidence}置信度回答"}]
    }


# -------------------- 节点5：降级处理（优化版）--------------------

def fallback_handler(state: QAState) -> Dict[str, Any]:
    """
    降级处理节点 - 优化版

    改进点：
    1. 优先从 RAG 检索结果提取相关参数
    2. 规则引擎作为补充
    """
    question = state["question"]
    context = state.get("retrieved_context", "")  # 获取RAG检索结果
    question_lower = question.lower()

    # ===== 0. 无关产品检查 =====
    outside_products = ["特斯拉", "苹果", "华为", "小米", "手机", "汽车", "笔记本", "电脑", "平板", "冰箱", "空调",
                        "洗衣机", "电视"]
    known_products = ["无人机", "星途", "网关", "智联", "净化器", "清氧"]
    mentions_outside = any(kw in question_lower for kw in outside_products)
    mentions_known = any(kw in question_lower for kw in known_products)
    if mentions_outside and not mentions_known:
        return {
            "answer": "抱歉，知识库中没有该产品的相关信息。我目前支持查询以下产品：星途Pro无人机、智联云盒网关、清氧净护净化器。",
            "confidence": "低",
            "sources": [],
            "messages": [{"role": "assistant", "content": "无关产品拒绝"}]
        }

    # ===== 1. 提取问题中的参数关键词 =====
    param_keywords = PARAM_SYNONYMS

    target_params = []
    for param, keywords in param_keywords.items():
        if any(kw in question_lower for kw in keywords):
            target_params.append(param)

    print(f"📊 降级处理 - 目标参数: {target_params}")

    # ===== 只提参数没提产品 =====
    if target_params and not mentions_known:
        return {
            "answer": f"您想了解哪个产品的{'、'.join(target_params)}？请指定产品名称，例如：\n- 星途Pro无人机的{target_params[0]}\n- 智联云盒网关的{target_params[0]}\n- 清氧净护净化器的{target_params[0]}",
            "confidence": "中",
            "sources": [],
            "messages": [{"role": "assistant", "content": "缺产品提示"}]
        }

    # =====  只提产品没提参数（介绍类） =====
    intro_words = ["介绍", "功能", "有什么", "特点", "概述", "简介"]
    only_product = mentions_known and not target_params
    is_intro = any(kw in question_lower for kw in intro_words)

    if only_product or is_intro:
        # 从RAG检索结果提取产品简介
        if context:
            product_name = ""
            for kw in known_products:
                if kw in question_lower:
                    if kw in ["无人机", "星途"]:
                        product_name = "星途Pro无人机"
                    elif kw in ["网关", "智联"]:
                        product_name = "智联云盒网关"
                    elif kw in ["净化器", "清氧"]:
                        product_name = "清氧净护净化器"
                    break

            # 找产品概述行
            for line in context.split('\n'):
                line_clean = line.replace('[产品说明-无人机.txt]', '').replace('[产品说明-智能网关.txt]', '').replace(
                    '[产品说明-空气净化器.txt]', '').replace('---', '').strip()
                if '是一款' in line_clean or '是面向' in line_clean or '产品概述' in line_clean or '是一款集' in line_clean:
                    return {
                        "answer": f"{product_name}：{line_clean}",
                        "confidence": "中",
                        "sources": ["RAG检索"],
                        "messages": [{"role": "assistant", "content": "产品介绍"}]
                    }

            # 没找到概述，返回产品名称
            if product_name:
                return {
                    "answer": f"{product_name}是知识库中的产品之一，请问您想了解哪个方面的信息？（如：参数、功能、特点等）",
                    "confidence": "中",
                    "sources": [],
                    "messages": [{"role": "assistant", "content": "产品名引导"}]
                }

    # ===== 2. 优先从 RAG 检索结果提取 =====
    if context:
        relevant_lines = []
        for line in context.split('\n'):
            for param in target_params:
                keywords = param_keywords.get(param, [param])
                if any(kw in line.lower() for kw in keywords):
                    relevant_lines.append(line.strip())
                    break

        # ===== 关键词直查：专业术语语义匹配差，直接用关键词 =====
        protocol_keywords = ["Modbus", "MQTT", "OPC", "HTTP"]
        if any(kw in question_lower for kw in ["协议", "protocol"]) or any(
                kw in question_lower for kw in [pk.lower() for pk in protocol_keywords]):
            for line in context.split('\n'):
                if any(kw in line for kw in protocol_keywords):
                    if line.strip() not in relevant_lines:
                        relevant_lines.insert(0, line.strip())  # 插到最前面

        if relevant_lines:
            relevant_lines = list(dict.fromkeys(relevant_lines))[:5]
            answer_lines = "\n".join([f"- {line}" for line in relevant_lines if line])
            if answer_lines:
                return {
                    "answer": f"根据知识库检索结果：\n{answer_lines}",
                    "confidence": "中",
                    "sources": ["RAG检索"],
                    "messages": [{"role": "assistant", "content": "降级回复-RAG提取"}]
                }

    # ===== 3. 规则引擎兜底 =====
    try:
        from utils import load_product_param_db, match_product
        param_db = load_product_param_db()
        product = match_product(question, param_db)

        if product:
            params = param_db.get(product, {})

            if params and target_params:
                filtered_params = {}
                for param in target_params:
                    for key, value in params.items():
                        if param in key or key in param:
                            filtered_params[key] = value

                if filtered_params:
                    param_list = "\n".join([f"- {k}: {v}" for k, v in filtered_params.items()])
                    return {
                        "answer": f"根据查询，{product}的相关参数：\n{param_list}",
                        "confidence": "中",
                        "sources": ["产品参数数据库"],
                        "messages": [{"role": "assistant", "content": "降级回复-规则引擎"}]
                    }
    except Exception as e:
        print(f"⚠️ 规则引擎调用失败: {e}")

    return {
        "answer": "抱歉，我无法理解您的问题或知识库中没有相关信息。\n\n您可以尝试：\n1. 明确说出产品名称（如：星途Pro）\n2. 说出具体想了解的参数（如：续航、重量）\n3. 换个方式描述您的问题",
        "confidence": "低",
        "sources": [],
        "messages": [{"role": "assistant", "content": "最终降级回复"}]
    }


# ============================================================================
# 第四部分：辅助生成函数
# ============================================================================
def filter_relevant_context(question: str, context: str) -> str:
    """
    过滤出与问题相关的上下文行

    改进：
    1. 用文档标记判断产品归属（不再逐行找产品名）
    2. 参数精准匹配（关键词在冒号前）
    3. 无关产品拒绝
    4. 产品介绍类查询支持
    """
    question_lower = question.lower()

    # 参数关键词映射
    param_keywords = PARAM_SYNONYMS

    # 文档标记 → 产品名映射
    doc_to_product = {
        "无人机": "无人机",
        "智能网关": "网关",
        "空气净化器": "净化器",
    }

    # 产品名 → 问题中可能出现的关键词
    product_query_keywords = {
        "无人机": ["无人机", "星途", "pro"],
        "网关": ["网关", "智联"],
        "净化器": ["净化器", "清氧"],
    }

    # 介绍类关键词
    intro_keywords = ["介绍", "概述", "简介", "功能", "特点", "有什么"]

    # 1. 找出问题想问的参数
    target_params = []
    for param, keywords in param_keywords.items():
        if any(kw in question_lower for kw in keywords):
            target_params.append(param)

    # 2. 找出问题提到的产品
    target_products = []
    for product, keywords in product_query_keywords.items():
        if any(kw in question_lower for kw in keywords):
            target_products.append(product)

    # 3. 无关产品检查
    known_product_words = ["无人机", "星途", "网关", "智联", "净化器", "清氧"]
    question_mentions_product = any(kw in question_lower for kw in known_product_words)
    outside_product_words = ["特斯拉", "苹果", "华为", "小米", "手机", "汽车", "笔记本", "电脑", "平板", "冰箱", "空调",
                             "洗衣机", "电视"]
    if any(kw in question_lower for kw in outside_product_words) and not question_mentions_product:
        return ""

    # 4. 按文档标记分段，只保留目标产品的段落
    sections = {}
    current_product = None
    current_lines = []

    for line in context.split('\n'):
        # 检查是否是文档标记行
        found_doc = False
        for doc_keyword, product in doc_to_product.items():
            if doc_keyword in line:
                # 保存上一段
                if current_product and current_lines:
                    if current_product not in sections:
                        sections[current_product] = []
                    sections[current_product].extend(current_lines)
                current_product = product
                current_lines = []
                found_doc = True
                break

        if not found_doc:
            line_clean = line.replace('---', '').strip()
            if line_clean:
                current_lines.append(line_clean)

    # 保存最后一段
    if current_product and current_lines:
        if current_product not in sections:
            sections[current_product] = []
        sections[current_product].extend(current_lines)

    # 5. 只取目标产品的段落
    if target_products:
        target_lines = []
        for product in target_products:
            target_lines.extend(sections.get(product, []))
    else:
        # 没指定产品，合并所有
        target_lines = []
        for lines in sections.values():
            target_lines.extend(lines)

    if not target_lines:
        return context[:500]

    # 6. 在目标段落里做参数过滤
    is_intro = any(kw in question_lower for kw in intro_keywords)
    key_lines = []  # 参数精准行
    product_lines = []  # 产品相关行（用于介绍类）

    import re

    for line in target_lines:
        product_lines.append(line)

        if target_params:
            for param in target_params:
                keywords = param_keywords.get(param, [param])
                if any(kw in line for kw in keywords):
                    is_key_line = False

                if any(kw in line for kw in keywords):
                    # 只要行里包含目标参数关键词，就算相关行
                    # 先尝试只提取相关子句（用逗号/分号拆分）
                    clauses = re.split(r'[，,；;]', line)
                    relevant_clauses = [c.strip() for c in clauses if any(kw in c for kw in keywords)]
                    if relevant_clauses:
                        key_lines.append('，'.join(relevant_clauses))
                    else:
                        key_lines.append(line)
                    break

    # 7. 组装结果
    if key_lines:
        return '\n'.join(key_lines[:5])

    if is_intro and product_lines:
        return '\n'.join(product_lines[:5])

    if target_products and product_lines:
        return '\n'.join(product_lines[:3])

    if not target_params and not target_products:
        return context[:500]

    return ""


def generate_simple_answer(question: str, context: str) -> str:
    """简单回答生成（无 LLM 时使用），先过滤无关内容"""
    filtered = filter_relevant_context(question, context)

    # 过滤函数明确返回空字符串 = 无关产品，直接拒绝
    if filtered == "":
        return "抱歉，知识库中没有与您问题相关的信息。请尝试询问产品相关的问题。"

    if filtered and filtered != context[:500]:
        # 清理文档标记
        for tag in ['[产品说明-无人机.txt]', '[产品说明-智能网关.txt]', '[产品说明-空气净化器.txt]', '---']:
            filtered = filtered.replace(tag, '')
        # 去重：相同参数值只保留最完整的一条
        seen_values = {}
        deduped_lines = []
        for line in filtered.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            # 提取冒号后的值作为去重key
            if '：' in line:
                value = line.split('：', 1)[1].strip()
                # 只保留值最完整的那条
                if value not in seen_values or len(line) > len(seen_values[value]):
                    if value in seen_values:
                        deduped_lines.remove(seen_values[value])
                    seen_values[value] = line
                    deduped_lines.append(line)
                continue
            deduped_lines.append(line)
        filtered = '\n'.join(deduped_lines)
        return f"根据知识库信息：{filtered.strip()}"

    # 过滤没匹配到，回退到关键词匹配
    question_words = question.lower().split()
    lines = context.split('\n')
    best_line = ""
    max_score = 0

    for line in lines:
        line_lower = line.lower()
        score = sum(1 for word in question_words if word in line_lower)
        if score > max_score:
            max_score = score
            best_line = line

    if best_line:
        for tag in ['[产品说明-无人机.txt]', '[产品说明-智能网关.txt]', '[产品说明-空气净化器.txt]', '---']:
            best_line = best_line.replace(tag, '')
        return f"根据知识库信息：{best_line.strip()}"

    return f"根据知识库信息：{context[:200]}..."


def generate_fallback_answer(question: str, context: str, llm=None) -> Dict[str, Any]:
    """
    低相关性时的混合回答策略
    """
    if llm is not None:
        prompt = f"""用户问题与检索到的知识相关性不高，但请尽量给出有用的回答。

【用户问题】
{question}

【检索到的部分相关信息】
{filter_relevant_context(question, context)}


【要求】
1. 承认相关性有限
2. 给出可能相关的回答
3. 建议用户如何获取更准确的信息

【回答】"""
        try:
            answer = llm.invoke(prompt)
            if isinstance(answer, str):
                answer = answer.strip()
            else:
                answer = str(answer)
        except:
            answer = generate_simple_answer(question, context)
    else:
        answer = generate_simple_answer(question, context)

    return {
        "answer": f"⚠️ 检索结果相关性较低，以下仅供参考：\n\n{answer}",
        "confidence": "中",
        "sources": [],
        "messages": [{"role": "assistant", "content": "低相关混合回答"}]
    }

def extract_sources(context: str) -> List[str]:
    """
    从上下文中提取来源信息
    """
    sources = []
    lines = context.split('\n')

    for line in lines:
        if '【' in line and '】' in line:
            sources.append(line.strip())

    return sources[:3]  # 最多返回 3 个来源

# ============================================================================
# 第五部分：边路由函数
# ============================================================================

def route_after_intent(state: QAState) -> str:
    """
    意图识别后的路由决策

    Returns:
        下一个节点名称
    """
    intent = state["intent"]

    # 闲聊直接生成回答
    if intent == "general_chat":
        return "answer_generator"

    # 其他意图都需要 RAG 检索
    return "rag_retriever"


def route_after_relevance(state: QAState) -> str:
    """
    相关性评估后的路由决策

    Returns:
        下一个节点名称
    """
    relevance = state["relevance_score"]

    # 高/中相关，直接生成回答
    if relevance >= 0.4:
        return "answer_generator"

    # 低相关，降级处理
    return "fallback_handler"

# ============================================================================
# 第六部分：工作流构建
# ============================================================================

def build_qa_workflow(llm=None) -> StateGraph:
    # 1. 创建状态图
    workflow = StateGraph(QAState)

    # 2. 添加节点
    workflow.add_node("intent_classifier", intent_classifier)
    workflow.add_node("rag_retriever", rag_retriever_node)
    workflow.add_node("relevance_evaluator", relevance_evaluator)
    workflow.add_node("answer_generator", lambda state: answer_generator(state, llm=llm))
    workflow.add_node("fallback_handler", lambda state: fallback_handler(state, llm=llm))

    # 3. 设置入口点
    workflow.set_entry_point("intent_classifier")

    # 4. 添加边
    # 意图识别 → 根据意图路由
    workflow.add_conditional_edges(
        "intent_classifier",
        route_after_intent,
        {
            "answer_generator": "answer_generator",  # 闲聊直接回答
            "rag_retriever": "rag_retriever"  # 其他意图先检索
        }
    )

    # RAG 检索 → 相关性评估
    workflow.add_edge("rag_retriever", "relevance_evaluator")

    # 相关性评估 → 根据相关性路由
    workflow.add_conditional_edges(
        "relevance_evaluator",
        route_after_relevance,
        {
            "answer_generator": "answer_generator",  # 高/中相关
            "fallback_handler": "fallback_handler"  # 低相关
        }
    )

    # 回答生成 / 降级处理 → 结束
    workflow.add_edge("answer_generator", END)
    workflow.add_edge("fallback_handler", END)

    # 5. 编译工作流
    return workflow.compile()

# ============================================================================
# 第七部分：便捷调用接口
# ============================================================================
def ask_question(question: str, llm=None, verbose: bool = True) -> Dict[str, Any]:
    """
    便捷的问答接口

    Args:
        question: 用户问题
        llm: 语言模型（可选）
        verbose: 是否打印详细信息

    Returns:
        包含 answer, confidence, sources 的字典
    """
    # 构建工作流
    app = build_qa_workflow(llm)

    # 创建初始状态
    initial_state = create_initial_state(question)

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"📝 问题: {question}")
        print(f"{'=' * 50}")

    # 执行工作流
    result = app.invoke(initial_state)

    if verbose:
        print(f"\n📊 执行结果:")
        print(f"   意图: {result['intent']} (置信度: {result['intent_confidence']:.2f})")
        print(f"   相关性: {result['relevance_score']:.2f}")
        print(f"   置信度: {result['confidence']}")
        print(f"\n💬 回答:")
        print(result['answer'])

        if result['sources']:
            print(f"\n📚 来源: {result['sources']}")

    return {
        "answer": result['answer'],
        "confidence": result['confidence'],
        "sources": result['sources'],
        "intent": result['intent']
    }

# ============================================================================
# 第八部分：测试代码
# ============================================================================

if __name__ == "__main__":
    print("🧪 LangGraph 问答工作流测试")
    print("=" * 50)

    # 测试问题列表
    test_questions = [
        "星途Pro的续航是多少？",
        "你好",
        "星途Pro无人机怎么样？",
        "网关有哪些参数？",
        "这个无人机能不能飞月亮？"
    ]

    # 执行测试
    for q in test_questions:
        ask_question(q, llm=None, verbose=True)
        print("\n" + "-" * 50)
