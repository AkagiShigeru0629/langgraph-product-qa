"""
LangGraph 最小示例：用户问答机器人
来源：AI大模型职场就业学习计划 - Week 5
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# ============ 第一步：定义 State（就是一本共享笔记本）============

class QARState(TypedDict):
    """问答状态：所有节点共享的信息"""
    question: str      # 用户的问题
    context: str       # 检索到的上下文
    answer: str        # 最终答案


# ============ 第二步：定义 Node（每个节点就是一个函数）============

def retrieve_info(state: QARState) -> QARState:
    """
    检索节点：根据问题搜索相关文档
    """
    question = state["question"]
    
    # 模拟搜索过程（替换为实际的 RAG 检索）
    mock_context = f"关于「{question}」的相关资料已检索到"
    
    return {"context": mock_context}


def generate_answer(state: QARState) -> QARState:
    """
    生成节点：根据上下文生成答案
    """
    answer = f"根据检索结果，回答您的问题：{state['question']}"
    return {"answer": answer}


# ============ 第三步：构建工作流（用边连接节点）============

def build_qa_graph():
    """
    构建问答工作流
    流程：START → retrieve → generate → END
    """
    # 创建图
    graph = StateGraph(QARState)
    
    # 添加节点
    graph.add_node("retrieve", retrieve_info)
    graph.add_node("generate", generate_answer)
    
    # 添加边（固定流程）
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    
    # 编译并返回
    return graph.compile()


# ============ 第四步：运行示例 ============

if __name__ == "__main__":
    # 构建工作流
    app = build_qa_graph()
    
    # 运行测试
    result = app.invoke({
        "question": "这个产品支持哪些功能？",
        "context": "",
        "answer": ""
    })
    
    # 输出结果
    print("=" * 40)
    print("问题:", result["question"])
    print("上下文:", result["context"])
    print("回答:", result["answer"])
    print("=" * 40)
