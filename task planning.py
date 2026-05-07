def agent_task_planning(task):
    # 第一步：拆解任务（用大模型生成任务列表）
    prompt = f"""
    你是任务规划Agent，把复杂任务拆解成可执行的子任务，输出格式为JSON数组。
    复杂任务：{task}
    示例：
    输入：生成星途Pro产品分析报告
    输出：["提取星途Pro核心参数","对比同类产品参数","生成分析报告"]
    """
    # 调用大模型获取任务列表
    task_list = llm_router(prompt, mode="cloud")  # 你之前的模型路由函数
    task_list = eval(task_list)  # 转成列表（职场注意：需校验格式）

    # 第二步：执行子任务
    results = []
    for sub_task in task_list:
        if "提取参数" in sub_task:
            results.append(agent_tool_calling("星途Pro的核心参数"))
        elif "对比产品" in sub_task:
            results.append(agent_tool_calling("星途Pro vs 大疆Mini4参数对比"))
        elif "生成报告" in sub_task:
            report_prompt = f"基于以下信息生成产品分析报告：{results}"
            results.append(llm_router(report_prompt, mode="cloud"))

    # 第三步：整合结果
    return "\n\n".join(results)


# 用法：复杂任务一键完成
task = "生成星途Pro产品分析报告，包含核心参数和竞品对比"
report = agent_task_planning(task)
print(report)