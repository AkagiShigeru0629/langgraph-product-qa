# 定义工具（产品参数库查询）
def query_product_param(product_name, param_name):
    """职场标准工具函数：参数明确、有异常处理、返回结构化数据"""
    param_db = load_product_param_db()  # 你之前的函数
    if product_name not in param_db:
        return {"code": 0, "data": "", "msg": "产品不存在"}
    if param_name not in param_db[product_name]:
        return {"code": 0, "data": "", "msg": "参数不存在"}
    return {"code": 1, "data": param_db[product_name][param_name], "msg": "成功"}


# Agent工具调用逻辑
def agent_tool_calling(question):
    # 第一步：解析问题（提取产品名+参数名）
    product_name = match_product(question, load_product_param_db())  # 你之前的函数
    param_names = normalize_question(question)  # 你之前的函数

    # 第二步：调用工具
    results = []
    for param in param_names:
        tool_result = query_product_param(product_name, param)
        if tool_result["code"] == 1:
            results.append(f"{param}：{tool_result['data']}")

    # 第三步：生成回答
    if results:
        return f"{product_name} - {'; '.join(results)}"
    return "未找到相关参数"