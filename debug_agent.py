import sys
sys.path.append('..')

# 强制重新加载模块
import importlib
utils = importlib.reload(sys.modules.get('utils', __import__('utils')))

# 验证类与方法
print("1. 类是否存在:", 'ProductParamAgent' in dir(utils))
agent = utils.ProductParamAgent(mode="local")
print("2. 实例方法是否存在:", hasattr(agent, 'answer_question'))
print("3. session属性是否存在:", hasattr(agent, 'session'))
print("4. 类方法列表:", [m for m in dir(agent) if not m.startswith('__')])

# 加载数据
param_db = utils.load_product_param_db()
full_doc = utils.load_full_doc_content()

# 测试连续问答（关键验证）
questions = [
    "星途Pro的电池容量是多少？",
    "那星途的最大续航呢？",
]

print("\n=== 连续问答测试 ===")
for q in questions:
    print(f"❓ 问题: {q}")
    try:
        print(f"  当前agent对象类型: {q, param_db, full_doc}")
        answer = agent.answer_question(q, param_db, full_doc)
        print(f"🤖 回答: {answer[:100]}...")
    except AttributeError as e:
        print(f"💥 属性错误: {e}")
        print(f"  当前agent对象类型: {type(agent)}")
        print(f"  当前agent对象字典: {agent.__dict__}")
    print("-" * 50)
