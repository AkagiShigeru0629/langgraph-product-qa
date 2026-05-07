import os
import re

# ===================== 核心常量定义（全模块覆盖版） =====================
# 产品关键词（用于匹配用户提问中的产品）
PRODUCT_KEYS = [
    "星途", "无人机", "星途Pro", "X-Drone",
    "智能网关", "网关", "智联云盒", "SG-800",
    "清氧净护", "空气净化器", "家用净化器"
]

# 产品参数列表（严格匹配文档中的参数名）
PARAM_KEYS = [
    # 星途 Pro 专业级航拍无人机
    "展开尺寸", "折叠尺寸", "起飞重量", "最大续航",
    "图传距离", "云台", "相机传感器", "有效像素",
    "视频分辨率", "避障系统", "工作温度", "电池容量",

    # 智联云盒 Pro 企业级智能网关
    "处理器", "内存", "存储", "网口", "无线",
    "蜂窝网络", "供电", "外壳材质", "尺寸",

    # 清氧净护家用空气净化器
    "产品尺寸", "净重", "额定电压", "额定功率",
    "噪音范围", "滤网寿命", "净化方式", "连接方式", "显示屏"
]

# 参数/介绍类别名映射（适配口语化/泛指提问）
PARAM_ALIAS = {
    # 产品介绍类关键词
    "产品亮点": ["介绍", "概述", "优势", "特点", "卖点", "产品亮点"],
    # 无人机参数别名
    "起飞重量": ["重量", "机身重量"],
    "最大续航": ["续航时间", "能飞多久", "飞行时长"],
    "图传距离": ["传输距离", "信号距离"],
    "电池容量": ["电池大小"],
    # 网关参数别名
    "尺寸": ["网关尺寸", "机身尺寸", "大小"],
    "供电": ["供电电压", "电源规格"],
    "网口": ["网络接口", "接口数量"],
    # 空气净化器参数别名
    "产品尺寸": ["净化器尺寸", "机身尺寸"],
    "净重": ["重量", "机身重量"],
    "噪音范围": ["噪音值", "工作噪音", "运行噪音"],
    "滤网寿命": ["滤网使用时间", "滤网更换周期"],
    # 泛指类参数提问
    "空气净化器技术参数": ["技术参数", "参数", "所有参数", "详细参数"],
    "无人机技术参数": ["无人机参数", "无人机所有参数"],
    "网关技术参数": ["网关参数", "网关所有参数"]
}

# 介绍类提问关键词
INTRO_KEYWORDS = ["介绍", "概述", "简介", "产品介绍", "产品详情", "产品亮点",
                  "核心功能", "应用场景", "典型应用方案", "使用场景",
                  "包装清单", "标准套装清单", "售后保障", "售后服务", "质保政策"]


# ===================== 产品参数数据库（硬编码备用） =====================
def load_product_param_db():
    """
    加载产品参数数据库（备用方案，当文档加载失败时使用）
    """
    return {
        "星途 Pro 专业级航拍无人机": {
            "展开尺寸": "415mm×380mm×150mm",
            "折叠尺寸": "225mm×165mm×90mm",
            "起飞重量": "980g",
            "最大续航": "42分钟（标准负载）",
            "图传距离": "15km（FCC标准）",
            "云台": "三轴机械云台",
            "相机传感器": "1英寸CMOS，有效像素2000万",
            "视频分辨率": "8K/24fps、4K/120fps",
            "避障系统": "六向视觉+红外",
            "工作温度": "-10℃~40℃",
            "电池容量": "5000mAh"
        },
        "智联云盒 Pro 企业级智能网关": {
            "处理器": "ARM Cortex-A72 四核 1.8GHz",
            "内存": "4GB DDR4",
            "存储": "32GB eMMC",
            "网口": "4×千兆电口 + 2×千兆光口",
            "无线": "Wi-Fi 6 + Bluetooth 5.2",
            "蜂窝网络": "5G NR / LTE Cat.6",
            "供电": "DC 12V/2A",
            "外壳材质": "工业级铝合金",
            "尺寸": "180mm×120mm×44mm"
        },
        "清氧净护家用空气净化器": {
            "产品尺寸": "350mm×350mm×650mm",
            "净重": "8.5kg",
            "额定电压": "220V/50Hz",
            "额定功率": "60W",
            "噪音范围": "28~55dB",
            "滤网寿命": "6~8个月",
            "净化方式": "HEPA H13滤网 + 活性炭 + 负离子",
            "连接方式": "Wi-Fi + 蓝牙",
            "显示屏": "OLED触控屏"
        }
    }


# ===================== 产品匹配函数 =====================
def match_product(question, param_db):
    """
    匹配用户提问中的产品名称
    :param question: 用户问题
    :param param_db: 产品参数库
    :return: 匹配到的产品名称
    """
    question_lower = question.lower()
    for product_name in param_db.keys():
        # 精确匹配产品全名
        if product_name.lower() in question_lower:
            print(f"if 1 in ")
            return product_name

    # 模糊匹配关键词
    for key in PRODUCT_KEYS:
        print(key.lower())
        print(f"question_lower={question_lower} ")

        if key.lower() in question_lower:
            print(f"if 2 in ")
            # 根据关键词匹配对应的产品全名
            if key in ["星途", "无人机", "星途Pro", "X-Drone"]:
                return "星途 Pro 专业级航拍无人机"
            elif key in ["智能网关", "网关", "智联云盒", "SG-800"]:
                return "智联云盒 Pro 企业级智能网关"
            elif key in ["清氧净护", "空气净化器", "家用净化器"]:
                return "清氧净护家用空气净化器"

    return None


# ===================== 参数归一化函数 =====================
def normalize_question(question, mode="local"):
    """
    归一化用户提问，提取匹配的参数名
    :param question: 用户输入的问题
    :param mode: 运行模式
    :return: 匹配到的参数列表
    """
    question_lower = question.lower()
    matched_params = []
    param_db = load_product_param_db()
    product_name = match_product(question, param_db)

    # 1. 匹配单个参数
    for param in PARAM_KEYS:
        if param.lower() in question_lower:
            matched_params.append(param)
        for alias in PARAM_ALIAS.get(param, []):
            if alias.lower() in question_lower:
                matched_params.append(param)

    # 2. 匹配泛指类参数提问
    if product_name:
        if "清氧净护" in product_name or "空气净化器" in product_name:
            if any(kw in question_lower for kw in ["技术参数", "参数", "所有参数", "详细参数"]):
                matched_params = [
                    "产品尺寸", "净重", "额定电压", "额定功率",
                    "噪音范围", "滤网寿命", "净化方式", "连接方式", "显示屏"
                ]
        elif "星途" in product_name or "无人机" in product_name:
            if any(kw in question_lower for kw in ["技术参数", "参数", "所有参数", "详细参数"]):
                matched_params = [
                    "展开尺寸", "折叠尺寸", "起飞重量", "最大续航",
                    "图传距离", "云台", "相机传感器", "有效像素",
                    "视频分辨率", "避障系统", "工作温度", "电池容量"
                ]
        elif "智联云盒" in product_name or "网关" in product_name:
            if any(kw in question_lower for kw in ["技术参数", "参数", "所有参数", "详细参数"]):
                matched_params = [
                    "处理器", "内存", "存储", "网口", "无线",
                    "蜂窝网络", "供电", "外壳材质", "尺寸"
                ]

    # 去重
    matched_params = list(set(matched_params))
    return matched_params


# ===================== 提取头部信息 =====================
def extract_header_info(doc_content, product_name, target_field):
    """
    提取文档头部的特定信息（如产品型号、目标用户等）
    :param doc_content: 文档内容
    :param product_name: 产品名称
    :param target_field: 目标字段
    :return: 提取的内容
    """
    lines = doc_content.split('\n')
    target_line = f"-  **{target_field}** "

    for i, line in enumerate(lines):
        if target_line in line:
            # 找到目标行，提取冒号后的内容
            if ':' in line:
                return line.split(':', 1)[1].strip()
            elif '：' in line:
                return line.split('：', 1)[1].strip()

    return f"暂未找到{target_field}信息"


# ===================== 模块内容提取函数 =====================
def extract_module_content(doc_content, module_name):
    """
    从文档中提取指定模块的内容
    支持多种模块标题格式："模块名"、"一、模块名"、"1、模块名"
    :param doc_content: 文档内容
    :param module_name: 模块名称
    :return: 提取的模块内容
    """
    lines = doc_content.split('\n')
    result_lines = []
    in_target_module = False

    # 支持多种模块标题格式
    module_patterns = [
        module_name,
        f'^[一二三四五六七八九十]+、{re.escape(module_name)}',
        f'^\\d+、{re.escape(module_name)}'
    ]

    for line in lines:
        # 检查是否进入目标模块
        if not in_target_module:
            for pattern in module_patterns:
                if re.search(pattern, line):
                    in_target_module = True
                    result_lines.append(line)
                    break
        else:
            # 检查是否退出当前模块（遇到新的一级标题）
            if re.match(r'^[一二三四五六七八九十]+、', line) and module_name not in line:
                break
            elif re.match(r'^\d+、', line) and module_name not in line:
                break
            result_lines.append(line)

    module_content = '\n'.join(result_lines).strip()
    if not module_content:
        return f"未找到【{module_name}】模块内容"

    return module_content


# ===================== 提取产品介绍（按产品类型区分 - 关键修复） =====================
def extract_product_intro(product_name, full_doc, question=""):
    """
    提取产品信息（支持所有模块+头部信息）
    关键修复：按产品类型区分模块映射，避免关键词冲突
    :param product_name: 产品名称
    :param full_doc: 完整文档
    :param question: 用户问题
    :return: 产品信息
    """
    doc_content = full_doc.get(product_name, f"暂无{product_name}的信息")

    if not question:
        return doc_content

    question_lower = question.lower()

    # 优先检查头部信息关键词
    if any(kw in question_lower for kw in ["产品型号", "型号"]):
        return extract_header_info(doc_content, product_name, "产品型号")

    if any(kw in question_lower for kw in ["目标用户", "用户"]):
        return extract_header_info(doc_content, product_name, "目标用户")

    if any(kw in question_lower for kw in ["适用人群", "人群"]):
        return extract_header_info(doc_content, product_name, "适用人群")

    if any(kw in question_lower for kw in ["适用场景"]):
        return extract_header_info(doc_content, product_name, "适用场景")

    if any(kw in question_lower for kw in ["发布时间", "发布"]):
        return extract_header_info(doc_content, product_name, "发布时间")

    # 关键修复：按产品类型区分模块映射（避免关键词冲突）
    if "星途" in product_name or "无人机" in product_name:
        # 无人机模块映射
        module_keywords = {
            "产品概述": ["概述", "介绍", "简介", "产品介绍", "产品详情"],
            "核心功能": ["功能", "核心功能", "能力", "主要功能", "特性"],
            "技术参数": ["参数", "规格", "技术规格", "技术参数", "详细参数"],
            "产品优势": ["产品优势", "优势", "产品亮点", "亮点", "产品优点", "优点", "特点", "卖点", "特色"],
            "应用场景": ["场景", "用途", "应用", "使用场景", "适用场景", "应用方案", "典型应用方案"],
            "标准套装清单": ["套装", "清单", "配件", "包装清单", "标准套装"],
            "售后保障": ["售后", "保修", "保障", "质保", "售后服务", "质保政策"]
        }

    elif "智联云盒" in product_name or "网关" in product_name:
        # 网关模块映射
        module_keywords = {
            "产品概述": ["概述", "介绍", "简介", "产品介绍", "产品详情"],
            "核心功能": ["功能", "核心功能", "能力", "主要功能", "特性"],
            "技术参数": ["参数", "规格", "技术规格", "技术参数", "详细参数"],
            "产品优势": ["产品优势", "优势", "产品亮点", "亮点", "产品优点", "优点", "特点", "卖点", "特色"],
            "典型应用方案": ["场景", "用途", "应用", "使用场景", "适用场景", "应用方案", "典型应用方案"],
            "包装清单": ["套装", "清单", "配件", "包装清单", "标准套装"],
            "售后服务": ["售后", "保修", "保障", "质保", "售后服务", "质保政策"]
        }

    elif "清氧净护" in product_name or "空气净化器" in product_name:
        # 空气净化器模块映射
        module_keywords = {
            "产品概述": ["概述", "介绍", "简介", "产品介绍", "产品详情"],
            "核心功能": ["功能", "核心功能", "能力", "主要功能", "特性"],
            "技术参数": ["参数", "规格", "技术规格", "技术参数", "详细参数"],
            "产品亮点": ["产品亮点", "亮点", "产品优势", "优势", "产品优点", "优点", "特点", "卖点", "特色"],
            "使用场景": ["场景", "用途", "应用", "使用场景", "适用场景", "应用方案", "典型应用方案"],
            "包装清单": ["套装", "清单", "配件", "包装清单", "标准套装"],
            "质保政策": ["售后", "保修", "保障", "质保", "售后服务", "质保政策"]
        }
    else:
        # 默认映射（不应到达这里）
        module_keywords = {}

    # 识别目标模块
    target_module = None
    for module, keywords in module_keywords.items():
        if any(kw in question_lower for kw in keywords):
            target_module = module
            break

    if not target_module:
        return doc_content

    module_content = extract_module_content(doc_content, target_module)
    return module_content


# ===================== 混合回答函数（全模块覆盖版） =====================
def hybrid_answer(question, param_db, full_doc, mode="local"):
    """
    生成产品问答的最终回答
    :param question: 用户问题
    :param param_db: 产品参数库
    :param full_doc: 产品完整文档
    :param mode: 运行模式
    :return: 最终回答文本
    """
    # 1. 匹配产品
    product_name = match_product(question, param_db)
    if not product_name:
        return "【硬拦截】抱歉，知识库中未找到相关产品信息"

    # 2. 判断提问类型
    question_lower = question.lower()
    is_intro_question = any(kw in question_lower for kw in INTRO_KEYWORDS)

    # 3. 处理产品介绍类提问
    if is_intro_question:
        intro = extract_product_intro(product_name, full_doc, question)
        return f"{product_name}：\n{intro.strip()}"

    # 4. 处理参数查询类提问
    target_params = normalize_question(question, mode)
    if not target_params:
        return f"抱歉，未识别到{product_name}的相关参数，可尝试提问：\n- 空气净化器：产品尺寸、净重、噪音范围等\n- 无人机：电池容量、最大续航、图传距离等\n- 网关：网口、供电、无线等"

    product_params = param_db.get(product_name, {})
    answer_parts = [f"{product_name} 技术参数："]
    for param in target_params:
        value = product_params.get(param, "暂无数据")
        answer_parts.append(f"- {param}：{value}")

    return "\n".join(answer_parts)


# ===================== 加载完整文档 =====================
def load_full_doc_content():
    """
    专业级自动读取：扫描指定文件夹下的所有txt文档
    优先从knowledge文件夹读取，如果不存在则从根目录读取
    """
    # 优先尝试knowledge文件夹
    DOC_DIR = "knowledge"
    full_doc = {}
    product_mapping = {
        "星途 Pro 专业级航拍无人机": ["无人机", "星途", "drone"],
        "智联云盒 Pro 企业级智能网关": ["网关", "智联云盒", "sg-800"],
        "清氧净护家用空气净化器": ["空气净化器", "清氧净护", "净化器"]
    }

    # 确定文档目录（knowledge或根目录）
    if os.path.exists(DOC_DIR) and os.listdir(DOC_DIR):
        scan_dir = DOC_DIR
    else:
        scan_dir = "."

    txt_files = [f for f in os.listdir(scan_dir) if f.endswith(".txt")]
    if not txt_files:
        return {
            "星途 Pro 专业级航拍无人机": f"⚠️ {scan_dir} 目录下无txt文档，请放入产品说明文档",
            "智联云盒 Pro 企业级智能网关": f"⚠️ {scan_dir} 目录下无txt文档，请放入产品说明文档",
            "清氧净护家用空气净化器": f"⚠️ {scan_dir} 目录下无txt文档，请放入产品说明文档"
        }

    for file_name in txt_files:
        file_path = os.path.join(scan_dir, file_name)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="gbk") as f:
                content = f.read().strip()

        file_name_lower = file_name.lower()
        for product_name, keywords in product_mapping.items():
            if any(kw.lower() in file_name_lower for kw in keywords):
                full_doc[product_name] = content
                break

    for product_name in product_mapping.keys():
        if product_name not in full_doc:
            full_doc[
                product_name] = f"⚠️ 未找到{product_name}对应的文档（请确保文件名含：{','.join(product_mapping[product_name])}）"

    return full_doc


# ===================== 辅助函数 =====================
def list_knowledge_files():
    """
    列出知识库中的文件
    :return: 文件列表
    """
    return ["产品说明-空气净化器.txt", "产品说明-无人机.txt", "产品说明-智能网关.txt"]


def llm_router(prompt, mode="local"):
    """
    LLM请求路由 - 升级为真实API调用

    参数:
        prompt: 提示词
        mode: 运行模式 ("local" 或 "cloud")

    返回:
        LLM返回结果
    """
    if mode == "cloud":
        # 云端模式（通义千问）实现
        # 待实现：调用DASHSCOPE API
        return call_dashscope_api(prompt)
    else:
        # 本地模式（Ollama）实现
        return call_ollama_local(prompt)

def call_ollama_local(prompt, model="qwen:7b", temperature=0.7, max_tokens=500):
    """
    调用本地Ollama模型

    参数:
        prompt: 用户提示词
        model: 模型名称，默认qwen:7b
        temperature: 随机性控制
        max_tokens: 最大生成token数
    """
    import requests

    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你是一位产品参数专家，准确提取产品参数并用简洁语言回答用户问题。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()

        result = response.json()
        return result["message"]["content"]

    except requests.exceptions.ConnectionError:
        return "无法连接到Ollama服务，请确保已运行 `ollama serve` 并下载了相应模型"
    except Exception as e:
        return f"API调用出错: {str(e)}"

# ===================== 智能Agent类（集成到现有utils.py末尾） =====================
import requests

class OllamaChatSession:
    """多轮对话会话管理类（从简报中提取）"""

    def __init__(self, model="qwen:7b", system_prompt=None):
        self.model = model
        self.messages = []

        # 添加系统提示（如果有）
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def chat(self, user_input):
        """发送用户消息并获取AI回复"""
        # 添加用户消息
        self.messages.append({"role": "user", "content": user_input})

        url = "http://localhost:11434/api/chat"
        payload = {
            "model": self.model,
            "messages": self.messages,
            "stream": False
        }

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()

            result = response.json()
            ai_reply = result["message"]["content"]

            # 添加AI回复到消息历史
            self.messages.append({"role": "assistant", "content": ai_reply})

            return ai_reply

        except Exception as e:
            return f"对话出错: {str(e)}"

    def get_history(self):
        """获取对话历史"""
        return self.messages

    def clear_history(self):
        """清空对话历史（保留系统提示）"""
        if self.messages and self.messages[0]["role"] == "system":
            system_prompt = self.messages[0]
            self.messages = [system_prompt]
        else:
            self.messages = []


class ProductParamAgent:
    """产品参数智能Agent（简报中步骤2的核心）"""

    def __init__(self, mode="local"):
        self.mode = mode
        self.session = OllamaChatSession(
            model="qwen:7b",
            system_prompt="""你是专业的产品参数问答AI助手，遵循以下规则：
1. 准确提取用户问题中的产品名称和参数需求
2. 优先从参数库中查找精确匹配
3. 如果参数库中没有，基于产品文档进行智能推断
4. 回答保持简洁、专业、实用
"""
        )
    def answer_question(self, question, param_db, full_doc):
        """回答产品参数问题（优化版：同时传递参数名+数值）"""
        # 1. 提取产品名称
        product_name = match_product(question, param_db)
        print(f"  当前product_name: {product_name}")
        if not product_name:
            return "未识别到产品名称，请确保问题中包含具体产品信息（如：星途Pro、空气净化器、网关）"

        # 2. 获取该产品的所有参数（名+值）
        product_params = param_db.get(product_name, {})

        # 3. 构建带完整参数的提示词
        enhanced_prompt = f"""
           用户问题：{question}
    
           产品名称：{product_name}
    
           **参数库（精确数据）** ：
           {chr(10).join([f'- {param}：{value}' for param, value in product_params.items()])}
    
           **文档摘要（补充信息）** ：
           {full_doc.get(product_name, '')[:500]}...
    
           请按以下规则回答：
           1. 如果参数库中有用户问的参数，直接返回该参数的完整信息（包括单位）
           2. 如果参数库中没有，则基于文档摘要进行合理推断
           3. 回答保持简洁、专业，不要出现“未明确提及”等模糊表述
           """

        # 4. 调用LLM
        answer = self.session.chat(enhanced_prompt)
        return answer




