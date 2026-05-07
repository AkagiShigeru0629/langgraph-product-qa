# 职场标准写法：用列表存储对话历史 + 实体追踪
import re


class AgentMemory:
    def __init__(self):
        self.history = []
        self.current_entity = None
        self.entity_history = []

    PRODUCT_NAMES = [
        "星途Pro", "星途Pro无人机", "无人机",
        "智联云盒", "智联云盒网关", "网关", "智能网关",
        "清氧净护", "清氧净护净化器", "净化器", "空气净化器"
    ]

    PRONOUNS = ["它", "这个", "那个", "该产品", "此产品"]

    def add_memory(self, role, content):
        self.history.append({"role": role, "content": content})
        self._extract_entity(content)
        if len(self.history) > 20:
            self.history = self.history[-20:]

    def _extract_entity(self, content):
        for product in self.PRODUCT_NAMES:
            if product in content:
                self.current_entity = self._normalize_entity(product)
                self.entity_history.append(self.current_entity)
                break

    def _normalize_entity(self, name):
        if "星途" in name or "无人机" in name:
            return "星途Pro无人机"
        elif "云盒" in name or "网关" in name:
            return "智联云盒网关"
        elif "净化器" in name or "清氧" in name:
            return "清氧净护净化器"
        return name

    def resolve_pronouns(self, question):
        if not self.current_entity:
            return question
        for pronoun in self.PRONOUNS:
            if pronoun in question:
                return question.replace(pronoun, self.current_entity)
        self._extract_entity(question)
        return question

    def get_memory(self):
        return self.history

    def get_current_entity(self):
        return self.current_entity

    def get_formatted_history(self, last_n=5):
        recent = self.history[-last_n * 2:] if len(self.history) > last_n * 2 else self.history
        formatted = ""
        for msg in recent:
            role = "用户" if msg["role"] == "user" else "助手"
            formatted += f"{role}: {msg['content']}\n"
        return formatted


# 测试代码
if __name__ == "__main__":
    memory = AgentMemory()

    memory.add_memory("user", "星途Pro的电池容量是多少？")
    print(f"[当前实体] {memory.get_current_entity()}")

    memory.add_memory("assistant", "星途Pro的电池容量为5000mAh")

    user_question2 = "它的续航时间呢？"
    resolved = memory.resolve_pronouns(user_question2)
    print(f"[原始问题] {user_question2}")
    print(f"[解析后] {resolved}")
