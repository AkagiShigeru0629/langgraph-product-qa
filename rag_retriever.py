# -*- coding: utf-8 -*-
"""
RAG 向量检索模块
功能：文档导入向量库、用户查询检索、检索结果整合
适用：多产品参数问答系统 v2.0

使用方法：
    from rag_retriever import RAGRetriever

    # 初始化
    rag = RAGRetriever()

    # 构建索引（首次使用）
    rag.build_index()

    # 检索
    results = rag.retrieve("星途Pro的续航如何？")
"""

import os
import re

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any


# ========== 配置区 ==========
class RAGConfig:
    """RAG配置类"""

    # Embedding模型配置
    # EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 轻量英文模型
    # 中文推荐模型（需要时可切换）：
    EMBEDDING_MODEL = "moka-ai/m3e-base"
    # EMBEDDING_MODEL = "shibing624/text2vec-base-chinese"

    # 文档分割配置
    CHUNK_SIZE = 300  # 文档块大小（字符数）
    CHUNK_OVERLAP = 50  # 重叠字符数

    # 检索配置
    TOP_K = 5  # 召回文档块数量
    SIMILARITY_THRESHOLD = 0.5  # 相似度阈值（低于此值不返回）

    # 路径配置
    KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
    VECTOR_STORE_DIR = Path(__file__).parent / "vector_store"


# ========== 文档处理工具 ==========
class Document:
    """文档块数据结构"""

    def __init__(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        self.page_content = content
        self.metadata = metadata or {}
        # 生成唯一ID（结合文件名+行号+内容，避免重复）
        source = metadata.get("source", "unknown") if metadata else "unknown"
        line_num = metadata.get("line_num", 0) if metadata else 0
        unique_str = f"{source}_{line_num}_{content[:50]}"
        self.id = hashlib.md5(unique_str.encode()).hexdigest()[:12]

    def __repr__(self):
        return f"Document(id={self.id}, content={self.page_content[:50]}...)"


def load_documents(knowledge_dir: Path) -> List[Document]:
    """
    加载知识库目录下的所有文档

    Args:
        knowledge_dir: 知识库目录路径

    Returns:
        Document列表
    """
    documents = []

    if not knowledge_dir.exists():
        print(f"⚠️ 知识库目录不存在：{knowledge_dir}")
        return documents

    # 支持的文档格式
    extensions = ['.txt', '.md', '.csv']

    for file_path in knowledge_dir.rglob('*'):
        if file_path.suffix.lower() in extensions:
            try:
                # 读取文件内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 跳过空文件
                if not content.strip():
                    continue

                # 按行分割（保留段落结构）
                lines = content.split('\n')

                # 创建文档块
                for i, line in enumerate(lines):
                    if line.strip():  # 跳过空行
                        doc = Document(
                            content=line.strip(),
                            metadata={
                                "source": file_path.name,
                                "line_num": i
                            }
                        )
                        documents.append(doc)

            except Exception as e:
                print(f"❌ 读取文件失败 {file_path}: {e}")

    print(f"✅ 加载了 {len(documents)} 个文档块")
    return documents


def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    文本分割（按字符数分割，保留重叠）

    Args:
        text: 输入文本
        chunk_size: 每块大小
        overlap: 重叠大小

    Returns:
        文本块列表
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # 重叠移动

    return chunks


# ========== Embedding工具 ==========
class EmbeddingWrapper:
    """
    Embedding包装器
    支持本地模型和API模式
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载Embedding模型"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            print(f"✅ Embedding模型加载成功：{self.model_name}")
        except ImportError:
            print("⚠️ sentence-transformers 未安装，将使用简化的Embedding")
            self.model = None

    def embed_query(self, text: str) -> List[float]:
        """
        将文本转为向量

        Args:
            text: 输入文本

        Returns:
            向量列表
        """
        if self.model is None:
            # 简化模式：返回随机向量（仅供测试）
            import random
            return [random.random() for _ in range(384)]

        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量将文档转为向量

        Args:
            texts: 文本列表

        Returns:
            向量列表
        """
        if self.model is None:
            import random
            return [[random.random() for _ in range(384)] for _ in texts]

        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()


# ========== 向量存储工具 ==========
class VectorStore:
    """
    简化版向量存储（ChromaDB封装）
    支持基本的增删查操作
    """

    def __init__(self, persist_dir: Path):
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection = None
        self._init_collection()

    def _init_collection(self):
        """初始化ChromaDB collection"""
        try:
            import chromadb
            from chromadb.config import Settings

            client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=Settings(anonymized_telemetry=False)
            )
            self.collection = client.get_or_create_collection(
                name="product_knowledge",
                metadata={"description": "产品知识库向量库"}
            )
            print("✅ ChromaDB初始化成功")
        except ImportError:
            print("⚠️ ChromaDB未安装，将使用内存模式")
            self.collection = None
        except Exception as e:
            print(f"⚠️ ChromaDB初始化失败：{e}，使用内存模式")
            self.collection = None

    def add_documents(self, documents: List[Document], embeddings: List[List[float]]):
        """
        添加文档到向量库

        Args:
            documents: Document列表
            embeddings: 对应的向量列表
        """
        if self.collection is None:
            print("⚠️ 向量库未初始化，跳过添加")
            return

        ids = [doc.id for doc in documents]
        contents = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas
        )
        print(f"✅ 添加了 {len(documents)} 个文档块到向量库")

    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Dict]:
        """
        向量相似度搜索

        Args:
            query_embedding: 查询向量
            top_k: 返回数量

        Returns:
            搜索结果列表
        """
        if self.collection is None:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        # 格式化结果
        formatted = []
        if results['documents'] and len(results['documents']) > 0:
            for i, doc in enumerate(results['documents'][0]):
                formatted.append({
                    "content": doc,
                    "distance": results['distances'][0][i] if 'distances' in results else 0,
                    "metadata": results['metadatas'][0][i] if 'metadatas' in results else {}
                })

        return formatted

    def count(self) -> int:
        """返回向量库中的文档数量"""
        if self.collection is None:
            return 0
        return self.collection.count()


# ========== RAG检索器核心类 ==========
class RAGRetriever:
    """
    RAG检索器主类

    功能：
    1. 构建向量索引（build_index）
    2. 检索相关文档（retrieve）
    3. 混合问答整合（集成到现有系统）

    使用示例：
        rag = RAGRetriever()
        rag.build_index()  # 首次使用
        results = rag.retrieve("星途Pro续航如何？")
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        """
        初始化RAG检索器

        Args:
            config: RAG配置对象
        """
        self.config = config or RAGConfig()

        # 初始化组件
        self.embedding = EmbeddingWrapper(self.config.EMBEDDING_MODEL)
        self.vector_store = VectorStore(self.config.VECTOR_STORE_DIR)

        # 缓存文档列表
        self.documents = []

        print(f"📦 RAGRetriever初始化完成")
        print(f"   - Embedding模型: {self.config.EMBEDDING_MODEL}")
        print(f"   - 知识库目录: {self.config.KNOWLEDGE_DIR}")
        print(f"   - 向量库目录: {self.config.VECTOR_STORE_DIR}")

    def build_index(self, force_rebuild: bool = False) -> int:
        """
        构建向量索引

        Args:
            force_rebuild: 是否强制重建（跳过已有向量库）

        Returns:
            索引的文档块数量
        """
        # 检查是否已有向量库
        if not force_rebuild and self.vector_store.count() > 0:
            count = self.vector_store.count()
            print(f"📦 向量库已存在，包含 {count} 个文档块，跳过重建")
            return count

        print("🔨 开始构建向量索引...")

        # 1. 加载文档
        self.documents = load_documents(self.config.KNOWLEDGE_DIR)

        if not self.documents:
            print("⚠️ 未找到文档，请检查知识库目录")
            return 0

        # 2. 生成向量
        print("⏳ 正在生成文档向量...")
        texts = [doc.page_content for doc in self.documents]
        embeddings = self.embedding.embed_documents(texts)

        # 3. 存储向量
        self.vector_store.add_documents(self.documents, embeddings)

        print(f"✅ 向量索引构建完成，共 {len(self.documents)} 个文档块")
        return len(self.documents)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> str:
        """
        检索与查询相关的文档内容

        Args:
            query: 用户问题
            top_k: 返回的文档块数量

        Returns:
            拼接的上下文字符串（可直接用于LLM Prompt）
        """
        top_k = top_k or self.config.TOP_K

        # 1. 将查询转为向量
        query_embedding = self.embedding.embed_query(query)

        # 2. 向量相似度搜索
        results = self.vector_store.search(query_embedding, top_k=top_k)

        if not results:
            return ""

        # 3. 过滤低相似度结果
        # ChromaDB返回的是L2距离（值越小越相似），不是0-1的相似度分数
        # L2距离>500基本不相关，设一个合理阈值
        DISTANCE_THRESHOLD = 500
        filtered = [
            r for r in results
            if r['distance'] < DISTANCE_THRESHOLD
        ]

        if not filtered:
            return ""

        # 4. 关键词重排序：查询中的关键词在文档中出现时，提升其优先级
        query_keywords = [w for w in re.split(r'[，。？、的了呢吗是多少\s]', query) if len(w.strip()) > 1]

        def keyword_score(result):
            content = result['content']
            hit_count = sum(1 for kw in query_keywords if kw in content)
            # 距离越小越好（向量相似度），关键词命中越多越好
            return result['distance'] - hit_count * 50

        filtered.sort(key=keyword_score)

        # 5. 拼接上下文
        context_parts = []
        for r in filtered:
            source = r['metadata'].get('source', '未知来源')
            content = r['content']
            context_parts.append(f"[{source}]\n{content}")

        context = "\n\n---\n\n".join(context_parts)
        return context

    def retrieve_with_score(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        检索并返回带分数的结果

        Args:
            query: 用户问题
            top_k: 返回数量

        Returns:
            包含content、score、metadata的列表
        """
        top_k = top_k or self.config.TOP_K

        query_embedding = self.embedding.embed_query(query)
        results = self.vector_store.search(query_embedding, top_k=top_k)

        # 计算相似度分数（distance越小，相似度越高）
        formatted_results = []
        for r in results:
            similarity = 1 - r['distance']  # 转换为相似度
            formatted_results.append({
                "content": r['content'],
                "score": similarity,
                "metadata": r['metadata']
            })

        return formatted_results

    def add_document(self, content: str, metadata: Optional[Dict] = None):
        """
        动态添加单个文档（用于增量更新）

        Args:
            content: 文档内容
            metadata: 元数据
        """
        doc = Document(content, metadata)
        embedding = self.embedding.embed_query(content)
        self.vector_store.add_documents([doc], [embedding])
        print(f"✅ 添加文档成功：{content[:50]}...")

    def get_stats(self) -> Dict[str, Any]:
        """获取向量库统计信息"""
        return {
            "vector_count": self.vector_store.count(),
            "embedding_model": self.config.EMBEDDING_MODEL,
            "chunk_size": self.config.CHUNK_SIZE,
            "knowledge_dir": str(self.config.KNOWLEDGE_DIR)
        }


# ========== 集成辅助函数 ==========
def create_hybrid_answer_func(rag_retriever: RAGRetriever):
    """
    创建混合问答函数（用于替换现有hybrid_answer）

    Args:
        rag_retriever: RAGRetriever实例

    Returns:
        混合问答函数
    """
    from utils import match_product, normalize_question, llm_router, load_product_param_db

    def hybrid_answer(question: str, param_db: dict, full_doc: str, mode: str = "local") -> str:
        """
        混合问答：规则匹配 + RAG检索 + LLM生成

        优先级：
        1. 规则匹配（精准参数查询）
        2. RAG检索（语义问答）
        3. LLM生成（兜底）
        """
        # 1. 尝试规则匹配（最高优先级，精准无幻觉）
        product_name = match_product(question, param_db)
        param_names = normalize_question(question, mode=mode)

        if product_name and param_names:
            results = []
            for param in param_names:
                if param in param_db.get(product_name, {}):
                    results.append(f"{param}：{param_db[product_name][param]}")

            if results:
                return f"📦 {product_name}\n" + "\n".join(results)
