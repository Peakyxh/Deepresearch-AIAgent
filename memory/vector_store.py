"""
向量存储模块

使用 ChromaDB 实现向量存储，提供语义搜索能力。
支持会话级隔离：不同 session_id 的数据存在不同集合中，互不可见。
"""

import logging
import uuid
from typing import Any

import chromadb

logger = logging.getLogger(__name__)


class VectorStore:
    """
    向量存储类

    基于 ChromaDB 的向量存储，支持：
    - 文本向量化与存储（使用 ChromaDB 默认的 embedding 模型）
    - 语义相似度搜索
    - 持久化到磁盘
    - 会话级隔离：集合名拼接 session_id，不同会话互不可见

    集合命名规则：
    - 无 session_id: research / search_results / papers
    - 有 session_id: research_abc123 / search_results_abc123 / papers_abc123
    """

    def __init__(self, persist_dir: str = "./chroma_data", session_id: str = ""):
        """
        初始化向量存储

        Args:
            persist_dir: ChromaDB 持久化目录
            session_id: 会话 ID，用于集合隔离。为空则不隔离（全局共享）
        """
        self.persist_dir = persist_dir
        self.session_id = session_id
        self._client = None
        self._collections: dict[str, Any] = {}
        logger.info(f"VectorStore 初始化，持久化目录: {persist_dir}，会话ID: {session_id or '无(全局共享)'}")

    def _build_collection_name(self, base_name: str) -> str:
        """
        构建带会话 ID 的集合名

        如果有 session_id，集合名为 "research_abc123"；
        如果没有，集合名为 "research"（全局共享，兼容旧数据）。

        Args:
            base_name: 基础集合名（research / search_results / papers）

        Returns:
            带会话 ID 的集合名
        """
        if self.session_id:
            return f"{base_name}_{self.session_id}"
        return base_name

    async def initialize(self) -> None:
        """
        初始化 ChromaDB 客户端

        使用 PersistentClient 将数据持久化到磁盘，
        重启后数据不会丢失。
        """
        try:
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            logger.info(f"ChromaDB 客户端初始化成功，持久化目录: {self.persist_dir}")
        except Exception as e:
            logger.error(f"ChromaDB 初始化失败: {e}")
            # 回退到内存模式
            self._client = chromadb.Client()
            logger.warning("回退到 ChromaDB 内存模式（数据不会持久化）")

    def _get_collection(self, collection_name: str = "research") -> Any:
        """
        获取或创建 ChromaDB 集合

        集合名会自动拼接 session_id，实现会话级隔离。

        Args:
            collection_name: 基础集合名称（会自动拼接 session_id）

        Returns:
            ChromaDB 集合对象
        """
        if not self._client:
            raise RuntimeError("VectorStore 未初始化，请先调用 initialize()")

        # 拼接带 session_id 的集合名
        full_name = self._build_collection_name(collection_name)

        if full_name not in self._collections:
            self._collections[full_name] = self._client.get_or_create_collection(
                name=full_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"创建/获取集合: {full_name}")

        return self._collections[full_name]

    async def add_documents(
        self,
        documents: list[dict[str, Any]],
        collection_name: str = "research",
    ) -> list[str]:
        """
        添加文档到向量存储

        每个文档应包含：
        - content: 文本内容（必须）
        - metadata: 元数据字典（可选）
        - id: 文档 ID（可选，自动生成）

        Args:
            documents: 文档列表
            collection_name: 基础集合名称（会自动拼接 session_id）

        Returns:
            文档 ID 列表
        """
        collection = self._get_collection(collection_name)

        ids = []
        contents = []
        metadatas = []

        for doc in documents:
            # 生成唯一 ID
            doc_id = doc.get("id", str(uuid.uuid4()))
            ids.append(doc_id)
            contents.append(doc.get("content", ""))

            # 元数据中不能有 None 值
            metadata = doc.get("metadata", {})
            metadata = {k: v for k, v in metadata.items() if v is not None}
            metadatas.append(metadata)

        # 批量添加文档
        collection.add(
            ids=ids,
            documents=contents,
            metadatas=metadatas,
        )

        full_name = self._build_collection_name(collection_name)
        logger.info(f"添加 {len(documents)} 个文档到集合 '{full_name}'")
        return ids

    async def search(
        self,
        query: str,
        top_k: int = 5,
        collection_name: str = "research",
    ) -> list[dict[str, Any]]:
        """
        语义相似度搜索

        Args:
            query: 搜索查询文本
            top_k: 返回最相似的 k 个结果
            collection_name: 基础集合名称（会自动拼接 session_id）

        Returns:
            搜索结果列表，每个结果包含 content、metadata、distance
        """
        collection = self._get_collection(collection_name)
        full_name = self._build_collection_name(collection_name)

        # 检查集合是否为空
        if collection.count() == 0:
            logger.info(f"集合 '{full_name}' 为空，无搜索结果")
            return []

        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, collection.count()),
        )

        # 格式化结果
        formatted = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                item = {
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0.0,
                    "id": results["ids"][0][i] if results["ids"] else "",
                }
                formatted.append(item)

        logger.info(f"搜索 '{query[:30]}...' 返回 {len(formatted)} 条结果")
        return formatted

    async def delete_document(
        self,
        doc_id: str,
        collection_name: str = "research",
    ) -> None:
        """
        删除指定文档

        Args:
            doc_id: 文档 ID
            collection_name: 基础集合名称（会自动拼接 session_id）
        """
        collection = self._get_collection(collection_name)
        collection.delete(ids=[doc_id])
        logger.info(f"删除文档: {doc_id}")

    async def get_collection_count(self, collection_name: str = "research") -> int:
        """
        获取集合中的文档数量

        Args:
            collection_name: 基础集合名称（会自动拼接 session_id）

        Returns:
            文档数量
        """
        collection = self._get_collection(collection_name)
        return collection.count()
