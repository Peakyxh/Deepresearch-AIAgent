"""
记忆管理器

管理研究过程中的记忆存储和检索，
包括研究记录的保存、搜索结果的缓存、历史研究的检索。
"""

import logging
from datetime import datetime
from typing import Any

from memory.vector_store import VectorStore

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    记忆管理器

    负责：
    - 存储研究过程中的关键信息（研究发现、搜索结果、论文信息）
    - 检索历史研究记录（语义搜索）
    - 管理不同类型的记忆（短期/长期）
    - 缓存搜索结果，避免重复搜索

    记忆类型：
    - "research": 完整的研究记录（问题 + 发现 + 报告）
    - "search_results": 搜索结果缓存
    - "papers": 论文信息缓存

    会话隔离：
    - 通过 session_id 实现会话级隔离
    - 不同 session_id 的记忆互不可见
    - session_id 为空时全局共享（兼容旧数据）
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        session_id: str = "",
        persist_dir: str = "./chroma_data",
    ):
        """
        初始化记忆管理器

        Args:
            vector_store: VectorStore 实例，如果为 None 则根据 session_id 和 persist_dir 自动创建
            session_id: 会话 ID，用于记忆隔离
            persist_dir: ChromaDB 持久化目录
        """
        if vector_store:
            self.vector_store = vector_store
        else:
            self.vector_store = VectorStore(
                persist_dir=persist_dir,
                session_id=session_id,
            )
        self._initialized = False
        logger.info(f"MemoryManager 初始化，会话ID: {session_id or '无(全局共享)'}")

    async def initialize(self) -> None:
        """
        初始化记忆管理器

        必须在使用其他方法之前调用此方法，
        确保 ChromaDB 客户端已初始化。
        """
        if not self._initialized:
            await self.vector_store.initialize()
            self._initialized = True
            logger.info("MemoryManager 初始化完成")

    async def save_research(
        self,
        topic: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        保存研究记录

        将完整的研究记录存入向量数据库，
        后续可以通过语义搜索检索。

        Args:
            topic: 研究主题/问题
            content: 研究内容（发现、报告等）
            metadata: 额外的元数据

        Returns:
            文档 ID
        """
        self._ensure_initialized()

        # 构造文档
        doc_metadata = {
            "type": "research",
            "topic": topic,
            "timestamp": datetime.now().isoformat(),
        }
        if metadata:
            doc_metadata.update(metadata)

        # 将主题和内容合并存储，方便语义搜索
        full_content = f"研究主题: {topic}\n\n 研究报告：\n{content}"

        doc = {
            "content": full_content,
            "metadata": doc_metadata,
        }

        ids = await self.vector_store.add_documents(
            documents=[doc],
            collection_name="research",
        )

        logger.info(f"保存研究记录: {topic[:30]}... (ID: {ids[0]})")
        return ids[0]

    async def save_search_results(
        self,
        query: str,
        results: list[dict[str, Any]],
    ) -> list[str]:
        """
        保存搜索结果缓存

        将搜索结果存入向量数据库，
        避免对同一关键词重复搜索。

        Args:
            query: 搜索关键词
            results: 搜索结果列表

        Returns:
            文档 ID 列表
        """
        self._ensure_initialized()

        documents = []
        for r in results:
            doc = {
                "content": f"{r.get('title', '')} {r.get('content', '')}",
                "metadata": {
                    "type": "search_result",
                    "query": query,
                    "url": r.get("url", ""),
                    "source": r.get("source", ""),
                    "timestamp": datetime.now().isoformat(),
                },
            }
            documents.append(doc)

        ids = await self.vector_store.add_documents(
            documents=documents,
            collection_name="search_results",
        )

        logger.info(f"缓存 {len(results)} 条搜索结果，关键词: {query}")
        return ids

    async def save_papers(
        self,
        query: str,
        papers: list[dict[str, Any]],
    ) -> list[str]:
        """
        保存论文信息缓存

        Args:
            query: 搜索关键词
            papers: 论文信息列表

        Returns:
            文档 ID 列表
        """
        self._ensure_initialized()

        documents = []
        for p in papers:
            doc = {
                "content": f"{p.get('title', '')} {p.get('abstract', '')}",
                "metadata": {
                    "type": "paper",
                    "query": query,
                    "url": p.get("url", ""),
                    "year": p.get("year"),
                    "authors": ", ".join(p.get("authors", [])[:3]),
                    "timestamp": datetime.now().isoformat(),
                },
            }
            documents.append(doc)

        ids = await self.vector_store.add_documents(
            documents=documents,
            collection_name="papers",
        )

        logger.info(f"缓存 {len(papers)} 篇论文，关键词: {query}")
        return ids

    async def recall_research(
        self, query: str, top_k: int = 3
    ) -> list[dict[str, Any]]:
        """
        检索历史研究记录

        通过语义搜索找到与当前问题最相关的历史研究。

        Args:
            query: 当前研究问题
            top_k: 返回最相关的 k 条记录

        Returns:
            历史研究记录列表
        """
        self._ensure_initialized()

        results = await self.vector_store.search(
            query=query,
            top_k=top_k,
            collection_name="research",
        )

        logger.info(f"检索到 {len(results)} 条历史研究记录")
        return results

    async def recall_search_results(
        self, query: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """
        检索缓存的搜索结果

        查找之前搜索过的相关结果，避免重复搜索。

        Args:
            query: 搜索关键词
            top_k: 返回最相关的 k 条结果

        Returns:
            缓存的搜索结果列表
        """
        self._ensure_initialized()

        results = await self.vector_store.search(
            query=query,
            top_k=top_k,
            collection_name="search_results",
        )

        return results

    async def recall_papers(
        self, query: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """
        检索缓存的论文信息

        查找之前搜索过的相关论文，避免重复搜索。

        Args:
            query: 搜索关键词
            top_k: 返回最相关的 k 篇论文

        Returns:
            缓存的论文信息列表
        """
        self._ensure_initialized()

        results = await self.vector_store.search(
            query=query,
            top_k=top_k,
            collection_name="papers",
        )

        return results

    def _ensure_initialized(self) -> None:
        """确保记忆管理器已初始化"""
        if not self._initialized:
            raise RuntimeError(
                "MemoryManager 未初始化，请先调用 initialize()"
            )
