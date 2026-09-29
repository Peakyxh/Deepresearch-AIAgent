"""
轻量级 DAG 工作流引擎

自研编排框架，不依赖 LangGraph/AutoGen，实现 DAG 拓扑执行。
支持固定边、条件边、并发执行、执行快照等能力。

核心概念：
- Node：异步执行函数，接收 state，返回 state 更新
- Edge：固定边（A → B）或条件边（A → router_fn → B/C/...）
- DAG：有向无环图，按拓扑排序执行
- 并发：同一层无依赖的节点通过 asyncio.gather 并发执行

使用示例：
    workflow = Workflow()
    workflow.add_node("planner", planner_fn)
    workflow.add_node("researcher", researcher_fn)
    workflow.add_node("critic", critic_fn)
    workflow.add_node("writer", writer_fn)

    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "critic")
    workflow.add_conditional_edges("critic", route_fn, {"writer": "writer", "researcher": "researcher"})
    workflow.add_edge("writer", END)

    result = await workflow.run(initial_state)
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)

# 终止标记：当边指向 END 时，表示工作流结束
END = "__END__"

# 节点函数类型：接收 state 字典，返回更新后的 state 字典
NodeFunc = Callable[[dict[str, Any]], Coroutine[Any, Any, dict[str, Any]]]

# 路由函数类型：接收 state 字典，返回下一个节点名称
RouterFunc = Callable[[dict[str, Any]], str]


class NodeStatus(str, Enum):
    """节点执行状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class NodeResult:
    """节点执行结果"""
    name: str
    status: NodeStatus
    state: dict[str, Any] | None = None
    error: str | None = None
    duration_ms: float = 0.0


@dataclass
class WorkflowSnapshot:
    """
    工作流执行快照

    记录工作流在某个时刻的完整状态，可用于：
    - 调试：查看每个节点执行后的状态变化
    - 断点续跑：从某个快照恢复执行
    - 审计：追踪状态变化的完整历史
    """
    # 当前正在执行的节点
    current_node: str = ""
    # 已完成的节点及其结果
    completed_nodes: list[NodeResult] = field(default_factory=list)
    # 当前状态
    state: dict[str, Any] = field(default_factory=dict)
    # 执行步数
    step: int = 0


class Workflow:
    """
    轻量级 DAG 工作流引擎

    支持：
    - 固定边：A 执行完后必定执行 B
    - 条件边：A 执行完后根据路由函数决定执行哪个节点
    - 并发执行：同一层无依赖的节点并发执行（fan-out / fan-in）
    - 执行快照：每个节点执行后记录状态快照
    - 最大步数限制：防止条件边导致的无限循环
    """

    def __init__(self, max_steps: int = 50):
        """
       初始化工作流

        Args:
            max_steps: 最大执行步数，防止无限循环（默认 50）
        """
        self._nodes: dict[str, NodeFunc] = {}
        self._edges: dict[str, list[str]] = defaultdict(list)
        self._conditional_edges: dict[str, tuple[RouterFunc, dict[str, str]]] = {}
        self._entry_point: str = ""
        self._max_steps = max_steps

        # 执行快照历史
        self._snapshots: list[WorkflowSnapshot] = []

        logger.info(f"Workflow 初始化，最大步数: {max_steps}")

    def add_node(self, name: str, func: NodeFunc) -> "Workflow":
        """
        添加节点

        Args:
            name: 节点名称（唯一标识）
            func: 异步执行函数，接收 state 字典，返回更新后的 state 字典

        Returns:
            self（支持链式调用）

        Raises:
            ValueError: 节点名称已存在
        """
        if name in self._nodes:
            raise ValueError(f"节点 '{name}' 已存在")
        if name == END:
            raise ValueError(f"节点名称不能使用保留字 '{END}'")

        self._nodes[name] = func
        logger.debug(f"添加节点: {name}")
        return self

    def add_edge(self, from_node: str, to_node: str | list[str]) -> "Workflow":
        """
        添加固定边

        from_node 执行完后，必定执行 to_node。
        支持一对多（fan-out）：from_node → [to_node1, to_node2]

        Args:
            from_node: 源节点名称
            to_node: 目标节点名称或名称列表

        Returns:
            self（支持链式调用）

        Raises:
            ValueError: 节点不存在
        """
        self._validate_node_exists(from_node)

        if isinstance(to_node, list):
            for target in to_node:
                self._validate_node_exists(target)
                self._edges[from_node].append(target)
                logger.debug(f"添加边: {from_node} → {target}")
        else:
            if to_node != END:
                self._validate_node_exists(to_node)
            self._edges[from_node].append(to_node)
            logger.debug(f"添加边: {from_node} → {to_node}")

        return self

    def add_conditional_edges(
        self,
        from_node: str,
        router: RouterFunc,
        mapping: dict[str, str],
    ) -> "Workflow":
        """
        添加条件边

        from_node 执行完后，调用 router(state) 获取下一个节点名称，
        然后在 mapping 中查找对应的目标节点。

        Args:
            from_node: 源节点名称
            router: 路由函数，接收 state，返回下一个节点的 key
            mapping: 路由映射，key 为 router 返回值，value 为目标节点名称

        Returns:
            self（支持链式调用）

        Raises:
            ValueError: 节点不存在或映射中的目标节点不存在

        示例：
            workflow.add_conditional_edges(
                "critic",
                lambda state: "writer" if state["passed"] else "researcher",
                {"writer": "writer", "researcher": "researcher"},
            )
        """
        self._validate_node_exists(from_node)

        for key, target in mapping.items():
            if target != END:
                self._validate_node_exists(target)

        self._conditional_edges[from_node] = (router, mapping)
        logger.debug(f"添加条件边: {from_node} → router({list(mapping.keys())})")
        return self

    def set_entry_point(self, name: str) -> "Workflow":
        """
        设置入口节点

        Args:
            name: 入口节点名称

        Returns:
            self（支持链式调用）

        Raises:
            ValueError: 节点不存在
        """
        self._validate_node_exists(name)
        self._entry_point = name
        logger.debug(f"设置入口节点: {name}")
        return self

    async def run(self, initial_state: dict[str, Any]) -> dict[str, Any]:
        """
        执行工作流

        从入口节点开始，按边的定义依次执行节点。
        支持固定边顺序执行、条件边动态路由、并发执行。

        Args:
            initial_state: 初始状态字典

        Returns:
            最终状态字典

        Raises:
            RuntimeError: 未设置入口节点、超过最大步数、节点执行失败
        """
        if not self._entry_point:
            raise RuntimeError("未设置入口节点，请调用 set_entry_point()")

        self._snapshots.clear()
        state = dict(initial_state)
        current_nodes = [self._entry_point]
        step = 0

        logger.info(f"工作流开始执行，入口节点: {self._entry_point}")

        while current_nodes and step < self._max_steps:
            # 并发执行当前层的所有节点
            if len(current_nodes) == 1:
                # 单节点：直接执行
                node_name = current_nodes[0]
                state = await self._execute_node(node_name, state, step)
                next_nodes = self._get_next_nodes(node_name, state)
            else:
                # 多节点：并发执行
                state, next_nodes = await self._execute_parallel(current_nodes, state, step)

            step += 1
            current_nodes = next_nodes

        if step >= self._max_steps:
            logger.warning(f"工作流超过最大步数限制 ({self._max_steps})，强制终止")

        logger.info(f"工作流执行完成，共 {step} 步")
        return state

    async def _execute_node(
        self, name: str, state: dict[str, Any], step: int
    ) -> dict[str, Any]:
        """
        执行单个节点

        Args:
            name: 节点名称
            state: 当前状态
            step: 当前步数

        Returns:
            更新后的状态

        Raises:
            RuntimeError: 节点不存在
        """
        if name not in self._nodes:
            raise RuntimeError(f"节点 '{name}' 不存在")

        import time
        start_time = time.monotonic()

        logger.info(f"[Step {step}] 执行节点: {name}")

        try:
            func = self._nodes[name]
            new_state = await func(state)

            # 合并状态：新状态覆盖旧状态
            if new_state:
                state.update(new_state)

            duration_ms = (time.monotonic() - start_time) * 1000
            result = NodeResult(
                name=name,
                status=NodeStatus.COMPLETED,
                state=state,
                duration_ms=duration_ms,
            )

            logger.info(f"[Step {step}] 节点 '{name}' 执行完成 ({duration_ms:.0f}ms)")

        except Exception as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            result = NodeResult(
                name=name,
                status=NodeStatus.FAILED,
                error=str(e),
                duration_ms=duration_ms,
            )
            logger.error(f"[Step {step}] 节点 '{name}' 执行失败: {e}")
            raise RuntimeError(f"节点 '{name}' 执行失败: {e}") from e

        # 记录快照
        snapshot = WorkflowSnapshot(
            current_node=name,
            completed_nodes=[result],
            state=dict(state),
            step=step,
        )
        self._snapshots.append(snapshot)

        return state

    async def _execute_parallel(
        self, node_names: list[str], state: dict[str, Any], step: int
    ) -> tuple[dict[str, Any], list[str]]:
        """
        并发执行多个节点

        同一层无依赖的节点通过 asyncio.gather 并发执行。
        所有节点共享同一个 state，执行结果合并到 state 中。

        Args:
            node_names: 要并发执行的节点名称列表
            state: 当前状态
            step: 当前步数

        Returns:
            (更新后的状态, 下一批要执行的节点名称列表)
        """
        logger.info(f"[Step {step}] 并发执行节点: {node_names}")

        async def run_single(name: str) -> tuple[str, dict[str, Any]]:
            """执行单个节点并返回 (名称, 状态更新)"""
            node_state = dict(state)  # 每个节点拿到独立的 state 副本
            func = self._nodes[name]
            result = await func(node_state)
            return name, result or {}

        # 并发执行所有节点
        results = await asyncio.gather(
            *[run_single(name) for name in node_names],
            return_exceptions=True,
        )

        # 合并结果
        next_nodes_set = set()
        for i, result in enumerate(results):
            name = node_names[i]

            if isinstance(result, Exception):
                logger.error(f"[Step {step}] 并发节点 '{name}' 执行失败: {result}")
                raise RuntimeError(f"节点 '{name}' 执行失败: {result}") from result

            _, state_update = result
            state.update(state_update)

            # 收集每个节点的后续节点
            node_next = self._get_next_nodes(name, state)
            next_nodes_set.update(node_next)

            # 记录快照
            snapshot = WorkflowSnapshot(
                current_node=name,
                completed_nodes=[NodeResult(
                    name=name,
                    status=NodeStatus.COMPLETED,
                    state=state,
                )],
                state=dict(state),
                step=step,
            )
            self._snapshots.append(snapshot)

        logger.info(f"[Step {step}] 并发执行完成: {node_names}")
        return state, list(next_nodes_set)

    def _get_next_nodes(self, node_name: str, state: dict[str, Any]) -> list[str]:
        """
        获取节点执行完后的下一批节点

        优先检查条件边，如果没有条件边则使用固定边。

        Args:
            node_name: 当前节点名称
            state: 当前状态

        Returns:
            下一批要执行的节点名称列表
        """
        # 优先检查条件边
        if node_name in self._conditional_edges:
            router, mapping = self._conditional_edges[node_name]
            route_key = router(state)
            next_node = mapping.get(route_key)

            if next_node is None:
                logger.warning(f"条件边路由结果 '{route_key}' 不在映射中，工作流结束")
                return []

            if next_node == END:
                logger.info(f"节点 '{node_name}' → END")
                return []

            logger.info(f"节点 '{node_name}' → {next_node} (条件路由: {route_key})")
            return [next_node]

        # 使用固定边
        next_nodes = self._edges.get(node_name, [])

        # 过滤掉 END
        filtered = [n for n in next_nodes if n != END]

        if not filtered and END in next_nodes:
            logger.info(f"节点 '{node_name}' → END")
            return []

        return filtered

    def get_snapshots(self) -> list[WorkflowSnapshot]:
        """
        获取执行快照历史

        Returns:
            快照列表，按执行顺序排列
        """
        return list(self._snapshots)

    def get_last_snapshot(self) -> WorkflowSnapshot | None:
        """
        获取最后一个执行快照

        Returns:
            最后一个快照，如果没有则返回 None
        """
        return self._snapshots[-1] if self._snapshots else None

    def _validate_node_exists(self, name: str) -> None:
        """
        验证节点是否存在

        Args:
            name: 节点名称

        Raises:
            ValueError: 节点不存在
        """
        if name != END and name not in self._nodes:
            raise ValueError(f"节点 '{name}' 不存在，请先通过 add_node() 添加")
