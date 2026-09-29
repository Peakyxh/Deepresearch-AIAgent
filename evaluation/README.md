# 评估系统 (Evaluation)

对 DeepResearch 生成的研究报告进行多维度评估，包括**质量评估**、**重复性评估**和**事实核查**。

## 评估维度

| 维度 | 评估器 | 说明 | 评分 |
|------|--------|------|------|
| 质量 | QualityEvaluator | 全面性、连贯性、清晰度、洞察力、总体 | 0-4 分 |
| 重复性 | RepeatabilityEvaluator | 段落间内容重复程度 | 0-4 分（4=几乎无重复） |
| 事实核查 | FactEvaluator | 引用句子是否被来源网页支持 | -1 不支持 / 0 不确定 / 1 支持 |

## 环境配置

在 `.env` 文件中配置以下参数：

```bash
# 评估专用 LLM 提供者（留空则使用全局 llm_provider）
EVALUATOR_LLM_PROVIDER=qwen

# Jina API 密钥（事实核查需要，用于抓取引用网页内容）
JINA_API_KEY=your_jina_api_key
```

## 执行流程

### 第一步：转换报告为 JSONL 格式

将 `outputs` 目录下的 Markdown 研究报告转换为评估用的 JSONL 文件：

```bash
cd deepresearch_agent/evaluation
python convert_to_jsonl.py --input ../outputs --output ./data/reports.jsonl
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | `../outputs` | 报告目录路径（包含 `.md` 文件） |
| `--output` | `./data/reports.jsonl` | 输出 JSONL 文件路径 |

**输出格式**（每行一条记录）：

```json
{"id": 1, "topic": "研究问题", "report": "# 研究报告全文..."}
```

### 第二步：执行评估

```bash
python run_evaluation.py --mode <模式> --input <输入文件>
```

**三种评估模式：**

#### 1. 单文件评估（file）

评估单个 Markdown 报告文件：

```bash
python run_evaluation.py --mode file --input ../outputs/report.md
```

#### 2. 批量评估（batch）

评估 JSONL 文件中的所有报告（第一步的输出）：

```bash
python run_evaluation.py --mode batch --input ./data/reports.jsonl --output ./results
```

支持断点续传，跳过已评估的报告：

```bash
python run_evaluation.py --mode batch --input ./data/reports.jsonl --resume
```

#### 3. 交互式评估（interactive）

交互式输入主题和报告内容进行评估：

```bash
python run_evaluation.py --mode interactive
```

输入 `q` 退出。

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | `interactive` | 评估模式：`file` / `batch` / `interactive` |
| `--input` | 无 | 输入文件路径（file/batch 模式必填） |
| `--output` | `settings.eval_output_dir` | 输出目录（batch 模式） |
| `--resume` | `False` | 断点续传（batch 模式） |

## 输出结果

每篇报告生成一个 `{file_id}.json` 文件，包含三个维度的完整评估结果：

```json
{
  "file_id": "abc123",
  "topic": "研究问题",
  "quality": {
    "comprehensiveness_score": 3.5,
    "coherence_score": 3.0,
    "clarity_score": 3.5,
    "insightfulness_score": 3.0,
    "overall_score": 3.0,
    "reason": "评分理由...",
    "success": true,
    "error": null
  },
  "repeatability": {
    "avg_repeat_score": 3.2,
    "pair_results": [],
    "total_pairs": 15,
    "success_pairs": 15
  },
  "avg_repeat_score": 3.2,
  "fact": {
    "total_checks": 30,
    "success_checks": 28,
    "supported": 22,
    "uncertain": 4,
    "not_supported": 2,
    "fact_score": 0.7857,
    "items": []
  },
  "fact_score": 0.7857,
  "success": true,
  "error": null
}
```

**关键字段说明：**

- `quality` — 五维度质量评分（0-4 分），`average_score` 为四维度均值
- `repeatability.avg_repeat_score` — 重复性评分（4=几乎无重复，0=过度重复）
- `fact.fact_score` — 事实支持率 = supported / success_checks

## 事实核查优化策略

为控制 token 消耗，事实核查采用以下优化：

1. **区分正文和参考文献** — 只从正文提取引用句子，参考文献区域仅用于 URL 映射
2. **按句子去重** — 同一句子引用多个来源时，合并网页内容后只调用一次 LLM
3. **最大核查数限制** — 默认最多核查 30 个句子，超出时随机采样

## 文件结构

```
evaluation/
├── __init__.py                 # 模块导出
├── convert_to_jsonl.py         # 报告转 JSONL
├── run_evaluation.py           # 评估入口脚本
├── evaluator.py                # 统一评估器（整合三个子评估器）
├── quality_evaluator.py        # 质量评估器
├── repeatability_evaluator.py  # 重复性评估器
├── fact_evaluator.py           # 事实核查评估器
├── prompts.py                  # 所有评估提示词
├── data/                       # JSONL 数据目录
└── README.md
```
