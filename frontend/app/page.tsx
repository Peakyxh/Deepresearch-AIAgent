"use client";

import {
  BookOpen,
  Check,
  ChevronLeft,
  ChevronRight,
  CircleAlert,
  CircleStop,
  Clipboard,
  Clock3,
  FileText,
  Globe2,
  LoaderCircle,
  Menu,
  MessageSquareText,
  PanelRightClose,
  PanelRightOpen,
  Plus,
  RotateCcw,
  Search,
  Send,
  Sparkles,
  X,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { API_BASE, api } from "@/lib/api";
import type {
  Interaction,
  ResearchSource,
  RunDetail,
  RunEvent,
  RunStatus,
  RunSummary,
} from "@/lib/types";

const terminalStatuses = new Set<RunStatus>(["completed", "failed", "cancelled"]);

const phaseNames: Record<string, string> = {
  queued: "等待启动",
  initializing: "初始化",
  clarifier: "澄清目标",
  planner: "制定计划",
  research: "检索研究",
  critic: "质量审查",
  writer: "撰写报告",
  completed: "已完成",
  failed: "运行失败",
  cancelled: "已取消",
};

const eventCopy: Record<string, { title: string; detail: string }> = {
  run_queued: { title: "研究任务已创建", detail: "等待工作流接管任务" },
  run_started: { title: "工作流已启动", detail: "正在初始化研究上下文" },
  interaction_required: { title: "需要你的确认", detail: "工作流已暂停，等待输入" },
  interaction_resolved: { title: "已收到反馈", detail: "工作流正在继续执行" },
  run_completed: { title: "研究已完成", detail: "最终报告已经生成" },
  run_failed: { title: "研究运行失败", detail: "请检查错误信息后重试" },
  run_cancelled: { title: "研究已取消", detail: "任务已安全停止" },
};

function statusLabel(status: RunStatus) {
  return {
    queued: "排队中",
    running: "运行中",
    waiting_input: "等待确认",
    completed: "已完成",
    failed: "失败",
    cancelled: "已取消",
  }[status];
}

function formatTime(value: string) {
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function eventPresentation(event: RunEvent) {
  if (event.type === "phase_started") {
    const phase = String(event.data.phase || "");
    return { title: `${phaseNames[phase] || phase}开始`, detail: "Agent 正在处理这一阶段" };
  }
  if (event.type === "phase_completed") {
    const phase = String(event.data.phase || "");
    return { title: `${phaseNames[phase] || phase}完成`, detail: "结果已写入任务状态" };
  }
  if (event.type === "agent_log") {
    return {
      title: String(event.data.agent || "Agent"),
      detail: String(event.data.message || ""),
    };
  }
  return eventCopy[event.type] || { title: event.type, detail: "状态已更新" };
}

function uniqueSources(detail: RunDetail | null): ResearchSource[] {
  if (!detail) return [];
  const nested = detail.state.sub_question_results?.flatMap((item) => item.sources || []) || [];
  const all = [...(detail.state.sources || []), ...nested];
  const seen = new Set<string>();
  return all.filter((source) => {
    const key = source.url || source.title || JSON.stringify(source);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

export default function Workspace() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [inspectorTab, setInspectorTab] = useState<"plan" | "sources" | "report">("plan");
  const [rightOpen, setRightOpen] = useState(true);
  const [leftOpen, setLeftOpen] = useState(false);
  const [answers, setAnswers] = useState<string[]>([]);
  const [feedback, setFeedback] = useState("");
  const [copied, setCopied] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const activeRun = runs.find((run) => !terminalStatuses.has(run.status));
  const hasActiveRun = Boolean(activeRun);

  const refreshRuns = useCallback(async () => {
    const items = await api.listRuns();
    setRuns(items);
    setSelectedId((current) => current || items[0]?.id || null);
  }, []);

  const refreshDetail = useCallback(async (runId: string) => {
    const result = await api.getRun(runId);
    setDetail(result);
    setRuns((current) => {
      const summary: RunSummary = result;
      return [summary, ...current.filter((item) => item.id !== runId)].sort((a, b) =>
        b.updated_at.localeCompare(a.updated_at),
      );
    });
  }, []);

  useEffect(() => {
    refreshRuns()
      .catch((reason: Error) => setError(`无法连接 API：${reason.message}`))
      .finally(() => setLoading(false));
  }, [refreshRuns]);

  useEffect(() => {
    if (!hasActiveRun) return;
    const timer = window.setInterval(() => {
      refreshRuns().catch(() => undefined);
    }, 4000);
    return () => window.clearInterval(timer);
  }, [hasActiveRun, refreshRuns]);

  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      return;
    }
    setError("");
    refreshDetail(selectedId).catch((reason: Error) => setError(reason.message));
  }, [selectedId, refreshDetail]);

  useEffect(() => {
    if (!selectedId || !detail || terminalStatuses.has(detail.status)) return;
    const lastId = detail.events.at(-1)?.id || 0;
    const source = new EventSource(`${API_BASE}/api/runs/${selectedId}/events?after=${lastId}`);
    source.onmessage = (message) => {
      const event = JSON.parse(message.data) as RunEvent;
      setDetail((current) => {
        if (!current || current.id !== selectedId || current.events.some((item) => item.id === event.id)) {
          return current;
        }
        return { ...current, events: [...current.events, event] };
      });
      if (event.type !== "agent_log") {
        refreshDetail(selectedId).catch(() => undefined);
      }
    };
    source.onerror = () => source.close();
    return () => source.close();
  }, [selectedId, detail?.status, refreshDetail]);

  useEffect(() => {
    const interaction = detail?.pending_interaction;
    setAnswers(interaction?.payload.questions?.map(() => "") || []);
    setFeedback("");
  }, [detail?.pending_interaction?.id]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [detail?.events.length]);

  const sources = useMemo(() => uniqueSources(detail), [detail]);

  async function createRun() {
    const trimmed = query.trim();
    if (!trimmed || submitting || loading) return;
    if (activeRun) {
      setSelectedId(activeRun.id);
      setError("当前研究尚未结束，请等待任务完成、取消任务，或先处理待确认的问题。");
      return;
    }
    setSubmitting(true);
    setError("");
    try {
      const created = await api.createRun(trimmed);
      setQuery("");
      setDetail(created);
      setSelectedId(created.id);
      setRuns((current) => [created, ...current]);
      setLeftOpen(false);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "创建任务失败");
    } finally {
      setSubmitting(false);
    }
  }

  async function submitInteraction(interaction: Interaction) {
    setSubmitting(true);
    setError("");
    try {
      if (interaction.kind === "clarification") {
        const questions = interaction.payload.questions || [];
        await api.respond(detail!.id, {
          interaction_id: interaction.id,
          answers: questions.map((item, index) => ({
            question: item.question,
            answer: answers[index]?.trim() || "无补充",
          })),
        });
      } else {
        await api.respond(detail!.id, {
          interaction_id: interaction.id,
          feedback: feedback.trim(),
        });
      }
      await refreshDetail(detail!.id);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "提交失败");
    } finally {
      setSubmitting(false);
    }
  }

  async function cancelRun() {
    if (!detail) return;
    try {
      await api.cancelRun(detail.id);
      await refreshDetail(detail.id);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "取消失败");
    }
  }

  const report = detail?.state.report || "";

  return (
    <main className="app-shell">
      <aside className={`sidebar ${leftOpen ? "mobile-open" : ""}`}>
        <div className="brand-row">
          <div className="brand-mark"><Sparkles size={16} /></div>
          <div><strong>DeepResearch</strong><span>Agent workspace</span></div>
          <button className="icon-button mobile-only" onClick={() => setLeftOpen(false)} aria-label="关闭任务栏"><X size={18} /></button>
        </div>
        <button
          className="new-task"
          disabled={hasActiveRun}
          title={hasActiveRun ? "当前研究完成后才能新建任务" : "新建研究"}
          onClick={() => { setSelectedId(null); setDetail(null); setLeftOpen(false); }}
        >
          <Plus size={16} /> 新建研究
          <span>⌘ N</span>
        </button>
        <div className="sidebar-label">最近任务</div>
        <div className="run-list">
          {loading && <div className="muted-row"><LoaderCircle className="spin" size={15} /> 加载任务</div>}
          {!loading && !runs.length && <div className="empty-list">还没有研究任务</div>}
          {runs.map((run) => (
            <button
              className={`run-item ${selectedId === run.id ? "active" : ""}`}
              key={run.id}
              onClick={() => { setSelectedId(run.id); setLeftOpen(false); }}
            >
              <span className={`status-dot ${run.status}`} />
              <span className="run-item-copy"><strong>{run.query}</strong><small>{phaseNames[run.current_phase] || statusLabel(run.status)} · {formatTime(run.updated_at)}</small></span>
              <ChevronRight size={14} />
            </button>
          ))}
        </div>
        <div className="sidebar-footer"><span className="online-dot" /> API · {API_BASE.replace(/^https?:\/\//, "")}</div>
      </aside>

      {leftOpen && <button className="mobile-scrim" aria-label="关闭" onClick={() => setLeftOpen(false)} />}

      <section className="workspace">
        <header className="topbar">
          <button className="icon-button mobile-only" onClick={() => setLeftOpen(true)} aria-label="打开任务栏"><Menu size={19} /></button>
          <div className="title-block">
            <span>{detail ? "研究任务" : "新任务"}</span>
            <strong>{detail?.query || "开始一次深入研究"}</strong>
          </div>
          <button
            className="icon-button"
            disabled={hasActiveRun}
            title={hasActiveRun ? "当前研究完成后才能新建任务" : "新建研究"}
            onClick={() => { setSelectedId(null); setDetail(null); setQuery(""); }}
            aria-label="新建研究"
          >
            <Plus size={18} />
          </button>
          {detail && <div className={`status-pill ${detail.status}`}><span />{statusLabel(detail.status)}</div>}
          {detail && !terminalStatuses.has(detail.status) && (
            <button className="stop-button" onClick={cancelRun}><CircleStop size={15} /> 停止</button>
          )}
          <button className="icon-button" onClick={() => setRightOpen((value) => !value)} aria-label="切换检查器">
            {rightOpen ? <PanelRightClose size={18} /> : <PanelRightOpen size={18} />}
          </button>
        </header>

        {error && <div className="error-banner"><CircleAlert size={16} /><span>{error}</span><button onClick={() => setError("")}><X size={15} /></button></div>}

        <div className="content-scroll">
          {!detail ? (
            <div className="welcome">
              <div className="welcome-icon"><Search size={26} /></div>
              <p className="eyebrow">RESEARCH WORKSPACE</p>
              <h1>把一个问题，研究透彻。</h1>
              <p>Agent 会先澄清目标，再拆解、检索、审查并撰写带来源的报告。关键节点由你确认。</p>
              <div className="suggestions">
                {["分析生成式 AI 对软件工程岗位的影响", "调研家庭储能市场的技术路线与格局", "比较主流 Agent 框架的架构与适用场景"].map((item) => (
                  <button key={item} onClick={() => setQuery(item)}>{item}<ChevronRight size={14} /></button>
                ))}
              </div>
            </div>
          ) : (
            <div className="timeline-wrap">
              <div className="run-heading">
                <p className="eyebrow">{detail.id}</p>
                <h1>{detail.query}</h1>
                <div className="run-meta"><Clock3 size={14} /> 创建于 {formatTime(detail.created_at)}<span />{detail.events.length} 个事件</div>
              </div>

              <div className="timeline">
                {detail.events.map((event) => {
                  const content = eventPresentation(event);
                  const isLog = event.type === "agent_log";
                  return (
                    <div className={`timeline-item ${isLog ? "log" : event.type}`} key={event.id}>
                      <div className="timeline-rail"><span>{event.type === "run_completed" || event.type === "phase_completed" ? <Check size={12} /> : event.type === "interaction_required" ? <MessageSquareText size={12} /> : isLog ? <span className="tiny-dot" /> : <LoaderCircle size={12} />}</span></div>
                      <div className="timeline-copy"><div><strong>{content.title}</strong><time>{formatTime(event.created_at)}</time></div><p>{content.detail}</p></div>
                    </div>
                  );
                })}
              </div>

              {detail.pending_interaction && (
                <InteractionCard
                  interaction={detail.pending_interaction}
                  answers={answers}
                  feedback={feedback}
                  submitting={submitting}
                  onAnswer={(index, value) => setAnswers((items) => items.map((item, itemIndex) => itemIndex === index ? value : item))}
                  onFeedback={setFeedback}
                  onSubmit={() => submitInteraction(detail.pending_interaction!)}
                />
              )}

              {detail.status === "failed" && <div className="terminal-card error"><CircleAlert size={19} /><div><strong>研究未完成</strong><p>{detail.error || "工作流遇到了未知错误。"}</p></div><button onClick={() => { setQuery(detail.query); setSelectedId(null); setDetail(null); }}><RotateCcw size={15} /> 重新发起</button></div>}
              {detail.status === "completed" && <div className="terminal-card success"><Check size={19} /><div><strong>报告已生成</strong><p>在右侧“报告”标签中查看完整结果。</p></div><button onClick={() => { setInspectorTab("report"); setRightOpen(true); }}><FileText size={15} /> 查看报告</button></div>}
              <div ref={bottomRef} />
            </div>
          )}
        </div>

        <div className="composer-wrap">
          <div className={`composer ${hasActiveRun ? "locked" : ""}`}>
            <textarea
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              disabled={loading || hasActiveRun}
              onKeyDown={(event) => {
                if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
                  event.preventDefault();
                  createRun();
                }
              }}
              placeholder={hasActiveRun
                ? activeRun?.status === "waiting_input"
                  ? "请先完成上方的确认，研究随后会继续…"
                  : "当前研究正在进行，完成后可输入新的问题…"
                : "描述你想深入研究的问题…"}
              rows={2}
            />
            <div className="composer-actions">
              <span>{hasActiveRun ? `${phaseNames[activeRun?.current_phase || ""] || "研究"}进行中` : "Ctrl ↵ 运行"}</span>
              <button disabled={loading || hasActiveRun || !query.trim() || submitting} onClick={createRun} aria-label="开始研究">{submitting ? <LoaderCircle className="spin" size={17} /> : hasActiveRun ? <LoaderCircle className="spin" size={17} /> : <Send size={17} />}</button>
            </div>
          </div>
          <p>Agent 可能会犯错，请核对重要来源和结论。</p>
        </div>
      </section>

      <aside className={`inspector ${rightOpen ? "open" : ""}`}>
        <div className="inspector-tabs">
          <button className={inspectorTab === "plan" ? "active" : ""} onClick={() => setInspectorTab("plan")}><BookOpen size={14} />计划</button>
          <button className={inspectorTab === "sources" ? "active" : ""} onClick={() => setInspectorTab("sources")}><Globe2 size={14} />来源{sources.length > 0 && <span>{sources.length}</span>}</button>
          <button className={inspectorTab === "report" ? "active" : ""} onClick={() => setInspectorTab("report")}><FileText size={14} />报告</button>
        </div>
        <div className="inspector-body">
          {!detail && <InspectorEmpty icon={<BookOpen size={22} />} text="选择或创建一个任务后，这里会显示计划、来源和报告。" />}
          {detail && inspectorTab === "plan" && <PlanPanel detail={detail} />}
          {detail && inspectorTab === "sources" && <SourcesPanel sources={sources} />}
          {detail && inspectorTab === "report" && (
            report ? <div className="report-panel"><div className="panel-heading"><div><p>FINAL REPORT</p><h2>研究报告</h2></div><button className="icon-button" onClick={async () => { await navigator.clipboard.writeText(report); setCopied(true); setTimeout(() => setCopied(false), 1500); }} aria-label="复制报告">{copied ? <Check size={16} /> : <Clipboard size={16} />}</button></div><article className="markdown"><ReactMarkdown remarkPlugins={[remarkGfm]}>{report}</ReactMarkdown></article></div> : <InspectorEmpty icon={<FileText size={22} />} text="报告会在研究与审查完成后出现在这里。" />
          )}
        </div>
      </aside>
    </main>
  );
}

function InteractionCard({ interaction, answers, feedback, submitting, onAnswer, onFeedback, onSubmit }: {
  interaction: Interaction;
  answers: string[];
  feedback: string;
  submitting: boolean;
  onAnswer: (index: number, value: string) => void;
  onFeedback: (value: string) => void;
  onSubmit: () => void;
}) {
  const questions = interaction.payload.questions || [];
  const [questionIndex, setQuestionIndex] = useState(0);

  useEffect(() => {
    setQuestionIndex(0);
  }, [interaction.id]);

  const currentQuestion = questions[questionIndex];
  const currentAnswer = answers[questionIndex] || "";
  const isLastQuestion = questionIndex >= questions.length - 1;

  return (
    <section className="interaction-card">
      <div className="interaction-title"><span><MessageSquareText size={17} /></span><div><p>WORKFLOW PAUSED</p><h2>{interaction.kind === "clarification" ? "确认研究边界" : "审阅报告大纲"}</h2></div></div>
      {interaction.kind === "clarification" ? (
        currentQuestion ? (
          <div className="question-step">
            <div className="question-progress">
              <span>问题 {questionIndex + 1} / {questions.length}</span>
              <div>{questions.map((_, index) => <i className={index <= questionIndex ? "active" : ""} key={index} />)}</div>
            </div>
            <h3>{currentQuestion.question}</h3>
            {currentQuestion.options?.length ? (
              <div className="answer-options">
                {currentQuestion.options.map((option) => (
                  <button
                    className={currentAnswer === option ? "selected" : ""}
                    type="button"
                    key={option}
                    onClick={() => onAnswer(questionIndex, option)}
                  >
                    <span>{currentAnswer === option && <Check size={12} />}</span>
                    {option}
                  </button>
                ))}
              </div>
            ) : null}
            <label className="custom-answer">
              <span>{currentQuestion.options?.length ? "或者输入其他回答" : "你的回答"}</span>
              <textarea
                rows={2}
                value={currentAnswer}
                onChange={(event) => onAnswer(questionIndex, event.target.value)}
                placeholder="输入补充说明…"
              />
            </label>
          </div>
        ) : <p className="empty-copy">没有需要回答的问题，可以直接继续。</p>
      ) : (
        <div className="outline-review"><pre>{interaction.payload.outline || "大纲已生成"}</pre><label><span>修改意见（没有意见可直接确认）</span><textarea rows={3} value={feedback} onChange={(event) => onFeedback(event.target.value)} placeholder="例如：增加技术风险章节，压缩市场背景…" /></label></div>
      )}
      <div className="interaction-actions">
        {interaction.kind === "clarification" && questionIndex > 0 ? (
          <button className="secondary" type="button" onClick={() => setQuestionIndex((index) => index - 1)}><ChevronLeft size={15} /> 上一题</button>
        ) : <span>提交后任务将从当前节点继续</span>}
        {interaction.kind === "clarification" && !isLastQuestion ? (
          <button type="button" disabled={!currentAnswer.trim()} onClick={() => setQuestionIndex((index) => index + 1)}>下一题 <ChevronRight size={15} /></button>
        ) : (
          <button disabled={submitting} onClick={onSubmit}>{submitting ? <LoaderCircle className="spin" size={15} /> : <Check size={15} />} {interaction.kind === "clarification" ? "提交回答并继续" : "确认并继续"}</button>
        )}
      </div>
    </section>
  );
}

function InspectorEmpty({ icon, text }: { icon: React.ReactNode; text: string }) {
  return <div className="inspector-empty"><span>{icon}</span><p>{text}</p></div>;
}

function PlanPanel({ detail }: { detail: RunDetail }) {
  const questions = detail.state.structured_sub_questions || [];
  const tokenUsage = detail.state.token_usage;
  return (
    <div className="plan-panel">
      <div className="panel-heading"><div><p>RESEARCH PLAN</p><h2>{phaseNames[detail.current_phase] || detail.current_phase}</h2></div>{detail.critique_score > 0 && <span className="score">{detail.critique_score}</span>}</div>
      <div className="phase-track">{["clarifier", "planner", "research", "critic", "writer"].map((phase) => { const phases = ["clarifier", "planner", "research", "critic", "writer", "completed"]; const activeIndex = phases.indexOf(detail.current_phase); const itemIndex = phases.indexOf(phase); return <div className={`${itemIndex < activeIndex || detail.status === "completed" ? "done" : ""} ${itemIndex === activeIndex ? "current" : ""}`} key={phase}><span>{itemIndex < activeIndex || detail.status === "completed" ? <Check size={11} /> : itemIndex + 1}</span><p>{phaseNames[phase]}</p></div>; })}</div>
      {tokenUsage && (tokenUsage.total_calls || 0) > 0 && (
        <section className="usage-card">
          <div><span>估算输入</span><strong>{(tokenUsage.input_tokens || 0).toLocaleString()}</strong></div>
          <div><span>估算输出</span><strong>{(tokenUsage.output_tokens || 0).toLocaleString()}</strong></div>
          <div><span>LLM 调用</span><strong>{tokenUsage.total_calls || 0}</strong></div>
          <small>基于字符数估算，最终账单以模型提供者为准</small>
        </section>
      )}
      {detail.state.research_plan && <section className="inspector-section"><h3>研究策略</h3><p className="preserve-lines">{detail.state.research_plan}</p></section>}
      <section className="inspector-section"><h3>子问题 <span>{questions.length}</span></h3>{questions.length ? <ol className="question-plan">{questions.map((item, index) => <li key={item.id || index}><span>{String(index + 1).padStart(2, "0")}</span><div><strong>{item.question || "待定问题"}</strong>{item.depends_on?.length ? <small>依赖 {item.depends_on.join(", ")}</small> : <small>可独立研究</small>}</div></li>)}</ol> : <p className="empty-copy">完成目标澄清后将生成结构化计划。</p>}</section>
      {detail.state.critique_feedback && <section className="inspector-section"><h3>审查反馈</h3><p className="preserve-lines">{detail.state.critique_feedback}</p></section>}
    </div>
  );
}

function SourcesPanel({ sources }: { sources: ResearchSource[] }) {
  if (!sources.length) return <InspectorEmpty icon={<Globe2 size={22} />} text="检索开始后，去重后的参考来源会出现在这里。" />;
  return <div className="sources-panel"><div className="panel-heading"><div><p>SOURCE LIBRARY</p><h2>{sources.length} 个参考来源</h2></div></div><div className="source-list">{sources.map((source, index) => <a href={source.url || undefined} target="_blank" rel="noreferrer" className="source-card" key={`${source.url || source.title}-${index}`}><span>{index + 1}</span><div><strong>{source.title || "未命名来源"}</strong><p>{source.type || "web"}{source.year ? ` · ${source.year}` : ""}</p><small>{source.url || "暂无链接"}</small></div><ChevronRight size={14} /></a>)}</div></div>;
}
