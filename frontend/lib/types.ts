export type RunStatus =
  | "queued"
  | "running"
  | "waiting_input"
  | "completed"
  | "failed"
  | "cancelled";

export interface RunSummary {
  id: string;
  query: string;
  status: RunStatus;
  current_phase: string;
  created_at: string;
  updated_at: string;
  error: string;
  source_count: number;
  critique_score: number;
}

export interface RunEvent {
  id: number;
  type: string;
  created_at: string;
  data: Record<string, unknown>;
}

export interface Interaction {
  id: string;
  kind: "clarification" | "outline_review";
  payload: {
    round?: number;
    questions?: Array<{ question: string; options?: string[] }>;
    outline?: string;
    outline_json?: ReportOutline | string;
  };
}

export interface ReportOutline {
  title?: string;
  summary_points?: string[];
  sections?: Array<{
    heading?: string;
    key_arguments?: string[];
  }>;
  conclusion_points?: string[];
}

export interface ResearchSource {
  title?: string;
  url?: string;
  type?: string;
  authors?: string[] | string;
  year?: string | number;
}

export interface SubQuestion {
  id?: string;
  question?: string;
  depends_on?: string[];
  priority?: number;
}

export interface ResearchState {
  research_plan?: string;
  structured_sub_questions?: SubQuestion[];
  sub_question_results?: Array<{
    sub_question_id?: string;
    sub_question?: string;
    findings?: string;
    sources?: ResearchSource[];
  }>;
  sources?: ResearchSource[];
  critique_total_score?: number;
  critique_passed?: boolean;
  critique_feedback?: string;
  report?: string;
  token_usage?: {
    estimated?: boolean;
    total_calls?: number;
    input_tokens?: number;
    output_tokens?: number;
    by_agent?: Record<string, {
      calls: number;
      input_tokens: number;
      output_tokens: number;
    }>;
  };
}

export interface RunDetail extends RunSummary {
  state: ResearchState;
  pending_interaction: Interaction | null;
  events: RunEvent[];
}
