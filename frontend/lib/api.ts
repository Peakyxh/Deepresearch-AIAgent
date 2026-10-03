import type { RunDetail, RunSummary } from "./types";

export const API_BASE = (
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000"
).replace(/\/$/, "");

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail || `请求失败 (${response.status})`);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export const api = {
  listRuns: () => request<RunSummary[]>("/api/runs"),
  getRun: (runId: string) => request<RunDetail>(`/api/runs/${runId}`),
  createRun: (query: string) =>
    request<RunDetail>("/api/runs", {
      method: "POST",
      body: JSON.stringify({ query }),
    }),
  cancelRun: (runId: string) =>
    request<{ status: string }>(`/api/runs/${runId}/cancel`, {
      method: "POST",
    }),
  respond: (
    runId: string,
    payload: {
      interaction_id: string;
      answers?: Array<{ question: string; answer: string }>;
      feedback?: string;
    },
  ) =>
    request<void>(`/api/runs/${runId}/responses`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
};
