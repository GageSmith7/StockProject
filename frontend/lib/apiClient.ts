import type {
  SearchResponse,
  PortfolioRequest,
  PortfolioResponse,
  SimulateRequest,
  SimulateResponse,
  PredictRequest,
  PredictResponse,
  ModelHealthResponse,
  OptimizeRequest,
  OptimizeResult,
} from "@/types/api"

// All requests go to Next.js API routes (BFF layer), never directly to FastAPI.
// NEXT_PUBLIC_API_URL is set in .env.local — defaults to empty string (same origin).
const BASE = process.env.NEXT_PUBLIC_API_URL ?? ""

async function post<TBody, TResponse>(
  path: string,
  body: TBody
): Promise<TResponse> {
  const res = await fetch(`${BASE}${path}`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(body),
  })
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(`${res.status}: ${detail}`)
  }
  return res.json() as Promise<TResponse>
}

async function get<TResponse>(path: string): Promise<TResponse> {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(`${res.status}: ${detail}`)
  }
  return res.json() as Promise<TResponse>
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export const apiClient = {
  search(q: string, limit = 10): Promise<SearchResponse> {
    const params = new URLSearchParams({ q, limit: String(limit) })
    return get<SearchResponse>(`/api/search?${params}`)
  },

  portfolio(req: PortfolioRequest): Promise<PortfolioResponse> {
    return post<PortfolioRequest, PortfolioResponse>("/api/portfolio", req)
  },

  simulate(req: SimulateRequest): Promise<SimulateResponse> {
    return post<SimulateRequest, SimulateResponse>("/api/simulate", req)
  },

  predict(req: PredictRequest): Promise<PredictResponse> {
    return post<PredictRequest, PredictResponse>("/api/predict", req)
  },

  modelHealth(): Promise<ModelHealthResponse> {
    return get<ModelHealthResponse>("/api/model-health")
  },

  optimize(req: OptimizeRequest): Promise<OptimizeResult> {
    return post<OptimizeRequest, OptimizeResult>("/api/optimize", req)
  },
}
