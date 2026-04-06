import { NextRequest, NextResponse } from "next/server"

const FASTAPI = process.env.FASTAPI_URL ?? "http://localhost:8000"

export async function POST(req: NextRequest) {
  try {
    const body = await req.json()
    const res  = await fetch(`${FASTAPI}/api/predict`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(body),
      cache:   "no-store",
    })
    const data = await res.json()
    return NextResponse.json(data, { status: res.status })
  } catch (err) {
    console.error("[BFF /predict]", err)
    return NextResponse.json({ error: "Backend unreachable" }, { status: 502 })
  }
}
