import { NextResponse } from "next/server"

const FASTAPI_URL = process.env.FASTAPI_URL ?? "http://localhost:8000"

export async function GET() {
  try {
    const res = await fetch(`${FASTAPI_URL}/api/model/health`, {
      cache: "no-store",
    })
    const data = await res.json()
    return NextResponse.json(data, { status: res.status })
  } catch {
    return NextResponse.json({ error: "Backend unreachable" }, { status: 502 })
  }
}
