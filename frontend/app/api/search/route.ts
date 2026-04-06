import { NextRequest, NextResponse } from "next/server"

const FASTAPI = process.env.FASTAPI_URL ?? "http://localhost:8000"

export async function GET(req: NextRequest) {
  const q     = req.nextUrl.searchParams.get("q") ?? ""
  const limit = req.nextUrl.searchParams.get("limit") ?? "10"

  try {
    const res = await fetch(
      `${FASTAPI}/api/search?q=${encodeURIComponent(q)}&limit=${limit}`,
      { cache: "no-store" }
    )
    const data = await res.json()
    return NextResponse.json(data, { status: res.status })
  } catch (err) {
    console.error("[BFF /search]", err)
    return NextResponse.json({ error: "Backend unreachable" }, { status: 502 })
  }
}
