import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.NEXUS_BACKEND_URL || "http://127.0.0.1:8000";
const API_TOKEN =
  process.env.NEXUS_API_TOKEN || "test-token-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  return proxy(request, await params);
}

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  return proxy(request, await params);
}

export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  return proxy(request, await params);
}

async function proxy(request: NextRequest, params: { path: string[] }) {
  const path = params.path.join("/");
  const targetUrl = path.startsWith("health")
    ? `${BACKEND_URL}/${path}`
    : `${BACKEND_URL}/api/v1/${path}`;

  const search = request.nextUrl.search;
  const fullUrl = `${targetUrl}${search}`;

  const headers = new Headers();
  headers.set("Authorization", `Bearer ${API_TOKEN}`);
  headers.set("Accept", "application/json");

  const contentType = request.headers.get("content-type");
  if (contentType) {
    headers.set("content-type", contentType);
  }

  const init: RequestInit = {
    method: request.method,
    headers,
  };

  if (["POST", "PUT", "PATCH"].includes(request.method)) {
    init.body = await request.text();
  }

  try {
    const res = await fetch(fullUrl, init);
    const data = await res.text();
    return new NextResponse(data, {
      status: res.status,
      headers: {
        "content-type": res.headers.get("content-type") || "application/json",
      },
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Failed to reach Nexus backend";
    return NextResponse.json(
      {
        error: {
          code: "backend_unavailable",
          message,
        },
      },
      { status: 502 }
    );
  }
}
