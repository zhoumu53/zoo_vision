function getApiBase(): string {
  if (typeof window === "undefined") {
    const base = process.env.API_BASE_SERVER ?? process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8001";
    return base.endsWith("/api/v1") ? base : `${base}/api/v1`;
  }
  return process.env.NEXT_PUBLIC_API_BASE ?? "/api/v1";
}

export const API_BASE = getApiBase();

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<T>;
}

export async function apiPost<T>(path: string, body?: BodyInit | null): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    body: body ?? null,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<T>;
}

export async function uploadElephantPhoto(
  file: File,
): Promise<{ job_id: number; uuid: string }> {
  const form = new FormData();
  form.append("photo", file);
  return apiPost("/galaxy/upload", form);
}

export async function getUploadStatus(
  jobId: number,
): Promise<import("./types").UploadJobStatus> {
  return apiGet(`/galaxy/upload/${jobId}/status`);
}
