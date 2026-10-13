export type ApiError = Error & { status?: number };

const apiBaseUrl =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/api/v1";

function errorDetail(payload: unknown): string {
  if (
    typeof payload === "object" &&
    payload !== null &&
    "detail" in payload &&
    typeof payload.detail === "string"
  ) {
    return payload.detail;
  }

  return "API request failed without a readable error response.";
}

export async function requestApi<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const response = await fetch(`${apiBaseUrl}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init.headers,
    },
  });
  const payload: unknown = await response.json().catch(() => null);

  if (!response.ok) {
    const error = new Error(errorDetail(payload)) as ApiError;
    error.status = response.status;
    throw error;
  }

  return payload as T;
}

export function formattedJson(value: unknown): string {
  return JSON.stringify(value, null, 2);
}
