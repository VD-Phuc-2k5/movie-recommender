"use client";

import { useCallback, useEffect, useState } from "react";

import { formattedJson, requestApi } from "@/app/components/api";

type Action = {
  label: string;
  path: string;
  body: Record<string, unknown>;
};

type ApiScreenProps = {
  title: string;
  description: string;
  readPath: string;
  action?: Action;
};

export function ApiScreen({
  title,
  description,
  readPath,
  action,
}: ApiScreenProps) {
  const [payload, setPayload] = useState<unknown>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const refresh = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      setPayload(await requestApi<unknown>(readPath));
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Unknown API error.");
    } finally {
      setIsLoading(false);
    }
  }, [readPath]);

  useEffect(() => {
    let isActive = true;

    async function loadInitialResponse() {
      try {
        const initialPayload = await requestApi<unknown>(readPath);
        if (isActive) {
          setPayload(initialPayload);
        }
      } catch (requestError) {
        if (isActive) {
          setError(requestError instanceof Error ? requestError.message : "Unknown API error.");
        }
      } finally {
        if (isActive) {
          setIsLoading(false);
        }
      }
    }

    void loadInitialResponse();
    return () => {
      isActive = false;
    };
  }, [readPath]);

  async function runAction() {
    if (!action) {
      return;
    }

    setIsSubmitting(true);
    setError(null);
    try {
      const result = await requestApi<unknown>(action.path, {
        method: "POST",
        body: JSON.stringify(action.body),
      });
      setPayload(result);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Unknown API error.");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <section className="space-y-6">
      <header className="flex flex-col justify-between gap-4 border-b border-zinc-200 pb-6 sm:flex-row sm:items-end">
        <div>
          <p className="text-xs font-semibold tracking-[0.16em] text-indigo-600 uppercase">
            Live API data
          </p>
          <h1 className="mt-2 text-3xl font-semibold tracking-tight text-zinc-950">
            {title}
          </h1>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-zinc-600">{description}</p>
        </div>
        <div className="flex gap-3">
          <button className="button-secondary" onClick={() => void refresh()} type="button">
            Refresh
          </button>
          {action ? (
            <button
              className="button-primary"
              disabled={isSubmitting}
              onClick={() => void runAction()}
              type="button"
            >
              {isSubmitting ? "Running…" : action.label}
            </button>
          ) : null}
        </div>
      </header>

      <div className="rounded-xl border border-zinc-200 bg-white shadow-sm">
        <div className="flex items-center justify-between border-b border-zinc-100 px-5 py-3">
          <span className="text-sm font-medium text-zinc-800">{readPath}</span>
          <span className="text-xs text-zinc-500">
            {isLoading ? "Loading actual state…" : "Last response"}
          </span>
        </div>
        {error ? <p className="api-error">{error}</p> : null}
        {isLoading ? (
          <p className="px-5 py-8 text-sm text-zinc-500">Fetching backend state…</p>
        ) : (
          <pre className="api-response">{formattedJson(payload)}</pre>
        )}
      </div>
    </section>
  );
}
