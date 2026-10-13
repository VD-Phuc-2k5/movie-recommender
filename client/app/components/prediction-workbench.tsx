"use client";

import { FormEvent, useEffect, useState } from "react";

import { formattedJson, requestApi } from "@/app/components/api";

type Mode = "auto" | "exact" | "ann";

export function PredictionWorkbench() {
  const [userId, setUserId] = useState("");
  const [movieId, setMovieId] = useState("");
  const [topK, setTopK] = useState("10");
  const [strategy, setStrategy] = useState<Mode>("auto");
  const [servingStatus, setServingStatus] = useState<unknown>(null);
  const [prediction, setPrediction] = useState<unknown>(null);
  const [recommendations, setRecommendations] = useState<unknown>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    void requestApi<unknown>("/serving/status")
      .then(setServingStatus)
      .catch((requestError: unknown) => {
        setError(requestError instanceof Error ? requestError.message : "Unknown API error.");
      });
  }, []);

  function requestBody(includeMovie: boolean) {
    const parsedUserId = Number(userId);
    const parsedMovieId = Number(movieId);
    const parsedTopK = Number(topK);
    if (!Number.isInteger(parsedUserId) || parsedUserId <= 0) {
      throw new Error("Enter a positive integer user ID.");
    }
    if (includeMovie && (!Number.isInteger(parsedMovieId) || parsedMovieId <= 0)) {
      throw new Error("Enter a positive integer movie ID for rating prediction.");
    }
    if (!Number.isInteger(parsedTopK) || parsedTopK < 1 || parsedTopK > 100) {
      throw new Error("Top-K must be an integer from 1 to 100.");
    }
    return { parsedUserId, parsedMovieId, parsedTopK };
  }

  async function submitPrediction(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsLoading(true);
    setError(null);
    try {
      const body = requestBody(true);
      setPrediction(
        await requestApi<unknown>("/predictions", {
          method: "POST",
          body: JSON.stringify({ user_id: body.parsedUserId, movie_id: body.parsedMovieId }),
        }),
      );
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Unknown API error.");
    } finally {
      setIsLoading(false);
    }
  }

  async function submitRecommendations() {
    setIsLoading(true);
    setError(null);
    try {
      const body = requestBody(false);
      setRecommendations(
        await requestApi<unknown>("/recommendations", {
          method: "POST",
          body: JSON.stringify({
            user_id: body.parsedUserId,
            k: body.parsedTopK,
            strategy,
          }),
        }),
      );
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Unknown API error.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <section className="space-y-6">
      <header className="border-b border-zinc-200 pb-6">
        <p className="text-xs font-semibold tracking-[0.16em] text-indigo-600 uppercase">
          Live model serving
        </p>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight text-zinc-950">
          Prediction &amp; Recommendation
        </h1>
        <p className="mt-2 max-w-2xl text-sm leading-6 text-zinc-600">
          Score a user/movie pair or return unseen Top-K films from the persisted ALS model.
          Redis ANN is only used when its real index is available.
        </p>
      </header>

      <form
        className="grid gap-4 rounded-xl border border-zinc-200 bg-white p-5 shadow-sm md:grid-cols-4"
        onSubmit={(event) => void submitPrediction(event)}
      >
        <label className="field-label">
          User ID
          <input className="field-input" inputMode="numeric" onChange={(event) => setUserId(event.target.value)} value={userId} />
        </label>
        <label className="field-label">
          Movie ID
          <input className="field-input" inputMode="numeric" onChange={(event) => setMovieId(event.target.value)} value={movieId} />
        </label>
        <label className="field-label">
          Top-K
          <input className="field-input" inputMode="numeric" onChange={(event) => setTopK(event.target.value)} value={topK} />
        </label>
        <label className="field-label">
          Candidate strategy
          <select className="field-input" onChange={(event) => setStrategy(event.target.value as Mode)} value={strategy}>
            <option value="auto">Auto</option>
            <option value="exact">Exact ALS</option>
            <option value="ann">Redis ANN</option>
          </select>
        </label>
        <div className="flex flex-wrap gap-3 md:col-span-4">
          <button className="button-primary" disabled={isLoading} type="submit">
            Predict rating
          </button>
          <button className="button-secondary" disabled={isLoading} onClick={() => void submitRecommendations()} type="button">
            {isLoading ? "Running…" : "Recommend Top-K"}
          </button>
        </div>
      </form>

      {error ? <p className="api-error">{error}</p> : null}
      <div className="grid gap-6 xl:grid-cols-2">
        <ResponseCard title="Serving status" value={servingStatus} />
        <ResponseCard title="Rating prediction" value={prediction} />
        <ResponseCard title="Top-K recommendations" value={recommendations} wide />
      </div>
    </section>
  );
}

function ResponseCard({ title, value, wide = false }: { title: string; value: unknown; wide?: boolean }) {
  return (
    <article className={`rounded-xl border border-zinc-200 bg-white shadow-sm ${wide ? "xl:col-span-2" : ""}`}>
      <h2 className="border-b border-zinc-100 px-5 py-3 text-sm font-medium text-zinc-800">{title}</h2>
      <pre className="api-response">{value === null ? "No API response yet." : formattedJson(value)}</pre>
    </article>
  );
}
