import { ApiScreen } from "@/app/components/api-screen";

export default function ResultsPage() {
  return (
    <ApiScreen
      action={{ label: "Build Redis ANN index", path: "/retrieval/index", body: {} }}
      description="Inspect the newest completed run: ALS versus baselines, RMSE/MAE, Recall@K/NDCG@K, and the exact ranking evaluation protocol."
      readPath="/results/latest"
      title="Results"
    />
  );
}
