import { ApiScreen } from "@/app/components/api-screen";

export default function TrainingPage() {
  return (
    <ApiScreen
      action={{
        label: "Train ALS",
        path: "/training",
        body: {
          rank: 80,
          max_iter: 15,
          reg_param: 0.08,
          seed: 42,
          nonnegative: true,
          positive_rating_threshold: 4,
          ranking_k: 10,
          ranking_user_limit: 5000,
          negative_samples: 100,
        },
      }}
      description="Run Spark MLlib ALS after processing completes. Training writes a model artifact, model card, validation metrics, test metrics, and baseline comparison."
      readPath="/training/runs"
      title="Training"
    />
  );
}
