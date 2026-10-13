import { ApiScreen } from "@/app/components/api-screen";

export default function DatasetsPage() {
  return (
    <ApiScreen
      action={{
        label: "Prepare MovieLens data",
        path: "/datasets/prepare",
        body: { train_ratio: 0.8, validation_ratio: 0.1, test_ratio: 0.1 },
      }}
      description="Inspect the local MovieLens 32M archive and launch the schema-first preparation run. The response is the emitted preprocessing manifest."
      readPath="/datasets"
      title="Datasets"
    />
  );
}
