import { ApiScreen } from "@/app/components/api-screen";

export default function ProcessingPage() {
  return (
    <ApiScreen
      description="Review the latest immutable Parquet output, data-quality checks, timestamp boundaries, and temporal split counts."
      readPath="/processing/latest"
      title="Processing"
    />
  );
}
