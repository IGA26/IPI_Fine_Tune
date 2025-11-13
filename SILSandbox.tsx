import { useMemo, useState } from "react";
import { Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/components/ui/use-toast";

interface QualityMetrics {
  word_count: number;
  valid_word_ratio: number;
  non_alnum_ratio: number;
  repeated_char_ratio: number;
}

interface PredictionEntry {
  category: "sil" | "emotion";
  prediction: string | number;
  confidence?: number;
  inference_time_ms: number;
  top3?: Array<{ label: string; confidence: number }>;
}

interface ClassificationResponse {
  input_text: string;
  normalized_text: string;
  quality: QualityMetrics;
  predictions: Record<string, PredictionEntry>;
  timing: {
    total_inference_ms: number;
    average_per_model_ms: number;
    measured_time_ms?: number;
  };
}

const PARAMETER_ORDER = [
  "topic",
  "stage",
  "intent",
  "query",
  "advice_risk",
  "domain",
  "emotion",
  "handover",
  "distress",
  "vulnerability",
  "severity",
] as const;

const PARAMETER_LABELS: Record<string, string> = {
  topic: "Topic",
  stage: "Stage",
  intent: "Intent",
  query: "Query",
  advice_risk: "Advice Risk",
  domain: "Domain",
  emotion: "Emotion",
  handover: "Handover",
  distress: "Distress",
  vulnerability: "Vulnerability",
  severity: "Severity",
};

const API_BASE =
  typeof import.meta !== "undefined" && import.meta.env?.VITE_API_BASE_URL
    ? `${import.meta.env.VITE_API_BASE_URL}`.replace(/\/$/, "")
    : "";

const HIGH_CONFIDENCE_THRESHOLD = 0.8;
const LOW_CONFIDENCE_THRESHOLD = 0.6;

interface ClassificationResult {
  parameter: string;
  value: string;
  confidence: string;
  status: "high" | "medium" | "low" | null;
  highlight?: boolean; // Highlight row in light red if true
  paramKey?: string; // Original parameter key for checking values
}

const SILSandbox = () => {
  const [utterance, setUtterance] = useState("What is an ISA ?");
  const [isClassified, setIsClassified] = useState(false);
  const [results, setResults] = useState<ClassificationResult[]>([]);
  const [normalized, setNormalized] = useState("");
  const [timing, setTiming] = useState<{
    measuredMs: number;
    sumMs: number;
    averageMs: number;
  } | null>(null);
  const [quality, setQuality] = useState<QualityMetrics | null>(null);
  const [rawResponse, setRawResponse] = useState<string | null>(null);
  const [detailsVisible, setDetailsVisible] = useState(false);
  const [isClassifying, setIsClassifying] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const { toast } = useToast();

  const classifyEndpoint = `${API_BASE}/sil/infer`.replace(/^\/\//, "/");
  const generateEndpoint = `${API_BASE}/sil/generate`.replace(/^\/\//, "/");

  const hasDetails = Boolean(rawResponse);

  const qualityBadges = useMemo(() => {
    if (!quality) return [];
    return [
      `Word count: ${quality.word_count}`,
      `Valid ratio: ${quality.valid_word_ratio.toFixed(2)}`,
      `Non-alnum: ${quality.non_alnum_ratio.toFixed(2)}`,
      `Repeat chars: ${quality.repeated_char_ratio.toFixed(2)}`,
    ];
  }, [quality]);

  const handleClassify = async () => {
    if (!utterance.trim()) {
      toast({
        variant: "destructive",
        title: "No utterance provided",
        description: "Please enter a user utterance before running classification.",
      });
      return;
    }

    setIsClassifying(true);
    setDetailsVisible(false);
    try {
      const response = await fetch(classifyEndpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: utterance }),
      });

      const payload: ClassificationResponse | { detail?: unknown } | undefined = await response.json().catch(() => undefined);

      if (!response.ok || !payload || !("predictions" in payload)) {
        const detail = (payload as { detail?: any })?.detail ?? payload;
        const message =
          typeof detail === "string"
            ? detail
            : detail?.error ||
              detail?.message ||
              (response.status === 422 ? "Input appears to be gibberish. Please rephrase." : "Failed to classify utterance.");

        const qualityDetail: QualityMetrics | null =
          detail && typeof detail === "object" && "quality" in detail ? (detail.quality as QualityMetrics) : null;

        setIsClassified(false);
        setResults([]);
        setNormalized("");
        setTiming(null);
        setQuality(qualityDetail);
        setRawResponse(null);

        toast({
          variant: "destructive",
          title: "Classification failed",
          description: message,
        });
        return;
      }

      const data = payload as ClassificationResponse;

      const predictionEntries: ClassificationResult[] = PARAMETER_ORDER.filter((param) =>
        Object.prototype.hasOwnProperty.call(data.predictions, param),
      ).map((param) => {
        const entry = data.predictions[param];
        const label = PARAMETER_LABELS[param] ?? param;
        const rawValue = entry?.prediction;
        const value =
          typeof rawValue === "number"
            ? rawValue.toFixed(param === "severity" ? 3 : 2)
            : rawValue ?? "—";
        const score = typeof entry?.confidence === "number" && !Number.isNaN(entry.confidence) ? entry.confidence : null;
        const confidence = score !== null ? `${(score * 100).toFixed(1)}%` : "—";

        let status: ClassificationResult["status"] = null;
        if (score !== null) {
          if (score >= HIGH_CONFIDENCE_THRESHOLD) {
            status = "high";
          } else if (score < LOW_CONFIDENCE_THRESHOLD) {
            status = "low";
          } else {
            status = "medium";
          }
        }

        // Check if row should be highlighted (light red background)
        // Highlight if: emotion == "negative" OR handover/distress/vulnerability == "true"
        const stringValue = String(value).toLowerCase();
        const shouldHighlight =
          (param === "emotion" && stringValue === "negative") ||
          (param === "handover" && stringValue === "true") ||
          (param === "distress" && stringValue === "true") ||
          (param === "vulnerability" && stringValue === "true");

        return {
          parameter: label,
          value: String(value),
          confidence,
          status,
          highlight: shouldHighlight,
          paramKey: param,
        };
      });

      setResults(predictionEntries);
      setNormalized(data.normalized_text ?? data.input_text);
      setQuality(data.quality ?? null);
      const total = data.timing?.total_inference_ms ?? 0;
      const average = data.timing?.average_per_model_ms ?? 0;
      const measured =
        typeof data.timing?.measured_time_ms === "number" && !Number.isNaN(data.timing?.measured_time_ms)
          ? data.timing.measured_time_ms
          : total;
      setTiming({
        measuredMs: measured,
        sumMs: total,
        averageMs: average,
      });
      setRawResponse(JSON.stringify(data, null, 2));
      setIsClassified(true);
      toast({
        title: "Classification complete",
        description: "All SIL and emotion parameters have been evaluated.",
      });
    } catch (error) {
      console.error(error);
      toast({
        variant: "destructive",
        title: "Unexpected error",
        description: error instanceof Error ? error.message : "An unexpected error occurred while classifying.",
      });
    } finally {
      setIsClassifying(false);
    }
  };

  const handleGenerate = async () => {
    setIsGenerating(true);
    try {
      const response = await fetch(generateEndpoint, { method: "POST" });
      const payload = await response.json().catch(() => null);

      if (!response.ok || !payload?.text) {
        toast({
          variant: "destructive",
          title: "Failed to generate query",
          description: payload?.detail ?? "The backend did not return a sample utterance.",
        });
        return;
      }

      setUtterance(payload.text);
      setIsClassified(false);
      setResults([]);
      setNormalized("");
      setTiming(null);
      setQuality(null);
      setRawResponse(null);
      setDetailsVisible(false);

      toast({
        title: "Sample utterance generated",
        description: "Review the text and click classify when ready.",
      });
    } catch (error) {
      console.error(error);
      toast({
        variant: "destructive",
        title: "Unexpected error",
        description: error instanceof Error ? error.message : "Unable to generate a query from the backend.",
      });
    } finally {
      setIsGenerating(false);
    }
  };

  const handleClear = () => {
    setUtterance("");
    setIsClassified(false);
    setResults([]);
    setNormalized("");
    setTiming(null);
    setQuality(null);
    setRawResponse(null);
    setDetailsVisible(false);
  };

  return (
    <div className="min-h-screen bg-background">
      <header className="bg-secondary text-secondary-foreground py-6 px-4 shadow-sm">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-3xl font-bold tracking-tight">SIL Interaction Sandbox</h1>
          <p className="text-secondary-foreground/80 mt-2">
            Type an utterance or let the LLM draft one, then inspect SIL outputs.
          </p>
        </div>
      </header>

      <main className="max-w-6xl mx-auto p-6 space-y-6">
        <Card className="p-6 shadow-lg">
          <div className="space-y-4">
            <div>
              <label htmlFor="utterance" className="text-sm font-semibold text-foreground mb-2 block">
                User Utterance
              </label>
              <Textarea
                id="utterance"
                value={utterance}
                onChange={(e) => setUtterance(e.target.value)}
                placeholder="Enter user utterance here..."
                className="min-h-[120px] resize-none font-mono text-base"
              />
            </div>

            <div className="flex flex-wrap gap-3">
              <Button onClick={handleClassify} className="font-semibold" disabled={isClassifying}>
                {isClassifying ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Classifying…
                  </>
                ) : (
                  "Classify"
                )}
              </Button>
              <Button
                variant="secondary"
                onClick={() => setDetailsVisible((prev) => !prev)}
                disabled={!hasDetails || isClassifying}
              >
                {detailsVisible ? "Hide Details" : "Details"}
              </Button>
              <Button variant="secondary" onClick={handleGenerate} disabled={isGenerating || isClassifying}>
                {isGenerating ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Generating…
                  </>
                ) : (
                  "Generate Query (LLM stub)"
                )}
              </Button>
              <Button variant="outline" onClick={handleClear} disabled={isClassifying}>
                Clear
              </Button>
            </div>

            {isClassified && (
              <div className="pt-4 space-y-4 animate-fade-in">
                <Badge variant="outline" className="text-success border-success">
                  Classification complete.
                </Badge>

                <div className="space-y-2 text-sm">
                  <div>
                    <span className="font-semibold">Original:</span>{" "}
                    <span className="text-muted-foreground">{utterance}</span>
                  </div>
                  <div>
                    <span className="font-semibold">Normalized:</span>{" "}
                    <span className="text-muted-foreground">{normalized}</span>
                  </div>
                  {timing && (
                    <div className="space-y-1">
                      <div>
                        <span className="font-semibold">Wall-clock time:</span>{" "}
                        <span className="text-muted-foreground">{timing.measuredMs.toFixed(1)} ms</span>
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Sum per-model: {timing.sumMs.toFixed(1)} ms • Avg per model: {timing.averageMs.toFixed(1)} ms
                      </div>
                    </div>
                  )}
                </div>

                {qualityBadges.length > 0 && (
                  <div className="flex flex-wrap gap-2">
                    {qualityBadges.map((text) => (
                      <Badge key={text} variant="outline" className="text-xs font-medium">
                        {text}
                      </Badge>
                    ))}
                  </div>
                )}

                <div className="rounded-lg border overflow-hidden">
                  <Table>
                    <TableHeader>
                      <TableRow className="bg-muted/50">
                        <TableHead className="font-bold">Parameter</TableHead>
                        <TableHead className="font-bold">Value</TableHead>
                        <TableHead className="font-bold">Confidence</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {results.map((result, index) => {
                        const confidenceClasses =
                          result.status === "high"
                            ? "text-emerald-600 font-semibold"
                            : result.status === "medium"
                            ? "text-amber-600 font-semibold"
                            : result.status === "low"
                            ? "text-red-600 font-semibold"
                            : "text-muted-foreground";

                        // Apply light red background if highlight is true
                        const rowClasses = result.highlight
                          ? "bg-red-50 hover:bg-red-100 transition-colors"
                          : "hover:bg-muted/30 transition-colors";

                        return (
                          <TableRow key={index} className={rowClasses}>
                            <TableCell className="font-medium">{result.parameter}</TableCell>
                            <TableCell className="font-mono text-sm">{result.value}</TableCell>
                            <TableCell className={`font-mono text-sm ${confidenceClasses}`}>{result.confidence}</TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </div>

                <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
                  <span>
                    <span className="font-semibold text-emerald-600">High</span> ≥ {(HIGH_CONFIDENCE_THRESHOLD * 100).toFixed(0)}%
                  </span>
                  <span>
                    <span className="font-semibold text-amber-600">Medium</span> between{" "}
                    {(LOW_CONFIDENCE_THRESHOLD * 100).toFixed(0)}%–{(HIGH_CONFIDENCE_THRESHOLD * 100).toFixed(0)}%
                  </span>
                  <span>
                    <span className="font-semibold text-red-600">Low</span> &lt; {(LOW_CONFIDENCE_THRESHOLD * 100).toFixed(0)}%
                  </span>
                </div>

                {detailsVisible && rawResponse && (
                  <pre className="max-h-80 overflow-auto bg-muted font-mono text-xs p-4 rounded-lg border">
                    {rawResponse}
                  </pre>
                )}
              </div>
            )}
          </div>
        </Card>
      </main>
    </div>
  );
};

export default SILSandbox;
