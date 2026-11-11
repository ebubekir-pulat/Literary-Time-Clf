
import { useState } from "react";
import "./App.css";

interface PredictionResult {
  logistic_regression: {
    prediction: string;
    confidence: number;
  };
  linear_svm: {
    prediction: string;
  };
  qwen: {
    prediction: string;
    confidence: number | null;
    explanation: string;
  };
}

function App() {
  const [text, setText] = useState("");
  const [results, setResults] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handlePredict() {
    if (!text.trim()) {
      setError("Please enter a text excerpt.");
      return;
    }

    setLoading(true);
    setError("");
    setResults(null);

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || "Something went wrong.");
      }

      const data: PredictionResult = await response.json();
      setResults(data);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Unable to connect to the API."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="container">
      <h1>Literary Time Classifier</h1>

      <p>
        Enter a literary excerpt to classify its literary time period.
      </p>

      <textarea
        placeholder="Paste your literary excerpt here..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />

      <button onClick={handlePredict} disabled={loading}>
        {loading ? "Analysing..." : "Classify Text"}
      </button>

      {error && <p className="error">{error}</p>}

      {results && (
        <div className="results">
          <h2>Classification Results</h2>

          <div className="result-card">
            <h3>Logistic Regression</h3>
            <p>Prediction: {results.logistic_regression.prediction}</p>
            <p>
              Confidence:{" "}
              {(results.logistic_regression.confidence * 100).toFixed(2)}%
            </p>
          </div>

          <div className="result-card">
            <h3>Linear SVM</h3>
            <p>Prediction: {results.linear_svm.prediction}</p>
          </div>

          <div className="result-card">
            <h3>Qwen LLM</h3>
            <p>Prediction: {results.qwen.prediction}</p>
            <p>
              Confidence:{" "}
              {results.qwen.confidence != null
                ? `${(results.qwen.confidence * 100).toFixed(2)}%`
                : "Not available"}
            </p>
            <p>Explanation: {results.qwen.explanation}</p>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
