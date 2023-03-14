import { useRef, useEffect } from "react";
import { InferenceSession, Tensor } from "onnxruntime-web";

// re-export
export { InferenceSession, Tensor };

/**
 * @param model Path to .onnx model file
 * @param options [SessionOptions docs](https://onnxruntime.ai/docs/api/js/interfaces/InferenceSession.SessionOptions)
 *
 * Defaults to:
 * ```
 * { executionProviders: ["webgl"], graphOptimizationLevel: "all" }
 * ```
 * @returns A promise that resolves to the results of an inference run
 */
export const useOnnxWebSession = (
  model: string,
  options: InferenceSession.SessionOptions = {
    executionProviders: ["webgl"],
    graphOptimizationLevel: "all",
  }
) => {
  // Save the session create promise instead of the session itself.
  // If runInference is requested before the session is ready, wait for the session, then run it.
  const sessionPromise = useRef<Promise<InferenceSession>>();

  useEffect(() => {
    sessionPromise.current = InferenceSession.create(model, options).then(
      (s) => s
    );
  }, [model, options]);

  const runInference = async (
    feeds: InferenceSession.OnnxValueMapType,
    options?: InferenceSession.RunOptions
  ) =>
    new Promise<InferenceSession.OnnxValueMapType>((resolve, reject) => {
      if (!sessionPromise.current) {
        // If for some reason sessionPromise has not been created:
        reject("Something's wrong, InferenceSession has not been initialized.");
      } else {
        // Return a promise of inference, waiting for session init if necessary.
        resolve(sessionPromise.current.then((s) => s.run(feeds, options)));
      }
    });

  return runInference;
};
