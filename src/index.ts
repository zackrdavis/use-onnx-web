import { useState, useRef, useEffect } from "react";
import { InferenceSession } from "onnxruntime-web";

export type RunProps = {
  feeds: InferenceSession.OnnxValueMapType;
  options?: InferenceSession.RunOptions;
};

/**
 * @param model Path to .onnx model file
 * @param options Defaults to `{executionProviders:["webgl"],graphOptimizationLevel:"all"}`
 * https://onnxruntime.ai/docs/api/js/interfaces/InferenceSession.SessionOptions.html
 * @returns A function that returns a promise. This promise resolves to the results of an inference run
 */
export const useOnnxWebSession = (
  model: string,
  options: InferenceSession.SessionOptions = {
    executionProviders: ["webgl"],
    graphOptimizationLevel: "all",
  }
) => {
  const [session, setSession] = useState<InferenceSession>();
  const makingSession = useRef(false);

  const makeSession = async () => {
    makingSession.current = true;
    const sess = await InferenceSession.create(model, options);
    setSession(sess);
  };

  useEffect(() => {
    if (!session && !makingSession.current) {
      makeSession();
    }
  }, []);

  const requestInference = async ({ feeds, options }: RunProps) => {
    // TODO: Make sure this can't be called too soon
    return await session!.run(feeds, options);
  };

  return { requestInference };
};
