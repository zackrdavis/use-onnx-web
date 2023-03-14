# useOnnxWebSession

## React hook for inference with [ONNX-Runtime web](https://onnxruntime.ai/docs/api/js/index.html)

`useOnnxWebSession` manages an ONNX inference session using a model file that you provide, and accepts additional [options](https://onnxruntime.ai/docs/api/js/interfaces/InferenceSession.SessionOptions.html). It will re-initialize if the model file is changed. Only one session should exist in the app at a time.

It returns a single function, `runInference`, which accepts data input and [options](https://onnxruntime.ai/docs/api/js/interfaces/InferenceSession.RunOptions.html), and returns a promise that resolves to inference results. The input properties are determined when you create the model file.

Executing `runInference` while a previous run is in progress will throw an error, so you may want to await or catch:

```
Uncaught (in promise) Error: output [...] already has value: ...
```

## Installation

```bash
$ npm install use-onnx-web-session
```

## Usage

```javascript
import {
  useOnnxWebSession,
  Tensor // For formatting input/output data.
  InferenceSession // For typing assistance.
} from "./useOnnxWebSession";

const App = () => {
  // This model performs matrix multiplication.
  const runInference = useOnnxWebSession("./model.onnx");

  useEffect(() => {
    const feeds = {
      a: new Tensor("float32",
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [3, 4]
      ),
      b: new Tensor("float32",
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
        [4, 3]
      ),
    };

    runInference(feeds).then((result) => {
      console.log(result.c.data);
      // >>> Float32Array(9) [ ... ]
    });
  }, []);
```

The `model.onnx` file used above is included in this repository, but it could be created in PyTorch like this:

```python
# set the trained model to inference mode
model.eval()

# single tensor or tuple for multiple inputs
trace_input = (
  torch.randn(3, 4),
  torch.randn(4, 3)
)

torch.onnx.export(
  model,
  "model.onnx", # where to save the model
  trace_input, # trace input
  input_names=["a, b"], # the model's input names
  output_names=["c"], # the model's output names
  export_params=True, # store the trained parameter weights inside the model file
)
```
