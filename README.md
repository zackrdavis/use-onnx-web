# useOnnxWebSession

React hook for initializing and using an inference session with ONNX-Runtime web.

[ONNX-Runtime web documentation](https://onnxruntime.ai/docs/api/js/index.html)

## _This Package is under development and receiving breaking changes. Don't use it for anything important._

## Install

```bash
$ npm install use-onnx-web-session
```

## Usage

```javascript
const { useOnnxWebSession, Tensor } = "use-onnx-web-session"

const App = () => {
  const [ input, setInput ] = useState()
  const [ result, setResult ] = useState()

  // initialize the session
  const { requestInference } = useOnnxWebSession("./model.onnx");

  useEffect(() => {
    // input must be a tensor
    const tensor = new Tensor("float32", input, [1, 100]);

    // await the result
    const res = await requestInference({ feeds: { myInputName: tensor } });

    setResult(res.myOutputName.data)
  }, [input])
}
```

The model.onnx file used above could be created in PyTorch like this:

```python
# set the trained model to inference mode
model.eval()

torch.onnx.export(
  model,
  "model.onnx", # where to save the model
  torch.randn(1, 100), # dummy input for tracing
  input_names=["myInputName"], # the model's input names
  output_names=["myOutputName"], # the model's output names
  export_params=True, # store the trained parameter weights inside the model file
)
```
