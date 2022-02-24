import torch.onnx
from Net import Net


def Convert_ONNX(onnxFile):
    # set the model to inference mode
    model.eval()

    # Let's create a dummy input tensor
    dummy_input = torch.randn(1, 3, 60, 160, requires_grad=True)

    # Export the model
    torch.onnx.export(model,  # model being run
                      dummy_input,  # model input (or a tuple for multiple inputs)
                      onnxFile,  # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['modelInput'],  # the model's input names
                      output_names=['modelOutput'],  # the model's output names
                      dynamic_axes={'modelInput': {0: 'batch_size'},  # variable length axes
                                    'modelOutput': {0: 'batch_size'}})
    print('Model has been converted to ONNX')


if __name__ == "__main__":
    model = Net()
    path = "model.pth"
    onnxFile = "mathcode.onnx"
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu'))['model_state_dict'])
    Convert_ONNX(onnxFile)
