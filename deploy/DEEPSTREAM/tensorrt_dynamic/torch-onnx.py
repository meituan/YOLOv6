from torchvision import models
import torch

if __name__ == "__main__":
    net = models.resnet18(pretrained=True)
    net.eval()
    dummy_input = torch.rand((2, 3, 224, 224))
    dynamic_axes = {"input": {0: "batch"}, "output":{0:"batch"}}
    torch.onnx.export(net, dummy_input, "resnet18_dynamic.onnx",
                        input_names=["input"], output_names=["output"],
                        export_params=True, dynamic_axes=dynamic_axes,
                        opset_version=11)
    # static batch size
    torch.onnx.export(net, dummy_input, "resnet18_static.onnx",
                        input_names=["input"], output_names=["output"],
                        export_params=True, 
                        opset_version=11)