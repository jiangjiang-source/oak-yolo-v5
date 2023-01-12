from pathlib import Path
import torch
from torch import nn
import blobconverter

class Model(nn.Module):
    def forward(self, img: torch.Tensor):#, scale: torch.Tensor, mean: torch.Tensor):
        # output = (input - mean) / scale
        # data=img.clone()
        # data[0,:,:]=img[2,:,:]
        # data[1, :, :] = img[1, :, :]
        # data[2,:,:]=img[0,:,:]
        img=img[[2,1,0],:,:]
        return img/255.#torch.div(torch.sub(img, mean), scale)

# Define the expected input shape (dummy input)
shape = (3, 300, 300)
X = torch.ones(shape, dtype=torch.float32)
# Arg = torch.ones((1,1,1), dtype=torch.float32)

path = Path("out/")
path.mkdir(parents=True, exist_ok=True)
onnx_file = "out/normalize.onnx"

print(f"Writing to {onnx_file}")
torch.onnx.export(
    Model(),
    X,
    onnx_file,
    opset_version=12,
    do_constant_folding=True,
    input_names = ['frame'], # Optional
    output_names = ['output'], # Optional
)
onnx_file = "out/normalize.onnx"
# No need for onnx-simplifier here
#
# # Use blobconverter to convert onnx->IR->blob
blobconverter.from_onnx(
    model=onnx_file,
    data_type="FP16",
    shaves=4,
    use_cache=False,
    output_dir="./out",
    optimizer_params=["--reverse_input_channels"],#"--reverse_input_channel"],
    compile_params =[]# ["--ip=U8"]
)

#--mean_values=[127.5,127.5,127.5] --scale_values=[255,255,255]