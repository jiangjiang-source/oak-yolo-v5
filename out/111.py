from pathlib import Path

import numpy as np
import torch
from torch import nn
import blobconverter

class Model(nn.Module):
    def forward(self, img: torch.Tensor):#, scale: torch.Tensor, mean: torch.Tensor):
        # output = (input - mean) / scale
        # img=torch.tensor(img,dtype=torch.float)
        data=img.clone()
        data[0,:,:]=img[2,:,:]
        data[1, :, :] = img[1, :, :]
        data[2,:,:]=img[0,:,:]
        return data#/1.#torch.div(torch.sub(img, mean), scale)

# Define the expected input shape (dummy input)
shape = (3, 300, 300)
X = torch.ones(shape, dtype=torch.float32)
# Arg = torch.ones((1,1,1), dtype=torch.float32)
model=Model()

import cv2

img=cv2.imread('raw.jpg')
data=torch.from_numpy(img.transpose(2,0,1))
output=model(data)
print(img.shape,output.shape)
output=output.detach().numpy().transpose(1,2,0)
while True:
    output=np.array(output,dtype='int')

    # cv2.imshow('raw',img)
    # cv2.imshow('new',output)
    cv2.imwrite('raw.jpg',img)
    cv2.imwrite('new.jpg', output)

    if cv2.waitKey(1) == ord('q'):
        break
