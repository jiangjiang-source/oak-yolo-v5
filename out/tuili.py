#!/usr/bin/env python3


from pathlib import Path
import sys
import numpy as np
import cv2
import depthai as dai
SHAPE = 300
nnPath='/home/jiang/桌面/depthai-experiments/gen2-mobile-object-localizer/out/normalize_openvino_2021.4_4shave.blob'

p = dai.Pipeline()
p.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

camRgb = p.createColorCamera()
# Model expects values in FP16, as we have compiled it with `-ip FP16`
camRgb.setFp16(True)
camRgb.setInterleaved(False)
camRgb.setPreviewSize(SHAPE, SHAPE)



nn = p.createNeuralNetwork()
nn.setBlobPath(nnPath)
nn.setNumInferenceThreads(2)


camRgb.preview.link(nn.input)

# Send normalized frame values to host
nn_xout = p.createXLinkOut()
nn_xout.setStreamName("nn")
nn_xout.input.setBlocking(False)

inner = p.createXLinkOut()
inner.setStreamName("inner")
inner.input.setBlocking(False)

nn.out.link(nn_xout.input)
camRgb.preview.link(inner.input)


with dai.Device(p) as device:
    qNn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    start=device.getOutputQueue(name="inner",maxSize=4,blocking=False)
    shape = (3, SHAPE, SHAPE)
    while True:
        fr=np.array(start.get().getFrame(),dtype='uint8')
        inNn = np.array(qNn.get().getData())
        frame = inNn.view(np.float16).reshape(shape)
        # Get back the frame. It's currently normalized to -0.5 - 0.5
        # print('inNn:',inNn[:100])
        data=qNn.get().getLayerFp16('output')
        data=np.array(data)#*255
        new=np.array(data,dtype='uint8').reshape(shape)#.transpose(1, 2, 0)

        cv2.imshow('1.jpg', fr.transpose(1, 2, 0))
        cv2.imshow('2.jpg',new.transpose(1,2,0))
        cv2.imwrite('raw.jpg',fr.transpose(1, 2, 0))
        cv2.imwrite('change.jpg',new.transpose(1, 2, 0))

        if cv2.waitKey(1) == ord('q'):
            break