#!/usr/bin/env python3
import cv2
import depthai as dai
import numpy as np
import argparse
import time
import blobconverter

'''
Mobile object localizer demo running on device on RGB camera.
Run as:
python3 -m pip install -r requirements.txt
python3 main.py

Link to the original model:
https://tfhub.dev/google/lite-model/object_detection/mobile_object_localizer_v1/1/default/1

Blob taken from:
https://github.com/PINTO0309/PINTO_model_zoo/tree/main/151_object_detection_mobile_object_localizer
'''

# --------------- Arguments ---------------
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--threshold', type=float, help="Coonfidence threshold", default=0.2)

args = parser.parse_args()
THRESHOLD = args.threshold
NN_PATH = blobconverter.from_zoo(name="mobile_object_localizer_192x192", zoo_type="depthai", shaves=6)
NN_WIDTH = 224
NN_HEIGHT = 224
PREVIEW_WIDTH = 600
PREVIEW_HEIGHT = 600

# 定义一个数据流管道
pipeline = dai.Pipeline()#创造pipline
pipeline.setOpenVINOVersion(version = dai.OpenVINO.VERSION_2021_4)


# 彩色相机和灰度相机的定义与初始化
cam = pipeline.create(dai.node.ColorCamera)
cam.setPreviewSize(PREVIEW_WIDTH, PREVIEW_HEIGHT)
cam.setInterleaved(False)
cam.setFps(50)

camleft = pipeline.create(dai.node.MonoCamera)
camleft.setBoardSocket(dai.CameraBoardSocket.LEFT)
camleft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
camleft.setFps(50)

##########图像resize#############################
# Create manip
manip = pipeline.create(dai.node.ImageManip)
manip.initialConfig.setResize(NN_WIDTH, NN_HEIGHT)
manip.initialConfig.setFrameType(dai.ImgFrame.Type.RGB888p)
manip.initialConfig.setKeepAspectRatio(False)

newmanip = pipeline.create(dai.node.ImageManip)
newmanip.initialConfig.setResize(NN_WIDTH, NN_HEIGHT)
newmanip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)


# 将需要尺寸改变的映射到节点中
cam.preview.link(manip.inputImage)
camleft.out.link(newmanip.inputImage)


#定义神经网络
# nn = pipeline.createNeuralNetwork()
# nn.setBlobPath(NN_PATH)

#将神经网络需要的数据映射给神经网络
# manip.out.link(nn.inputs["left"])


# 创造最终输出的节点
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("cam")
xout_rgb.input.setBlocking(False)

xout_manip = pipeline.create(dai.node.XLinkOut)
xout_manip.setStreamName("manip")
xout_manip.input.setBlocking(False)

xout_gray=pipeline.create(dai.node.XLinkOut)
xout_gray.setStreamName("camleft")
xout_gray.input.setBlocking(False)

xout_mainleft=pipeline.create(dai.node.XLinkOut)
xout_mainleft.setStreamName("mainleft")
xout_mainleft.input.setBlocking(False)

#将数据映射到 最终输出节点
cam.preview.link(xout_rgb.input)
manip.out.link(xout_manip.input)
camleft.out.link(xout_gray.input)
newmanip.out.link(xout_mainleft.input)


# Pipeline defined, now the device is assigned and pipeline is started
with dai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    q_cam = device.getOutputQueue(name="cam", maxSize=4, blocking=False)#####原图像
    q_manip = device.getOutputQueue(name="manip", maxSize=4, blocking=False)#####更改后的图像
    g_cam=device.getOutputQueue(name="camleft",maxSize=4,blocking=False)
    g_newmain=device.getOutputQueue(name="mainleft",maxSize=4,blocking=False)

    while True:
        in_cam = q_cam.get()
        in_manip = q_manip.get()
        gray=g_cam.get()
        in_gray=g_newmain.get()
        frame = in_cam.getCvFrame()


        frame_manip = in_manip.getCvFrame()
        frame_manip = cv2.cvtColor(frame_manip, cv2.COLOR_RGB2BGR)
        frame1 = gray.getCvFrame()
        frame2 = in_gray.getCvFrame()


        print(frame.shape,frame_manip.shape,frame1.shape,frame2.shape)

        cv2.imshow("Localizer", frame)
        cv2.imshow("Manip + NN", frame_manip)
        cv2.imshow("gray", frame1)
        cv2.imshow("resize",frame2)
        if cv2.waitKey(1) == ord('q'):
            break