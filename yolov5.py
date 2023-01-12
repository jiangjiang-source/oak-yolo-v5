#!/usr/bin/env python3
import cv2
import depthai as dai
import numpy as np
import argparse
import time
# import blobconverter
# from after_precess import DecodeBox
import torch

from jiema import YOLO
from PIL import Image
# --------------- Arguments ---------------
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--threshold', type=float, help="Coonfidence threshold", default=0.2)

args = parser.parse_args()
THRESHOLD = args.threshold
NN_PATH ='out/models_openvino_2021.4_6shave.blob'     #blobconverter.from_zoo(name="mobile_object_localizer_192x192", zoo_type="depthai", shaves=6)
NN_WIDTH = 192
NN_HEIGHT = 192
PREVIEW_WIDTH = 800
PREVIEW_HEIGHT = 800

# --------------- Pipeline ---------------
pipeline = dai.Pipeline()

pipeline.setOpenVINOVersion(version = dai.OpenVINO.VERSION_2021_4)


# 彩色相机初始化
cam = pipeline.create(dai.node.ColorCamera)
cam.setPreviewSize(PREVIEW_WIDTH, PREVIEW_HEIGHT)
# cam.setFp16(True)
cam.setInterleaved(False)
cam.setFps(30)


# 图像修改尺寸
manip = pipeline.create(dai.node.ImageManip)
manip.setMaxOutputFrameSize(1800000)
manip.initialConfig.setResize(NN_WIDTH, NN_HEIGHT)
manip.initialConfig.setFrameType(dai.ImgFrame.Type.RGB888p)
manip.initialConfig.setKeepAspectRatio(False)

# 图像修改尺寸
cam.preview.link(manip.inputImage)

# # 神经网络定义
detection_nn = pipeline.create(dai.node.NeuralNetwork)
detection_nn.setBlobPath(NN_PATH)
detection_nn.setNumPoolFrames(4)
detection_nn.input.setBlocking(False)
detection_nn.setNumInferenceThreads(2)

# #将数据传输到神经网络
# manip.out.link(detection_nn.input)

# 定义三个输出节点
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("cam")
xout_rgb.input.setBlocking(False)

xout_manip = pipeline.create(dai.node.XLinkOut)
xout_manip.setStreamName("manip")
xout_manip.input.setBlocking(False)


nn_xin = pipeline.create(dai.node.XLinkIn)
nn_xin.setStreamName("nnInput")


xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("nnoutput")
xout_nn.input.setBlocking(False)

#数据传输到节点
cam.preview.link(xout_rgb.input)
manip.out.link(xout_manip.input)

nn_xin.out.link(detection_nn.input)
detection_nn.out.link(xout_nn.input)

# Pipeline defined, now the device is assigned and pipeline is started
with dai.Device(pipeline) as device:
    np.random.seed(0)
    colors_full = np.random.randint(255, size=(100, 3), dtype=int)

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    q_cam = device.getOutputQueue(name="cam", maxSize=4, blocking=False)#####原图像
    q_manip = device.getOutputQueue(name="manip", maxSize=4, blocking=False)#####更改后的图像

    in_nn = device.getInputQueue(name="nnInput", maxSize=4, blocking=False)  ##神经网络输出
    out_nn = device.getOutputQueue(name="nnoutput", maxSize=4, blocking=False)##神经网络输出

    start_time = time.time()
    counter = 0
    fps = 0
    layer_info_printed = False

    while True:
        in_cam = q_cam.get()

        in_manip = q_manip.get()

        frame = in_cam.getCvFrame()
        frame_manip = in_manip.getCvFrame()
        dai_frame = dai.ImgFrame()
        dai_frame.setHeight(192)
        dai_frame.setWidth(192)


        # frame_manip=cv2.cvtColor(frame_manip, cv2.COLOR_BGR2BGR)
        cv2.imwrite("new_img.jpg",frame_manip)
        img = 'new_img.jpg'
        image = Image.open(img)
        image=np.array(image).transpose(2,0,1)

        # image1=cv2.cvtColor(frame_manip, cv2.COLOR_BGR2RGB)
        # image = np.array(image1).transpose(2, 0, 1)
        # image=list(image)

        dai_frame.setData(image)
        in_nn.send(dai_frame)

        out_nn1 = out_nn.get()
        out1=np.array(out_nn1.getLayerFp16('output3')).reshape(1,75,24,24)
        out2= np.array(out_nn1.getLayerFp16('output2')).reshape(1,75,12,12)
        out3 = np.array(out_nn1.getLayerFp16('output1')).reshape(1,75,6,6)
        out1=torch.from_numpy((out1))
        out2 = torch.from_numpy((out2))
        out3 = torch.from_numpy((out3))
        out=(out3,out2,out1)

        yolo = YOLO()
        newimage = Image.open(img)
        r_image = yolo.detect_image(newimage, out, crop=False, count=False)
        # r_image.show()

        r_image=np.array(r_image)
        r_image = cv2.cvtColor(r_image, cv2.COLOR_RGB2BGR)
        cv2.namedWindow('result', cv2.WINDOW_NORMAL)
        cv2.imshow('result',np.array(r_image))
# # ############################    后处理    ############################
# #         anchors=np.array([[ 10.,  13.],[ 16.,  30.],[ 33.,  23.],
# #                           [ 30.,  61.],[ 62.,  45.],[ 59., 119.],
# #                           [116.,  90.],[156., 198.],[373., 326.]])
# #         num_classes=80
# #         input_shape=(640,640)
# #         anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]]
# #         bbox_util = DecodeBox(anchors, num_classes, (input_shape[0], input_shape[1]), anchors_mask)
# #         output=bbox_util.decode_box(out)
#
#         # ---------------------------------------------------#
#
#         # cv2.imwrite("ram_img.jpg",frame)
#         # cv2.imwrite("new_img.jpg",frame_manip)
#

        counter += 1
        if (time.time() - start_time) > 1:
            fps = counter / (time.time() - start_time)
            print(11,fps)

            counter = 0
            start_time = time.time()


        if cv2.waitKey(1) == ord('q'):
            break