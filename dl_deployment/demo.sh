#!/bin/bash
set -ex

BUILD=./build-aarch64-linux-gnu
INSTALL_BIN=$BUILD/install/bin
MODELS=./models
IMAGES=./images

export LD_LIBRARY_PATH=$BUILD/install/lib:$LD_LIBRARY_PATH

echo "分类任务"
$INSTALL_BIN/tm_classification -m $MODELS/mobilenet.tmfile -i $IMAGES/cat.jpg -g 224,224 -s 0.017,0.017,0.017 -w 104.007,116.669,122.679

echo "人脸关键点检测任务"
$INSTALL_BIN/tm_landmark -m $MODELS/landmark.tmfile -i $IMAGES/mobileface02.jpg -r 1 -t 1

echo "人脸检测任务"
$INSTALL_BIN/tm_retinaface -m $MODELS/retinaface.tmfile -i $IMAGES/mtcnn_face4.jpg -r 1 -t 1

echo "目标检测任务"
$INSTALL_BIN/tm_yolofastest -m $MODELS/yolo-fastest-1.1.tmfile -i $IMAGES/ssd_dog.jpg -r 1 -t 1
$INSTALL_BIN/tm_efficientdet -m $MODELS/efficientdet.tmfile -i $IMAGES/ssd_horse.jpg -r 1 -t 1
