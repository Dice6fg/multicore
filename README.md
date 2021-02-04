# Multicore Project

## 프로젝트의 목적

  미리 학습된 CNN 모델 VGG16을 통한 3000장의 32*32 image의 CIFAR10 데이터 셋 이미지 분류 예측의 순차적인 코드를 
  OpenCL을 통해 병렬처리가 가능한 프로그램을 작성하여 최대한 가속한다.
  

## 하드웨어 환경

![alt text](https://github.com/Dice6fg/multicore/blob/master/about/device.JPG)

## VGG16 모델의 구조

![alt text](https://github.com/Dice6fg/multicore/blob/master/about/vgg16.JPG)

## 수행 방법

- 순차적 수행 코드와 다르게 같은 필터를 사용하는 3000장의 이미지를 한번에 병렬처리 하도록 구성함.

![alt text](https://github.com/Dice6fg/multicore/blob/master/about/input%20order.JPG)

- 주어진 GPU는 256개의 MAX_WORK_GROUP_SIZE를 보유하고 있으므로 32*32의 이미지를 16*16의 4개의 부분으로 나누어 convolution을 진행한다.

{16, 16, 1}

![alt text](https://github.com/Dice6fg/multicore/blob/master/about/32.JPG)

- pooling을 통하여 16*16로 줄어든 이미지는 그대로 WORK_GROUP에 배정하여 convolution을 진행한다.

{16, 16, 1}

![alt text](https://github.com/Dice6fg/multicore/blob/master/about/16.JPG)

- 8*8이미지를 4개의 채널을 통해 convolution을 진행한다.

{8, 8, 4}

![alt text](https://github.com/Dice6fg/multicore/blob/master/about/8.JPG)

- 4*4의 이미지는 16개의 채널을 통하여 work group을 최대한 활용하여야 하나 3000장의 이미지가 채널 수로 나누어 떨어지지 않으므로 15개의 채널을 활용하였다.

{4, 4, 15}

![alt text](https://github.com/Dice6fg/multicore/blob/master/about/4.JPG)

- 2*2의 이미지 또한 같은 이유로 64개의 채널이 아닌 60개의 채널을 활용하였다.

{2, 2, 60}

![alt text](https://github.com/Dice6fg/multicore/blob/master/about/2.JPG)

- fc 레이어에서 또한 같은 이유로 256개의 채널이 아닌 250개의 채널을 활용하였다.

{1, 250}

![alt text](https://github.com/Dice6fg/multicore/blob/master/about/fc.JPG)

- loop invariant 변수를 찾고 loop unrolling을 진행하여 최적화를 진행하였다.

## 결과

- i5-7500으로 진행한 sequential processing code: 약 900초

- GPU를 이용한 parallel processing code: 15.466초

![alt text](https://github.com/Dice6fg/multicore/blob/master/about/result.JPG)
