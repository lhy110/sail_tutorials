# Installation and Environment Setup
By Pac Hung and Zac Todd

This tutorial will cover the setup of three different software enviroments we use at SAIL. They are python and tensorflow 1.X, python and tensorflow 2.X with tensorflow model garden, and Matlab.

## Prerequisite Software
### Windows
Download and install [Git](https://git-scm.com/download/win).

Download and install [Annaconda3](https://www.anaconda.com/distribution/).

Download and install [Microsoft Visual Studio](https://visualstudio.microsoft.com/downloads/).

Install [Visual Studio Build Tools 2019 and Visual C++ build tools](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16) (link downloads the .exe).

### Ubuntu
Install Git and build essentials.
```commandline
sudo apt-get update
sudo apt install git build-essential
```

Download and install [Annaconda3](https://www.anaconda.com/distribution/). Or via command line with:

```bash
sudo apt-get update
sudo apt-get install curl

# Input the version you want to install e.g. 2020.2 which enables python 3.7
curl –O https://repo.anaconda.com/archive/Anaconda3-<version>-Linux-x86_64.sh

sha256sum Anaconda3–<version>–Linux–x86_64.sh

bash Anaconda3–<version>–Linux–x86_64.sh


## Python with Tensorflow 1.X
```bash
git clone https://github.com/lhy/sail_tutorials.git
cd sail_tutorials
cd 1.\ Installation\ and\ Environment\ Setup/


conda create --name SAIL python=3.6
conda activate SAIL

# Conda install if using a GPU, for CUDA and cuDNN.
conda install -c anaconda tensorflow-gpu


# Python Dependencies
pip install -r python_tf1_requirements.txt

# COCO dependency
# Windows
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

# Linux
pip install git+https://github.com/waleedka/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI

pip install git+https://github.com/matterport/Mask_RCNN.git
````

## Python and Tensorflow 2.x with Tensorflow Model Garden
Creating environment and installing tensorflow.
```bash
conda create --name python_tf2
conda activate python_tf2
pip install tensorflow

conda install protobuf

# Dependencies to be installed if using a GPU. This installs CUDA and cuDNN.
conda install cudatoolkit=10.1 cudnn

# Then restart the enviorment
conda deactivate
conda activate python_tf2
```

Verify tensorflow using:
```bash
python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

A result similar to the following will be printed out.
```
2020-08-05 14:15:28.713790: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
2020-08-05 14:15:29.388029: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcuda.so.1
2020-08-05 14:15:29.402977: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-08-05 14:15:29.403254: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 0 with properties:
pciBusID: 0000:01:00.0 name: GeForce RTX 2080 Ti computeCapability: 7.5
coreClock: 1.545GHz coreCount: 68 deviceMemorySize: 10.76GiB deviceMemoryBandwidth: 573.69GiB/s
2020-08-05 14:15:29.403269: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
2020-08-05 14:15:29.404100: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
2020-08-05 14:15:29.404999: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcufft.so.10
2020-08-05 14:15:29.405124: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcurand.so.10
2020-08-05 14:15:29.405970: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusolver.so.10
2020-08-05 14:15:29.406445: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusparse.so.10
2020-08-05 14:15:29.408268: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
2020-08-05 14:15:29.408328: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-08-05 14:15:29.408655: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-08-05 14:15:29.408898: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1858] Adding visible gpu devices: 0
2020-08-05 14:15:29.409083: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2020-08-05 14:15:29.432774: I tensorflow/core/platform/profile_utils/cpu_utils.cc:104] CPU Frequency: 3600000000 Hz
2020-08-05 14:15:29.433142: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x556a887caa30 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-05 14:15:29.433155: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2020-08-05 14:15:29.569641: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-08-05 14:15:29.569970: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x556a8acc2db0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2020-08-05 14:15:29.569984: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): GeForce RTX 2080 Ti, Compute Capability 7.5
2020-08-05 14:15:29.570089: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-08-05 14:15:29.570327: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 0 with properties:
pciBusID: 0000:01:00.0 name: GeForce RTX 2080 Ti computeCapability: 7.5
coreClock: 1.545GHz coreCount: 68 deviceMemorySize: 10.76GiB deviceMemoryBandwidth: 573.69GiB/s
2020-08-05 14:15:29.570344: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
2020-08-05 14:15:29.570357: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
2020-08-05 14:15:29.570365: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcufft.so.10
2020-08-05 14:15:29.570373: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcurand.so.10
2020-08-05 14:15:29.570381: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusolver.so.10
2020-08-05 14:15:29.570389: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusparse.so.10
2020-08-05 14:15:29.570397: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
2020-08-05 14:15:29.570424: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-08-05 14:15:29.570665: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-08-05 14:15:29.570886: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1858] Adding visible gpu devices: 0
2020-08-05 14:15:29.570902: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
2020-08-05 14:15:29.845639: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1257] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-08-05 14:15:29.845665: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1263]      0
2020-08-05 14:15:29.845670: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1276] 0:   N
2020-08-05 14:15:29.845788: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-08-05 14:15:29.846060: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-08-05 14:15:29.846299: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1402] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 9952 MB memory) -> physical GPU (device: 0, name: GeForce RTX 2080 Ti, pci bus id: 0000:01:00.0, compute capability: 7.5)
tf.Tensor(-614.579, shape=(), dtype=float32)
```

Installing the object detection dependencies.
```bash
git clone https://github.com/tensorflow/models.git

cd models/research

# Compile protos.

protoc object_detection/protos/*.proto --python_out=.

# COCO dependency
# Windows
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

# Linux
pip install git+https://github.com/waleedka/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI

# Install TensorFlow Object Detection API.
cp object_detection/packages/tf2/setup.py .
python -m pip install .
```

Test Object Detection dependencies have been installed using:
```bash
python object_detection/builders/model_builder_tf2_test.py
```

A result similar to the following will be printed out.
```
...
INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_create_ssd_models_from_config): 15.98s
I0805 14:16:54.646245 140250536642368 test_util.py:1972] time(__main__.ModelBuilderTF2Test.test_create_ssd_models_from_config): 15.98s
[       OK ] ModelBuilderTF2Test.test_create_ssd_models_from_config
[ RUN      ] ModelBuilderTF2Test.test_invalid_faster_rcnn_batchnorm_update
INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_invalid_faster_rcnn_batchnorm_update): 0.0s
I0805 14:16:54.653034 140250536642368 test_util.py:1972] time(__main__.ModelBuilderTF2Test.test_invalid_faster_rcnn_batchnorm_update): 0.0s
[       OK ] ModelBuilderTF2Test.test_invalid_faster_rcnn_batchnorm_update
[ RUN      ] ModelBuilderTF2Test.test_invalid_first_stage_nms_iou_threshold
INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_invalid_first_stage_nms_iou_threshold): 0.0s
I0805 14:16:54.654255 140250536642368 test_util.py:1972] time(__main__.ModelBuilderTF2Test.test_invalid_first_stage_nms_iou_threshold): 0.0s
[       OK ] ModelBuilderTF2Test.test_invalid_first_stage_nms_iou_threshold
[ RUN      ] ModelBuilderTF2Test.test_invalid_model_config_proto
INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_invalid_model_config_proto): 0.0s
I0805 14:16:54.654542 140250536642368 test_util.py:1972] time(__main__.ModelBuilderTF2Test.test_invalid_model_config_proto): 0.0s
[       OK ] ModelBuilderTF2Test.test_invalid_model_config_proto
[ RUN      ] ModelBuilderTF2Test.test_invalid_second_stage_batch_size
INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_invalid_second_stage_batch_size): 0.0s
I0805 14:16:54.655431 140250536642368 test_util.py:1972] time(__main__.ModelBuilderTF2Test.test_invalid_second_stage_batch_size): 0.0s
[       OK ] ModelBuilderTF2Test.test_invalid_second_stage_batch_size
[ RUN      ] ModelBuilderTF2Test.test_session
[  SKIPPED ] ModelBuilderTF2Test.test_session
[ RUN      ] ModelBuilderTF2Test.test_unknown_faster_rcnn_feature_extractor
INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_unknown_faster_rcnn_feature_extractor): 0.0s
I0805 14:16:54.656313 140250536642368 test_util.py:1972] time(__main__.ModelBuilderTF2Test.test_unknown_faster_rcnn_feature_extractor): 0.0s
[       OK ] ModelBuilderTF2Test.test_unknown_faster_rcnn_feature_extractor
[ RUN      ] ModelBuilderTF2Test.test_unknown_meta_architecture
INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_unknown_meta_architecture): 0.0s
I0805 14:16:54.656579 140250536642368 test_util.py:1972] time(__main__.ModelBuilderTF2Test.test_unknown_meta_architecture): 0.0s
[       OK ] ModelBuilderTF2Test.test_unknown_meta_architecture
[ RUN      ] ModelBuilderTF2Test.test_unknown_ssd_feature_extractor
INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_unknown_ssd_feature_extractor): 0.0s
I0805 14:16:54.657182 140250536642368 test_util.py:1972] time(__main__.ModelBuilderTF2Test.test_unknown_ssd_feature_extractor): 0.0s
[       OK ] ModelBuilderTF2Test.test_unknown_ssd_feature_extractor
----------------------------------------------------------------------
Ran 20 tests in 19.541s

OK (skipped=1)
```

```bash
# Then restart the enviorment
conda deactivate
conda activate python_tf2
```

Return to the tutorial directory
```
pip install -r python_tf2_requirements.txt
```

## Matlab
Matlab contains a lot of prebuild and tested computer vision, deep learning and sensor fusion tools, which maybe be uses as an inital tutorial in understaning how different methods work. They also provide many notebook style and video tutiorals found [here](https://au.mathworks.com/) and [here](https://www.youtube.com/user/MATLAB) respectivly.

Instructions for installing matlab using the University of Canterbury accoutn are located [here](https://www.canterbury.ac.nz/engineering/schools/mathematics-statistics/student-advice-and-resources/matlab-on-personal-devices/).
