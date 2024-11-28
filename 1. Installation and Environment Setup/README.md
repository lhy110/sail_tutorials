# Installation and Environment Setup
By Zac Todd, Meng Zhang and Thomas Li

This tutorial will cover the setup of four different software enviroments we use at SAIL. They are python and tensorflow 1.X, python and tensorflow 2.X with tensorflow model garden, and Matlab.

## Prerequisite Software
### Windows
Note when install select options that add commands to the path to make it easyer to uses the dependencies.

Download and install [Git](https://git-scm.com/download/win).

Download and install [Annaconda3](https://www.anaconda.com/distribution/).

Install [Visual Studio Build Tools 2019 and Visual C++ build tools](https://aka.ms/vs/17/release/vc_redist.x64.exe) (link downloads the .exe).

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
```

## PyTorch with OpenMMLab's models (GPU)
[OpenMMLab](https://github.com/open-mmlab) has models for classification, detection and segemantion, in their [mmclassification](https://github.com/open-mmlab/mmclassification), [mmdetection](https://github.com/open-mmlab/mmdetection), and [mmsegmentation](https://github.com/open-mmlab/mmsegmentation).

Please follow the instrution for installing [mmdetection](https://github.com/open-mmlab/mmdetection).

In addition to the repo instrutions it will also be useful to install jupyter notebooks.
```bash
conda activate openmmlab
pip install jupyterlab
```

And then install the classification and segmenation libraies.
```bash
mim install mmcls mmseg
```

## Tensorflow with Tensorflow Model Garden
Creating environment and installing tensorflow.
```bash
conda create --name tf2_model_garden python=3.8 protobuf
conda activate tf2_model_garden

# Dependencies to be installed if using a GPU. This installs CUDA and cuDNN.
conda install cudatoolkit=10.1 cudnn

# Then restart the enviorment
conda deactivate
conda activate tf2_model_garden
```


Installing the object detection dependencies. Places this in a direcroty so that it can be easily acces in the future.
```bash
# Clone Tensorflow model garden
git clone https://github.com/tensorflow/models.git
cd models/research

# Compile protos.
protoc object_detection/protos/*.proto --python_out=.

# Install TensorFlow Object Detection API.
# Winodws
copy object_detection\\packages\\tf2\\setup.py .

# Linux
cp object_detection/packages/tf2/setup.py .


python -m pip install .
```

Verify tensorflow using:
```bash
python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
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


## Matlab
Matlab contains a lot of prebuild and tested computer vision, deep learning and sensor fusion tools, which maybe be uses as an inital tutorial in understaning how different methods work. They also provide many notebook (mlx) and video tutiorals found [here](https://au.mathworks.com/) and [here](https://www.youtube.com/user/MATLAB) respectivly.

Instructions for installing matlab using the University of Canterbury accoutn are located [here](https://www.canterbury.ac.nz/engineering/schools/mathematics-statistics/student-advice-and-resources/matlab-on-personal-devices/).
