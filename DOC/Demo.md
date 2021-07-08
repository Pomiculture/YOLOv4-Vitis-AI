# Demo Guide

In order to demonstrate the interest of the [Vitis AI Model Zoo](https://github.com/Xilinx/Vitis-AI/tree/master/models/AI-Model-Zoo "Model Zoo"), this example runs a pre-trained YOLOv4 model over the [COCO dataset](https://cocodataset.org/#home "COCO dataset"). The Vitis AI Model Zoo offers a [list of ready to use deep learning models from popular machine learning frameworks](https://github.com/Xilinx/Vitis-AI/tree/master/models/AI-Model-Zoo/model-list "Model Zoo list") for fast deployment on Xilinx hardware platforms.

To run the demo, just run the following script after launching the Vitis AI Docker image. As this model was trained with the COCO dataset, it can detect a [large range of objects](https://github.com/AlexeyAB/darknet/blob/master/data/coco.names "COCO class list"), but on the other hand it can only detected apples without differentiating the clean to the damaged ones. Thus, it is better to replace the dataset content of test samples, */data/test_images/*, by another set of images, like the [COCO test set](https://cocodataset.org/#download "COCO dataset download").

```
source ./run_demo.sh
```

We are now going to briefly describe the few steps required to run this demo.

### Table of contents
1) [Compile the App](#compile)
2) [Load the model from Model Zoo](#zoo)
3) [Deploy the model](#deploy)
4) [Load the overlay](#overlay)
5) [Run the App](#inference)

---
<div id='compile'/>

## 1) Compile the App
Compile the C++ application code that will be used during inference by running the following script. 
```
source ./workflow/11_compile_app.sh
```
---
<div id='zoo'/>

## 2) Load the model from Model Zoo
Load the pre-compiled YOLOv4 model from Model Zoo. The below script downloads the It will download the archive of the [pre-trained YOLOv4 model]( https://github.com/Xilinx/Vitis-AI/blob/master/models/AI-Model-Zoo/model-list/dk_yolov4_coco_416_416_60.1G_1.3/model.yaml "YOLOv4 COCO"), more precisely the *yolov4_leaky_spp_m-u50lv-u280* model, that fits our accelerator card. 
```
source ./workflow/demo/0_load_yolov4_demo_model.sh
```

---
<div id='deploy'/>

## 3) Deploy the model
The next step is to deploy the model to the */usr/share/vitis_ai_library/models* in which the Vitis AI library will look during inference.
```
source ./workflow/demo/1_deploy_demo_model.sh
```

---
<div id='overlay'/>

## 4) Load the overlay
We have to load the Alveo U280 overlay to configure the DPU (Deep-learning Processor Unit). We place the files to the */usr/lib* folder.
```
source ./workflow/15_load_overlay.sh
```

---
<div id='inference'/>

## 5) Run the App
Finally, we just have to run our custom application code. The output images can be found in the folder */output/0_demo/images*.
```
source ./workflow/demo/2_run_demo_model.sh
```

