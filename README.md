# YOLOv4-Vitis-AI
Custom YOLOv4 for apple recognition (clean/damaged) on Alveo U280 accelerator card using Vitis AI framework.

### Table of contents
1) [Context](#context)
2) [Vitis AI](#vitis)
3) [YOLOv4](#yolo)
4) [Requirements](#requirements)
5) [User Guide](#guide)
6) [Results](#results)
7) [References](#references)

---
<div id='context'/>

## 1) Context
A deep-learning model is caracterized by two distinct computation-intensive processes that are training and inference. [During the training step, the model is taught to perform a specific task. On the other hand, inference is the deployment of the trained model to perform on new data](https://www.intel.com/content/www/us/en/artificial-intelligence/posts/deep-learning-training-and-inference.html "The Difference Between Deep Learning Training and Inference"). Real-time inference of deep neural network (DNN) models is a big challenge that the Industry faces, with the growth of latency constrained applications. For this reason, inference acceleration has become more critical than faster training. [While the training step is most often carried out by GPUs, due to their high throughput, massive parallelism, simple control flow, and energy efficiency](https://medium.com/syncedreview/deep-learning-in-real-time-inference-acceleration-and-continuous-training-17dac9438b0b "Deep Learning in Real Time — Inference Acceleration and Continuous Training"), FPGA devices (Field Programmable Gate Arrays) are more adapted to AI inference, by providing better performance per watt of power consumption than GPUs thanks to their flexible hardware configuration.

An important axis of research is the deployment of AI models on embedded platforms. To achieve that, along with smaller neural network architectures, some techniques like quantization and pruning allow to reduce the size of existing architectures without losing much accuracy. It minimizes the hardware footprint and energy consumption of the target board. These techniques perform well on FPGAs, over GPU.

One significant issue about conjugating AI inference with hardware acceleration is the expertise required in both domains, especially regarding low level development on accelerator cards. Fortunately, some frameworks make hardware more accessible to software engineers and data scientists. [With the Xilinx’s Vitis AI toolset, we can quite easily deploy models from Keras-TensorFlow straight onto FPGAs](https://beetlebox.org/vitis-ai-using-tensorflow-and-keras-tutorial-part-1/ "Vitis AI using TensorFlow and Keras Tutorial"). 

---
<div id='vitis'/>

## 2) Vitis AI
[Vitis™](https://www.xilinx.com/products/design-tools/vitis.html "Vitis") is a unified software platform for embedded software and accelerated applications development on Xilinx® hardware platforms, with [Edge, Cloud or Hybrid computing](https://www.redhat.com/en/topics/cloud-computing/cloud-vs-edge "Cloud vs Edge computing"). The application code can be developed using high-level programming languages such as C++ and Python.

[Vitis™ AI](https://www.xilinx.com/products/design-tools/vitis/vitis-ai.html "Vitis AI") is a development environment whose purpose is to accelerate AI inference. Thanks to optimized [IP cores](https://anysilicon.com/ip-intellectual-property-core-semiconductors/ "IP core") and tools, it allows to implement pre-compiled or custom AI models and provides libraries to accelerate the application by interacting with the processor unit of the target platform. With Vitis AI, the user can easily develop [Deep Learning](https://machinelearningmastery.com/what-is-deep-learning/ "Deep Learning") inference applications without having a strong FPGA knowledge.

![VART (Vitis AI Runtime) stack](IMAGES/Vart_stack.png)

We chose to use the Vitis AI TensorFlow framework. For more information on Vitis AI, please refer to the official [user guide](https://www.xilinx.com/html_docs/vitis_ai/1_3/ "User Guide").

![Vitis AI workflow](IMAGES/workflow.jpg)

In our case, the hardware platform is an [Alveo™ Data Center Accelerator Card](https://www.xilinx.com/products/boards-and-kits/alveo.html "Alveo"). This [FPGA (Field Programmable Gate Arrays)](https://www.xilinx.com/products/silicon-devices/fpga/what-is-an-fpga.html "Xilinx FPGA") is a Cloud device to accelerate the computing workloads of deep learning inference algorithms. Its processor unit is called a  [Deep-Learning Processor Unit (DPU)](https://www.xilinx.com/html_docs/vitis_ai/1_3/tools_overview.html#nwc1570695738475 "DPU"), a a group of parameterizable IP cores pre-implemented on the hardware optimized for deep neural networks, compatible with the Vitis AI specialized instruction set. Different versions exists so as to offer different levels of throughput, latency, scalability, and power. The [Alveo U280 Data Center Accelerator Card](https://www.xilinx.com/products/boards-and-kits/alveo/u280.html "Alveo U280") supports the [Xilinx DPUCAHX8H DPU](https://www.xilinx.com/html_docs/vitis_ai/1_3/tools_overview.html#xyt1583919665886 "DPUCAHX8H") optimized for high throughput applications involving [Convolutional Neural Networks (CNNs)](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53 "Convolutional Neural Networks"). It is composed of a high performance scheduler module, a hybrid computing array module, an instruction fetch unit module, and a global memory pool module.

![DPUCAHX8H Top-Level Block Diagram](IMAGES/DPUCAHX8H.png)

---
<div id='yolo'/>

## 3) YOLO
................................

---
<div id='requirements'/>

## 4) Requirements
Before running the project, check the [requirements from Vitis AI](https://www.xilinx.com/html_docs/vitis_ai/1_3/oqy1573106160510.html "Minimum system requirements") and make sure to complete the following steps :
- **[Install the Vitis AI Docker image](DOC/Docker_&_Vitis_AI.md "Install Vitis AI Docker")**
- **[Set up the Alveo U280 accelerator card](DOC/Alveo_Setup.md "Alveo U280 setup")**

**Versions** :
- Docker : 20.10.6
- Docker Vitis AI image : 1.3.598   
- Vitis AI : 1.3.2      
- TensorFlow : 1.15.2
- Python : 3.6.12
- Anaconda : 4.9.2

**Hardware** :
- [Alveo U280 Data Center Accelerator Card](https://www.xilinx.com/products/boards-and-kits/alveo/u280.html "Alveo U280")

---
<div id='guide'/>

## 5) User Guide
In this section, we are going to explain how to run the project. \
This project is executed through a succession of bash files, located in the */workflow/* folder. \
You may need to first set the permissions for the bash files :
```
cd ./docker_ws/workflow/
chmod +x *.sh
cd ..
chmod +x *.sh
```
You can either run the scripts from the */workflow/* folder step by step, or run the two main scripts. \
The first script to run serves to open the Vitis AI image in the Docker container. \
Indeed, we can use the Vitis™ AI software through [Docker Hub](https://www.docker.com/ "Docker"). It contains the tools such as the Vitis AI quantizer, AI compiler, and AI runtime for cloud DPU. We chose to use the Vitis AI Docker image for host [CPU](https://blogs.nvidia.com/blog/2009/12/16/whats-the-difference-between-a-cpu-and-a-gpu/ "CPU vs GPU").
```
cd docker_ws
 source ./workflow/0_run_docker_cpu.sh
```

![Vitis AI workflow](IMAGES/docker.png)

Then, run the following script to execute the whole process.
```
 source ./run_all.sh
```
This project is based on the workflow from Vitis AI tutorials using the [Anaconda](https://www.anaconda.com/ "Anaconda") environment for [TensorFlow](https://www.tensorflow.org/?hl=en "TensorFlow"), such as the [MNIST Classification using Vitis AI and TensorFlow 1.15](https://github.com/Xilinx/Vitis-Tutorials/tree/master/Machine_Learning/Design_Tutorials/02-MNIST_classification_tf "MNIST Classification using Vitis AI and TensorFlow 1.15").

For more details, please consult this **[guide](DOC/Documentation.md "Documentation")**.




........................................

Open a terminal and make sur to be located in the workspace directory.

blabla 5 et sous parties pour summary



### 5.1 Demo
See this **[guide](DOC/Demo.md "Documentation")**.
```
source ./run_demo.sh
```
We used these model and dataset to quickly test our application code before deploying our own model.

### 5.2 Application
...

---
<div id='results'/>

## 6) Results
Here are some results after running the model on the FPGA :

![Vitis AI workflow](IMAGES/res_1.jpg) ![Vitis AI workflow](IMAGES/res_2.jpg) ![Vitis AI workflow](IMAGES/res_3.jpg) 

![Vitis AI workflow](IMAGES/res_4.jpg) ![Vitis AI workflow](IMAGES/res_5.jpg) ![Vitis AI workflow](IMAGES/res_6.jpg) 

![Vitis AI workflow](IMAGES/res_7.jpg) ![Vitis AI workflow](IMAGES/res_8.jpg)

Let's evaluate the [_mAP_ score](https://www.kdnuggets.com/2020/08/metrics-evaluate-deep-learning-object-detectors.html "Metrics to Use to Evaluate Deep Learning Object Detectors") of the model running on the accelerator card. We set the confidence trheshold to 0.6 and the IoU threshold to 0.5.

| Model              | Original   | Intermediate graph  | App (on Alveo U280)         | 
| :---:              |   :---:    |  :---:              |  :---:                      | 
| mAP @ IOU50 score  |    75.0 %  |  x                  |  91.0 % on the training set |
| FPS                |    x       |  x                  |  12                         |

---
<div id='references'/>

## 7) References
The mentionned projects below were used for this project as tools or source of inspiration :
- [YOLO: Real-Time Object Detection by pjreddie](https://pjreddie.com/darknet/yolo/ "YOLO: Real-Time Object Detection")
- [YOLO Darknet by AlexeyAB](https://github.com/AlexeyAB/darknet "Darknet")
- [TF Keras YOLOv4/v3/v2 Modelset by David8862](https://github.com/david8862/keras-YOLOv3-model-set "david8862/keras-YOLOv3-model-set")
- [Xilinx - Vitis AI Tutorials](https://github.com/Xilinx/Vitis-Tutorials/tree/master/Machine_Learning "Vitis AI tutorials")

---

original : https://medium.com/analytics-vidhya/train-a-custom-yolov4-object-detector-using-google-colab-61a659d4868
change according to "Create your custom config file and upload it to your drive" et Create your obj.data and obj.names files and upload to your drive 

axe amelio :
create new test set car biaisé (mais faute de temps)
ameliorer fps
eval fs des autres
modify alexey app to measure execution time of inference
evalgraph : normalize data to be able to evaluate its score, of the tf graph
better annotation and more images
+ draw boxes when runnign graph
amelio run software get inference duraiton and mAP score
axe amelio = améliroer l’affichage des labels parfois coupés si en bordure ou peu lisible si chevauchemetn avec autres

L’application produit un fichier du même format pour les confronter et obtenir le score.

https://towardsdatascience.com/map-mean-average-precision-might-confuse-you-5956f1bfa9e2

link thomas scrapper apples
https://github.com/Menchit-ai/parse-google-image

dire comment mettre son custom cfg (et attentions modifs) : modif dossier model (cfg et weights) + setenv dire quoi + dataset dire où

dire qupoi changer pour propre truc

attention software not same output https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects ?

amelio : run profiler + eval scroe graph not normalized

link to mAP score  with iou indication
+ link yolov4 tuto vitis ai
+ + autres liens utiles

fps could be better (https://www.xilinx.com/html_docs/vitis_ai/1_3/him1591152509554.html)

ANNOTATIONS : https://github.com/developer0hye/Yolo_Label explqieur valeurs retournées

parler des modifs du cfg * 2
Changed MaxPool size & Mish to Leaky ReLu

https://drive.google.com/drive/folders/1EnpvadEDiTrAUQzlouOIwPMlPIdV6Pfh?usp=sharing

 objBox.box.setX(leftX); // translation: midX -> leftX
                objBox.box.setY(topY); // translation: midY -> topY
                objBox.box.setWidth(width);
                objBox.box.setHeight(height);
                => CHANGER output file et eval_script, ou alors adapter label file

https://towardsdatascience.com/image-data-labelling-and-annotation-everything-you-need-to-know-86ede6c684b1
https://machinethink.net/blog/object-detection-with-yolo/
https://medium.com/@vinay.dec26/yat-an-open-source-data-annotation-tool-for-yolo-8bb75bce1767

est-ce que annotations côtés gauhe/drote bas/hat vs centrex, centry, widfth, height
ou pt départ et déplacement selon x et selon y ?  => modifier output file et script d'éval
todo = tracer coords dans une image

https://www.kdnuggets.com/2020/08/metrics-evaluate-deep-learning-object-detectors.html
https://towardsdatascience.com/evaluating-performance-of-an-object-detection-model-137a349c517b
https://blog.paperspace.com/mean-average-precision/

---
add a picture
+ def Yolo + histoire + Darknet 
+ TODO : dans intro expliquer les changements apportés au cfg (dire pk) et réintrainté derrière pour obtenir weights
+ TODO : dans intro faire references (Alexey et lien vers notebook et demander Thomas quelles modifs apportées pour pommes etc.)
+ TODO : def anchors + bounding box + obj detect + NMS & IoU
+ pommes saines/affectées qui sert pour classificateur de pommes défectueuses
+ axes amélio (score mAP faible, compile input shape écrite en dur?)
+ Vitis Library (offers YOLOv3 ... avec lien)
+ dire que propres pommes + lien vers Git Thomas scraper
+ annotations plateforme lien et screens
+ unzip dataset + model weights google drive link all downloadable and put in path
https://becominghuman.ai/explaining-yolov4-a-one-stage-detector-cdac0826cbd7
https://blog.roboflow.com/a-thorough-breakdown-of-yolov4/
+ blabla notebook original et alexey pour le Darknet
+ dire quoi modifier pour sa propre app (dataset + setenv dataset adn shapes + node names after converting to TF, etc. + DPU arch pour compile)
+ Constitution du dataset par requêtes Internet des variétés de pommes via l'API GoogleSearch dont les noms sont indiqués dans un fichier txt. ?

axes amelio : taille labels + test cuda img cpu
