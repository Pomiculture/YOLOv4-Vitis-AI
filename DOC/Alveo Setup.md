# Setting Up the Host for Alveo accelerator card U280
Here, we explain how to setup the [Alveo U280 Data Center Accelerator Card](https://www.xilinx.com/products/boards-and-kits/alveo/u280.html "Xilinx Alveo U280").

*Sources* :
- [Vitis AI Alveo card setup](https://github.com/Xilinx/Vitis-AI/blob/master/setup/alveo/u50_u50lv_u280/README.md "Alveo card setup")
- [Vitis AI Quick start for Alveo](https://github.com/Xilinx/Vitis-AI/blob/master/tools/Vitis-AI-Library/README.md#quick-start-for-alveo "Quick start for Alveo")

## Prerequisites 
- Install Docker 
- Install Vitis AI Docker image
- Download Vitis AI workspace

## Setup Alveo Accelerator Card with HBM for DPUCAHX8H
Xilinx DPU IP family for convolution neural network (CNN) inference application supports Alveo accelerator cards with HBM, 
[including Alveo U280 card](https://www.xilinx.com/html_docs/vitis_ai/1_3/oqy1573106160510.html "Supported FPGA boards"). The Xilinx DPUs for Alveo-HBM card include DPUCAHX8H, optimized for high throughput. 
The on-premise DPUCAHX8H overlay is released along with Vitis AI.

### Alveo Card Setup
- Install Xilinx runtime (XRT) for Ubuntu 18.04 [here](https://www.xilinx.com/bin/public/openDownload?filename=xrt_202020.2.8.726_18.04-amd64-xrt.deb "xrt_202020.2.8.726_18.04-amd64-xrt.deb")
- Install the Alveo card gen3x4 target platform files for U280 card with Ubuntu 18.04 [here](https://www.xilinx.com/bin/public/openDownload?filename=xilinx-u280-xdma-201920.3-2789161_18.04.deb "xilinx-u280-xdma-201920.3-2789161_18.04.deb")
- Update the Alveo card flash
```
sudo /opt/xilinx/xrt/bin/xbmgmt flash --update --shell xilinx_u280_xdma_201920_3
```
- Cold reboot the host machine to finish the setup

## Download and install the *.xclbin file* (FPGA binary container file used by the host program)
### Run Vitis AI Docker image
- Move to the [Vitis AI repository workspace](https://github.com/Xilinx/Vitis-AI "Vitis AI") and run the Docker container loading the Vitis AI image
```
cd Vitis-AI 

./docker_run.sh xilinx/vitis-ai-cpu:latest
```
### Download the *.xclbin* files 
- Run following commands within Docker to download the overlay for the Alveo U280 card and export it to the */usr/lib* folder
``` 
cd /workspace

wget https://www.xilinx.com/bin/public/openDownload?filename=alveo_xclbin-1.3.1.tar.gz -O alveo_xclbin-1.3.1.tar.gz

tar -xzvf alveo_xclbin-1.3.1.tar.gz

cd alveo_xclbin-1.3.1/U280/14E300M

sudo cp dpu.xclbin hbm_address_assignment.txt /usr/lib
```
### Running a Vitis AI library example
The Vitis-AI-Libray examples are located in the */workspace/demo/Vitis-AI-Library/* folder in the docker system.
- Load a model from the [Model Zoo](https://github.com/Xilinx/Vitis-AI/tree/master/models/AI-Model-Zoo/model-list "Model Zoo") adapted to the target platform
A *.yaml* file describes all the details about the model. It contains the model's download links for different platforms. Choose the corresponding model and download it.
Take densebox_320_320 as an example.
```
wget https://www.xilinx.com/bin/public/openDownload?filename=densebox_320_320-u50lv-u280-r1.3.1.tar.gz -O densebox_320_320-u50lv-u280-r1.3.1.tar.gz
```	
- Install the model package to the path */usr/share/vitis_ai_library/models*
```
sudo mkdir /usr/share/vitis_ai_library/models

tar -xzvf densebox_320_320-u50lv-u280-r1.3.1.tar.gz

sudo cp densebox_320_320 /usr/share/vitis_ai_library/models -r
```	
- Download the vitis_ai_library_r1.3.x_images.tar.gz and vitis_ai_library_r1.3.x_video.tar.gz packages and untar them
```
cd /workspace

wget https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_library_r1.3.1_images.tar.gz -O vitis_ai_library_r1.3.1_images.tar.gz

wget https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_library_r1.3.0_video.tar.gz -O vitis_ai_library_r1.3.0_video.tar.gz

tar -xzvf vitis_ai_library_r1.3.1_images.tar.gz -C demo/Vitis-AI-Library/

tar -xzvf vitis_ai_library_r1.3.0_video.tar.gz -C demo/Vitis-AI-Library/
```
- Move to the directory of the sample project and then compile it. Take *facedetect* as an example.
```
cd /workspace/demo/Vitis-AI-Library/samples/facedetect

bash -x build.sh
```	
- Run the image test example
```
./test_jpeg_facedetect densebox_320_320 sample_facedetect.jpg
```
- Run the video test example
```
./test_video_facedetect densebox_320_320 video_input.mp4 -t 8
```
- Test the performance of model by running the following command
```
./test_performance_facedetect densebox_320_320 test_performance_facedetect.list -t 8 -s 60
```

