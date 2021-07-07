# Docker and Vitis AI installation
The Vitis™ AI software is [made available through Docker Hub](https://www.xilinx.com/html_docs/vitis_ai/1_3/installation.html#ariaid-title2 "Downloading the Vitis AI Development Kit"). The tools container contains the Vitis AI quantizer, AI compiler, and AI runtime for cloud DPU.

## Requirements
- As the TensorFlow library was compiled to use [AVX512F instructions](https://forums.xilinx.com/t5/AI-and-Vitis-AI/Quantizer-Illegal-Instruction-Core-Dumped/td-p/1124112 "Quantizer Illegal Instruction (Core Dumped)"), check that the host machine supports AVX512 instruction. \
To do so, open a terminal and enter the following commands :
```
lscpu

grep avx2 /proc/cpuinfo
```
- We chose to use Ubuntu 18.04, which is an OS version compliant with the [Vitis AI requirements](https://www.xilinx.com/html_docs/vitis_ai/1_3/oqy1573106160510.html "Minimum System Requirements").
## Install Docker for Ubuntu 18.04
- Remove eventual old versions of Docker. It is OK if apt-get reports that none of the packages are installed.
```
sudo apt-get remove docker docker-engine docker.io containerd runc
```
- Install Docker for Ubuntu 18.04 [using the repository](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository "Install Docker ofr Ubuntu"). For this project, we are using the [version 20.10.6](https://docs.docker.com/engine/release-notes/#20106 "Docker 20.10.6")
  - Set up the repository
  ```
  sudo apt-get update
  
  sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release  
    
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
  
  echo \
    "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
  ```
  - Install Docker Engine
  ```
  sudo apt-get update
  
  sudo apt-get install docker-ce docker-ce-cli containerd.io
  ```
  - Verify that Docker Engine is installed correctly 
  ```
  sudo docker run hello-world
  ```
- Get the Linux user in the docker group to [manage Docker as a non-root user](https://docs.docker.com/engine/install/linux-postinstall/ "Post-installation steps for Linux")
  - Create a Unix group called docker and add users to it.
  ```
  sudo groupadd docker
  
  sudo usermod -aG docker $USER
  
  newgrp docker 
  ```
  - Check that the group has been correctly set
  ```
  groups $USER  
  
  members docker
  ```
  - Verify that you can run docker commands without sudo
  ```
  docker run hello-world
  ```
## Download the Vitis AI Docker image (for CPU)
- We can now download the Vitis AI Docker image. We are using the CPU version but the GPU one can also be used. We are loading by default the latest image version, which was [1.3.598](https://hub.docker.com/layers/xilinx/vitis-ai-cpu/1.3.598/images/sha256-cb502f96f071126f0efc90ee36df90cd0dba5b285891aca05c91dd0d91a74a09?context=explore "Vitis AI CPU") at the time.
```
docker pull xilinx/vitis-ai-cpu:latest  
```
## Download the Vitis AI repository
- Create a workspace for Vitis AI projects, for example *~/vitis_ai_ws*, and move to this path
```
mkdir ~/vitis_ai_ws
cd ~/vitis_ai_ws
```
- Clone the [Vitis AI repository](https://github.com/Xilinx/Vitis-AI "Vitis AI repository"). For this project, we are using the [version 1.3](https://github.com/Xilinx/Vitis-AI/releases/tag/v1.3.2 "Vitis AI 1.3.2")
```
git clone --recurse-submodules https://github.com/Xilinx/Vitis-AI  
```
- Move to the repository folder and run the pre-built Docker image for Vitis AI (with CPU)
```
cd Vitis-AI

./docker_run.sh xilinx/vitis-ai-cpu:latest
```

