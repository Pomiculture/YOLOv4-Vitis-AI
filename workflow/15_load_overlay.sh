#!/bin/bash

###############################################################################################################

# Load the overlay for Alveo U280.

###############################################################################################################

if [ ! -d "${DPU_ARCHIVE}" ] 
	then
    		echo "Directory ${DPU_ARCHIVE} does not exist."
		echo "-----------------------------------------"
		echo "DOWNLOADING ALVEO ${ALVEO_MODEL} OVERLAY FILES"
		echo "-----------------------------------------"

		# Download Alveo xclbin files overlays
		wget --no-clobber https://www.xilinx.com/bin/public/openDownload?filename=${DPU_ARCHIVE}.tar.gz -O ${DPU_ARCHIVE}.tar.gz

		# Extract content from archive
		tar xfz ${DPU_ARCHIVE}.tar.gz

		# Remove archive folder
		rm alveo_xclbin-1.3.0.tar.gz   

		echo "-----------------------------------------"
		echo "DOWNLOAD AND EXTRACTION COMPLETE"
		echo "-----------------------------------------" 
	else 
		echo "Directory ${DPU_ARCHIVE} already exists."
fi

###############################################################################################################

echo "-----------------------------------------"
echo "COPYING OVERLAY FILE (DPU ${DPU_FREQ^^}) TO '/usr/lib' PATH"
echo "-----------------------------------------"

# Copy DPU overlay to /usr/lib path
sudo cp ${DPU_ARCHIVE}/${ALVEO_MODEL}/${DPU_FREQ}/dpu.xclbin /usr/lib
sudo cp ${DPU_ARCHIVE}/${ALVEO_MODEL}/${DPU_FREQ}/hbm_address_assignment.txt /usr/lib

echo "-----------------------------------------"
echo "COPY COMPLETE"
echo "-----------------------------------------"  
