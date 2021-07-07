#!/bin/bash

###############################################################################################################

# Profile the execution on the Alveo U280 board.

###############################################################################################################

# Parameters 
export PROFILING_DIR=workflow/profiling 

###############################################################################################################

echo "-----------------------------------------"
echo "PROFILING APP ON ALVEO ${ALVEO_MODEL}"
echo "-----------------------------------------" 

# Run app with profiling using vaitrace configuration file (and generate trace data for Vitis Analyzer tool)
sudo vaitrace -c ./${PROFILING_DIR}/trace_cfg.json --va

echo "-----------------------------------------"
echo "RUN COMPLETE"
echo "-----------------------------------------" 
