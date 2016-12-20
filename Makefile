################################################################################
#
# Build script for project
#
################################################################################
# Please use whatever option needed to be run - naive or final parallel algo
# #############################################################
# OPTION 1 - NAIVE - uncomment executable, cufiles and cc files
# #######
# Add source files here
##EXECUTABLE	:= dA_naive
# Cuda source files (compiled with cudacc)
##CUFILES		:= dA_naive.cu
# C/C++ source files (compiled with gcc / c++)
##CCFILES		:= dA_gold_naive.c
#
# #############################################################
# OPTION 2 - FINAL - uncomment executable, cufiles and cc files
# #######
# Add source files here
EXECUTABLE	:= dA
# Cuda source files (compiled with cudacc)
CUFILES		:= dA.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= dA_gold.c

################################################################################
# Rules and targets


include ../../common/common.mk
