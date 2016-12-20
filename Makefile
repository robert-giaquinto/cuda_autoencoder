################################################################################
#
# Build script for project
#
################################################################################

# Add source files here
EXECUTABLE	:= dA_naive
# Cuda source files (compiled with cudacc)
CUFILES		:= dA_naive.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= dA_gold_naive.c

################################################################################
# Rules and targets


include ../../common/common.mk