################################################################################
#
# Build script for project
#
################################################################################

# Add source files here
EXECUTABLE	:= dA
# Cuda source files (compiled with cudacc)
CUFILES		:= dA.cu
# C/C++ source files (compiled with gcc / c++)
#CCFILES		:= dA.c

################################################################################
# Rules and targets


include ../../common/common.mk