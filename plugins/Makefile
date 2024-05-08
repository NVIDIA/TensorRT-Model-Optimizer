# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

include Makefile.config

GENCODE_SM89 = -gencode=arch=compute_89,code="sm_89"
GENCODE_SM90 = -gencode=arch=compute_90,code="sm_90"

OBJECTS = obj/FP8Conv2DPlugin.cpp.o
OBJECTS += obj/common/checkMacrosPlugin.cpp.o
OBJECTS += obj/common/cudaDriverWrapper.cpp.o
OBJECTS += obj/common/plugin.cpp.o
OBJECTS += obj/common/reducedMathPlugin.cpp.o

.PHONY: all
all:
	+ $(MAKE) dirs
	$(MAKE) bin/FP8Conv2DPlugin.so
	$(MAKE) bin/groupNormPlugin.so

dirs:
	if [ ! -d bin ]; then mkdir -p bin; fi
	if [ ! -d obj/generated ]; then mkdir -p obj/generated; fi
	if [ ! -d obj/common ]; then mkdir -p obj/common; fi
	if [ ! -d obj/groupNormPlugin ]; then mkdir -p obj/groupNormPlugin; fi

cubin:
	$(MAKE) dirs
	$(MAKE) $(CUBIN_CPP)

clean:
	rm -rf bin obj cubin

###################################################################################################
PREBUILT_KERNEL ?= ./prebuilt/

bin/FP8Conv2DPlugin.so: $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -shared -o $@ $^ -L$(CUDA)/lib64 -lcudart -L$(TRT_LIBPATH) -lnvinfer -L$(PREBUILT_KERNEL) -lfp8convkernel

	#$(CXX) $(CXX_FLAGS) -o $@ $^ -L$(CUDA)/lib64 -Wl,-rpath=$(CUDA)/lib64 -lcudart -L$(TRT_LIBPATH) -lnvinfer
###################################################################################################

obj/FP8Conv2DPlugin.cpp.o: ./FP8Conv2DPlugin/FP8Conv2DPlugin.cpp
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_FLAGS) -I./ $(GENCODE_SM89) $(GENCODE_SM90) -c -o $@ $<

obj/common/%.cpp.o: ./common/%.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) -I./ -c -o $@ $<

###################################################################################################
### groupnorm plugin
OBJECTS = obj/groupNormPlugin/groupNormPlugin.cpp.o
OBJECTS += obj/groupNormPlugin/groupNormPlugin.cu.o
OBJECTS += obj/common/checkMacrosPlugin.cpp.o
OBJECTS += obj/common/cudaDriverWrapper.cpp.o
OBJECTS += obj/common/plugin.cpp.o
OBJECTS += obj/common/reducedMathPlugin.cpp.o

bin/groupNormPlugin.so: $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -shared -o $@ $^ -L$(CUDA)/lib64 -lcudart -L$(TRT_LIBPATH) -lnvinfer

	#$(CXX) $(CXX_FLAGS) -o $@ $^ -L$(CUDA)/lib64 -Wl,-rpath=$(CUDA)/lib64 -lcudart -L$(TRT_LIBPATH) -lnvinfer
###################################################################################################

obj/groupNormPlugin/groupNormPlugin.cpp.o: ./groupNormPlugin/groupNormPlugin.cpp
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_FLAGS) -I./ $(GENCODE_SM89) $(GENCODE_SM90) -c -o $@ $<

obj/groupNormPlugin/groupNormPlugin.cu.o: ./groupNormPlugin/groupNormKernel.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_FLAGS) -I./ $(GENCODE_SM89) $(GENCODE_SM90) -c -o $@ $<
