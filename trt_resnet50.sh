#!/bin/bash

if [ -z "$1" ];
then
	echo "$0 <model path>"
	exit 1
fi
trtexec --onnx=$1 --saveEngine=resnet_engine_intro.engine --stronglyTyped
