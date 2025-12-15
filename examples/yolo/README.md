This example shows how to implement YOLO object detection with ggml using pretrained model.

# YOLOv3-tiny

```bash
(base) harrytsz@sense:~/Workspace/Harrytsz-ggml/ggml$ cmake -B build
-- Warning: ccache not found - consider installing it for faster compilation or disable this warning with GGML_CCACHE=OFF
-- CMAKE_SYSTEM_PROCESSOR: x86_64
-- GGML_SYSTEM_ARCH: x86
-- Including CPU backend
-- x86 detected
-- Adding CPU backend variant ggml-cpu: -march=native 
-- x86 detected
-- Linux detected
-- ggml version: 0.9.4
-- ggml commit:  unknown
-- Configuring done
-- Generating done
-- Build files have been written to: /home/harrytsz/Workspace/Harrytsz-ggml/ggml/build
```

```
(base) harrytsz@sense:~/Workspace/Harrytsz-ggml/ggml$ cmake --build build --config Release --target yolov3-tiny
Scanning dependencies of target common
[  0%] Building CXX object examples/CMakeFiles/common.dir/common.cpp.o
[  3%] Linking CXX static library libcommon.a
[  3%] Built target common
[ 32%] Built target ggml-base
[ 82%] Built target ggml-cpu
[ 89%] Built target ggml
Scanning dependencies of target yolov3-tiny
[ 92%] Building CXX object examples/yolo/CMakeFiles/yolov3-tiny.dir/yolov3-tiny.cpp.o
[ 96%] Building CXX object examples/yolo/CMakeFiles/yolov3-tiny.dir/yolo-image.cpp.o
[100%] Linking CXX executable ../../bin/yolov3-tiny
[100%] Built target yolov3-tiny
```

Download the model weights:

```bash
$ wget https://pjreddie.com/media/files/yolov3-tiny.weights
$ sha1sum yolov3-tiny.weights 
40f3c11883bef62fd850213bc14266632ed4414f  yolov3-tiny.weights
```

Convert the weights to GGUF format:

```bash
$ ./convert-yolov3-tiny.py yolov3-tiny.weights
yolov3-tiny.weights converted to yolov3-tiny.gguf
```

Alternatively, you can download the converted model from [HuggingFace](https://huggingface.co/rgerganov/yolo-gguf/resolve/main/yolov3-tiny.gguf)

Object detection:

```bash
$ wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/dog.jpg
$ ./yolov3-tiny -m yolov3-tiny.gguf -i dog.jpg
load_model: using CUDA backend
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA T1200 Laptop GPU, compute capability 7.5, VMM: yes
Layer  0 output shape:  416 x 416 x   16 x   1
Layer  1 output shape:  208 x 208 x   16 x   1
Layer  2 output shape:  208 x 208 x   32 x   1
Layer  3 output shape:  104 x 104 x   32 x   1
Layer  4 output shape:  104 x 104 x   64 x   1
Layer  5 output shape:   52 x  52 x   64 x   1
Layer  6 output shape:   52 x  52 x  128 x   1
Layer  7 output shape:   26 x  26 x  128 x   1
Layer  8 output shape:   26 x  26 x  256 x   1
Layer  9 output shape:   13 x  13 x  256 x   1
Layer 10 output shape:   13 x  13 x  512 x   1
Layer 11 output shape:   13 x  13 x  512 x   1
Layer 12 output shape:   13 x  13 x 1024 x   1
Layer 13 output shape:   13 x  13 x  256 x   1
Layer 14 output shape:   13 x  13 x  512 x   1
Layer 15 output shape:   13 x  13 x  255 x   1
Layer 18 output shape:   13 x  13 x  128 x   1
Layer 19 output shape:   26 x  26 x  128 x   1
Layer 20 output shape:   26 x  26 x  384 x   1
Layer 21 output shape:   26 x  26 x  256 x   1
Layer 22 output shape:   26 x  26 x  255 x   1
dog: 57%
car: 52%
truck: 56%
car: 62%
bicycle: 59%
Detected objects saved in 'predictions.jpg' (time: 0.057000 sec.)
```

Copy data/cocos.names to ~/Workspace/Harrytsz-ggml/ggml
```
(base) harrytsz@sense:~/Workspace/Harrytsz-ggml/ggml$ ./build/bin/yolov3-tiny -m ./examples/yolo/yolov3-tiny.gguf -i ./examples/yolo/dog.jpg
create_backend: using CPU backend
main: failed to load labels from 'data/coco.names'
(base) harrytsz@sense:~/Workspace/Harrytsz-ggml/ggml$ ./build/bin/yolov3-tiny -m ./examples/yolo/yolov3-tiny.gguf -i ./examples/yolo/dog.jpg
create_backend: using CPU backend
Layer  0 output shape:  416 x 416 x   16 x   1
Layer  1 output shape:  208 x 208 x   16 x   1
Layer  2 output shape:  208 x 208 x   32 x   1
Layer  3 output shape:  104 x 104 x   32 x   1
Layer  4 output shape:  104 x 104 x   64 x   1
Layer  5 output shape:   52 x  52 x   64 x   1
Layer  6 output shape:   52 x  52 x  128 x   1
Layer  7 output shape:   26 x  26 x  128 x   1
Layer  8 output shape:   26 x  26 x  256 x   1
Layer  9 output shape:   13 x  13 x  256 x   1
Layer 10 output shape:   13 x  13 x  512 x   1
Layer 11 output shape:   13 x  13 x  512 x   1
Layer 12 output shape:   13 x  13 x 1024 x   1
Layer 13 output shape:   13 x  13 x  256 x   1
Layer 14 output shape:   13 x  13 x  512 x   1
Layer 15 output shape:   13 x  13 x  255 x   1
Layer 18 output shape:   13 x  13 x  128 x   1
Layer 19 output shape:   26 x  26 x  128 x   1
Layer 20 output shape:   26 x  26 x  384 x   1
Layer 21 output shape:   26 x  26 x  256 x   1
Layer 22 output shape:   26 x  26 x  255 x   1
dog: 57%
car: 52%
truck: 56%
car: 62%
bicycle: 59%
Detected objects saved in 'predictions.jpg' (time: 0.161000 sec.)
```
