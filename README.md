## 编译与运行
```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90   # Hopper/H20
make -j
./gemm_softmax

