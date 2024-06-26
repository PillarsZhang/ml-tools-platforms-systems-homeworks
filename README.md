# torch.compile 与 TVM 的性能对比

## 结果

|    | model_name                 | test_item     |   batch_size |   compile_time_s |   infer_time_mean_ms |
|---:|:---------------------------|:--------------|-------------:|-----------------:|---------------------:|
|  0 | facebook/convnext-tiny-224 | torch         |            1 |        /       |              4.47948 |
|  1 | facebook/convnext-tiny-224 | torch_compile |            1 |          2.57027 |              3.45069 |
|  2 | facebook/convnext-tiny-224 | tvm           |            1 |         37.7127  |              4.57279 |
|  3 | facebook/convnext-tiny-224 | torch         |            4 |        /       |              4.4797  |
|  4 | facebook/convnext-tiny-224 | torch_compile |            4 |          2.05128 |              3.37868 |
|  5 | facebook/convnext-tiny-224 | tvm           |            4 |         43.3252  |             13.6116  |
|  6 | facebook/convnext-tiny-224 | torch         |           16 |        /       |              6.63025 |
|  7 | facebook/convnext-tiny-224 | torch_compile |           16 |          1.99523 |              4.62228 |
|  8 | facebook/convnext-tiny-224 | tvm           |           16 |         71.2845  |             53.154   |
|  9 | microsoft/resnet-50        | torch         |            1 |        /       |              5.51754 |
| 10 | microsoft/resnet-50        | torch_compile |            1 |          2.23461 |              3.12204 |
| 11 | microsoft/resnet-50        | tvm           |            1 |         62.4482  |              1.86983 |
| 12 | microsoft/resnet-50        | torch         |            4 |        /       |              5.56561 |
| 13 | microsoft/resnet-50        | torch_compile |            4 |          1.96603 |              3.04268 |
| 14 | microsoft/resnet-50        | tvm           |            4 |         66.6161  |              6.66447 |
| 15 | microsoft/resnet-50        | torch         |           16 |        /       |              5.58056 |
| 16 | microsoft/resnet-50        | torch_compile |           16 |          1.89306 |              4.17158 |
| 17 | microsoft/resnet-50        | tvm           |           16 |         66.1151  |              9.86793 |

![facebook/convnext-tiny-224](docs/facebook_convnext-tiny-224_infer_time.png)
![microsoft/resnet-50](docs/microsoft_resnet-50_infer_time.png)

## 环境

测试环境没有特殊要求，以下是我使用的环境：
- pytorch 2.1.0
- cuda 12.1.0

测试环境也需要nvcc，可以直接安装完整工具链（但这样似乎在同一环境里编译TVM和测试更方便）
```shell
conda install -c "nvidia/label/cuda-12.1.0" --override-channels cuda
```

### TVM安装

找不到新版本或支持较新的CUDA版本的TVM预编译包，但是从[源码编译](https://tvm.apache.org/docs/install/from_source.html)他是一个很痛苦的过程，这里记录一下编译的过程。

```shell
# 应对中国大陆特殊网络环境，建议配置ssh.github.com或代理
git clone --branch v0.16.0 --depth 1 git@github.com:apache/tvm.git
cd tvm
sed -i 's/https:\/\/github.com\//git@github.com:/g' .gitmodules
git submodule sync
git submodule update --init --recursive

# 还好有conda
conda env create --file conda/build-environment.yaml
conda activate tvm-build

# 安装目标CUDA工具链，我觉得这个版本要和测试环境的CUDA版本一致
conda install nvidia/label/cuda-12.1.0::cuda

# 按照说明配置CUDA、LLVM等编译选项
mkdir build
cp cmake/config.cmake build
CONFIG_FILE="build/config.cmake"
sed -i 's/set(USE_CUDA OFF)/set(USE_CUDA ON)/g' $CONFIG_FILE
sed -i 's/set(USE_LLVM OFF)/set(USE_LLVM "llvm-config --link-static")/g' $CONFIG_FILE
echo -e '\n# Avoid potential symbol conflicts between different versions LLVM used by TVM and PyTorch\nset(HIDE_PRIVATE_SYMBOLS ON)' >> $CONFIG_FILE

# 依赖librhash.so.0，但是conda的rhash版本都已经1.x.x，所以需要手动创建一个软链接，希望不会有问题
ln -s $(readlink $CONDA_PREFIX/lib/librhash.so) $CONDA_PREFIX/lib/librhash.so.0

# 需要应对 -- Found CUDA_CUDA_LIBRARY=CUDA_CUDA_LIBRARY-NOTFOUND
cd build
cmake -DCMAKE_LIBRARY_PATH=$CONDA_PREFIX/lib/stubs ..
make -j4

# 如果没有错误，切换到测试环境
conda deactivate
conda activate your_devlopment_env
cd python
pip install .
```

## 测试

运行所有测试，注意修改变量以适配硬件环境

```shell
bash benchmark_all.sh
```