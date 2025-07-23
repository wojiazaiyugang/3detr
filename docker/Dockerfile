# 打包命令
# docker build -t registry.cn-hangzhou.aliyuncs.com/xiaoliutech/vfill_scan_tooth_detection:20250723 -f docker/Dockerfile . --network host --build-arg HTTP_PROXY=http://127.0.0.1:7890 --build-arg HTTPS_PROXY=http://127.0.0.1:7890
FROM nvidia/cuda:11.4.3-devel-ubuntu20.04

WORKDIR /workspace

# 安装系统依赖和Python
RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y --fix-missing --no-install-recommends libx11-dev libxext-dev libxtst-dev libxrender-dev libxmu-dev libxmuu-dev libgl1-mesa-glx software-properties-common libopenblas-dev && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install -y --no-install-recommends --fix-missing python3.9 python3.9-dev python3.9-distutils python3-pip && \
    ln -sf /usr/bin/python3.9 /usr/bin/python && \
    apt clean -y && rm -rf /var/cache/debconf/* /var/lib/apt/lists/* /var/log/* /var/tmp/* && rm -rf /tmp/* 

# 安装python包
COPY pyproject.toml poetry.lock /workspace/
RUN apt update && apt install -y --no-install-recommends libmysqlclient-dev && \
    python -m pip install -U pip setuptools && \
    python -m pip install "poetry<2" && cd /workspace && poetry export -f requirements.txt -o requirements.txt --without-hashes && \
    python -m pip install -r requirements.txt --no-cache-dir
# 打包代码
COPY third_party/pointnet2 /workspace/third_party/pointnet2
# 编译
RUN cd /workspace/third_party/pointnet2 && python setup.py clean --all && TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0+PTX" python setup.py install --install-lib=/usr/local/lib/python3.9/dist-packages
## 复制模型
COPY . /workspace/
COPY outputs/scan_tooth/42/checkpoint_best.pth /workspace/outputs/scan_tooth/42
CMD ["python", "service_http.py"]
