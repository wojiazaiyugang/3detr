# 打包命令
# docker build -t registry.cn-hangzhou.aliyuncs.com/xiaoliutech/vfill_scan_tooth_detection:20250928 -f docker/Dockerfile . --network host --build-arg HTTP_PROXY=http://127.0.0.1:7890 --build-arg HTTPS_PROXY=http://127.0.0.1:7890
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel AS builder
WORKDIR /workspace
# 安装python包
COPY docker/sources.list /etc/apt/sources.list
COPY pyproject.toml poetry.lock /workspace/
RUN apt update && apt install -y gcc g++ libmysqlclient-dev libx11-dev libx11-6 libxext-dev libxtst-dev libxrender-dev libxmu-dev libxmuu-dev libgl1-mesa-glx libgomp1 --fix-missing --ignore-missing && \
    python -m pip install -U pip setuptools && \
    python -m pip install "poetry<2" && cd /workspace && poetry export -f requirements.txt -o requirements.txt --without-hashes && \
    python -m pip install -r requirements.txt
# 打包代码
COPY third_party/pointnet2 /workspace/third_party/pointnet2
# 编译
RUN cd /workspace/third_party/pointnet2 && python setup.py clean --all && TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0+PTX" python setup.py install --install-lib=/opt/conda/lib/python3.10/site-packages/
COPY datasets /workspace/datasets/
COPY detr /workspace/detr/
COPY models /workspace/models/
COPY scannet /workspace/scannet/
COPY utils /workspace/utils/
COPY third_party /workspace/third_party/
COPY service_http.py /workspace/
COPY config.py /workspace/
COPY main.py /workspace/
COPY engine.py /workspace/
COPY optimizer.py /workspace/
COPY criterion.py /workspace/
RUN python -m nuitka --module --remove-output --no-pyi-file --include-package=datasets datasets && rm -r datasets && \
    python -m nuitka --module --remove-output --no-pyi-file --include-package=detr detr && rm -r detr && \
    python -m nuitka --module --remove-output --no-pyi-file --include-package=models models && rm -r models && \
    python -m nuitka --module --remove-output --no-pyi-file --include-package=scannet scannet && rm -r scannet && \
    python -m nuitka --module --remove-output --no-pyi-file --include-package=utils utils && rm -r utils && \
    python -m nuitka --module --remove-output --no-pyi-file --include-package=third_party third_party && rm -r third_party && \
    python -m nuitka --module --remove-output --no-pyi-file config.py && rm config.py && \
    python -m nuitka --module --remove-output --no-pyi-file main.py && rm main.py && \
    python -m nuitka --module --remove-output --no-pyi-file engine.py && rm engine.py && \
    python -m nuitka --module --remove-output --no-pyi-file optimizer.py && rm optimizer.py && \
    python -m nuitka --module --remove-output --no-pyi-file criterion.py && rm criterion.py && \
    python -m nuitka --remove-output --no-pyi-file service_http.py --static-libpython=no && rm service_http.py
## 复制模型
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
RUN apt update && apt install -y gcc g++ libx11-dev libx11-6 libxext-dev libxtst-dev libxrender-dev libxmu-dev libxmuu-dev libgl1-mesa-glx libgomp1 --fix-missing --ignore-missing
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /opt/conda/lib/python3.10/site-packages/ /opt/conda/lib/python3.10/site-packages/
COPY --from=builder /workspace/*.so /workspace/
COPY --from=builder /workspace/service_http.bin /workspace/
# 复制模型文件
COPY "outputs/倒凹牙齿检测/2/checkpoint_best.pth.enc" /workspace/outputs/倒凹牙齿检测/2/
CMD ["./service_http.bin"]
