FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
RUN apt-get update; apt-get install -y tmux
COPY requirements.txt .
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

CMD ["bash"]

