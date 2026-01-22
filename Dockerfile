FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
RUN apt-get update; apt-get install -y tmux
RUN apt-get update; apt-get install -y cabal-install # install haskell
RUN apt-get update; apt-get install -y ocaml-nox # install ocaml
RUN apt-get update; apt-get install -y scala # install scala
RUN apt-get update; apt-get install -y openjdk-17-jdk # install java

COPY requirements.txt .
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

CMD ["bash"]

