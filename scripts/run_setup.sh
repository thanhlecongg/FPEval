pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
apt-get update
apt-get install -y tmux
opam install ounit2
eval $(opam env)
opam install ocamlfind ounit2