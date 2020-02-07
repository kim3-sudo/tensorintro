python3 --version
pip3 --version
virtualenv --version

sudo apt update
sudo apt install python3-dev python3-pip
sudo pip3 install -U virtualenv  # system-wide install

virtualenv --system-site-packages -p python3 ./venv

source ./venv/bin/activate  # sh, bash, ksh, or zsh

pip install --upgrade pip
pip list

pip install --upgrade tensorflow
python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

deactivate
