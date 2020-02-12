# TensorFlow

Some Python3 code that runs on Linux, assuming dependencies. To install those dependencies, run:
`
sudo apt install python3
sudo pip install keras
sudo pip install tensorflow
`

The  `tensor.py` file will build some models, then run those models. It works best inside of a virtual environment (venv) set up in Ubuntu Server 18.04 LTS. To install it, you'll need the `python3-venv` package. Install it using `apt`. You can use the configuration shell script in this repository to do the rest of the configuration. `mv` the script into the right directory, then make the file eXecutable using the command: `chmod +x tensorflowsetup.sh`. Run the script using Run with `sudo ./tensorflowsetup.sh`.

Or, just run this on Google Colab.
