## Cloning the OpenVINO inference demo
```zsh
git clone https://github.com/garybaughintel/openvino_inference_demo.git
git lfs install
git lfs pull
cd openvino_inference_demo
```

## Installing OpenVINO Python API
[The latest installation instructions.](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html)
```zsh
python -m venv openvino_env
openvino_env/Scripts/activate
python -m pip install --upgrade pip
pip install openvino-dev==2022.3.0
```

## Running the reference demo
```zsh
python ov_inference.py
```
## References
[Introduction to convolutional neural networks (CNN)](https://github.com/baughg/lenet-mnist.git)

[Installing Python on Windows](https://www.tomshardware.com/how-to/install-python-on-windows-10-and-11)

[Git for Windows](https://gitforwindows.org/)
