## Cloning the OpenVINO inference demo
```zsh
git clone https://github.com/garybaughintel/openvino_inference_demo.git
cd openvino_inference_demo
git lfs install
git lfs pull

```

## Installing OpenVINO Python API
[The latest installation instructions.](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html)
```zsh
python -m venv .venv
.venv\Scripts\activate
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

[OpenVINO model zoo demos](https://docs.openvino.ai/latest/omz_demos.html#doxid-omz-demos)

[YoloV4](https://www.youtube.com/watch?v=h08N0HX16l8)

[Netron for viewing OpenVINO models](https://netron.app/)

## Downloading and converting YoloV3 model
```zsh
omz_downloader --name yolo-v3-onnx
omz_converter --name yolo-v3-onnx
```
[More details on getting models from the OpenVINO model zoo](https://docs.openvino.ai/latest/omz_tools_downloader.html)

## Tracking large binary files with git lfs
[Managing large files](https://docs.github.com/en/repositories/working-with-files/managing-large-files/configuring-git-large-file-storage)
An example to add *.bin files to git lfs
```zsh
git lfs track "*.bin"
```

