# Video Quality Enhancer

## Testing Guide

### Steps
1. Clone this github repo.
```
git clone https://github.com/prakhar760/Vid-Upscale
cd Vid-Upscale
```
#### Dependencies
- Create a virtual environment and activate it.
```
python -m venv venv
.\venv\Scripts\activate
```
- PyTorch >= 1.0 (CUDA version >= 7.5 if installing with CUDA) 
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # for CUDA Supported system
pip install torch torchvision torchaudio # for non-CUDA support system
```
- Python packages:  `pip install -r requirements.txt`
<br>
<br>
2. Put your own **low-resolution images** in `LR` folder. <br>
3. Run `vid_test.py` for a video file `OR` `test.py` for an image file.<br>
```
python vid_test.py
python test.py
```
4. The results are in `results` folder.

## NOTE
- It is recommended to run the run the code on a cuda supported system with GPU memory or a GPU server to get much faster processing.
- The model may not provide good enough results hence I'll be needing to train the model on my custom dataset which will require GPU server support.
