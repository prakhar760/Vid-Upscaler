# Video Quality Enhancer

## Testing Guide
#### Dependencies
- Create a virtual environment and activate it.
```
python -m venv venv
.\venv\Scripts\activate
```
- PyTorch >= 1.0 (CUDA version >= 7.5 if installing with CUDA) 
1. `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` (for CUDA Supported system).
2. `pip3 install torch torchvision torchaudio` (for non-CUDA support system).

- Python packages:  `pip install -r requirements.txt`

### Steps
1. Clone this github repo.
```
git clone https://github.com/prakhar760/Vid-Upscale
cd Vid-Upscale
```
2. Put your own **low-resolution images** in `LR` folder. 
3. Run `vid_test.py` for a video file `OR` `test.py` for an image file.
```
python test.py
```
4. The results are in `results` folder.

## NOTE
- It is recommended to run the run the code on a cuda supported system with GPU memory or a GPU server to get much faster processing.
- The model may not provide good enough results hence I'll be needing to train the model on my custom dataset which will require GPU server support.
