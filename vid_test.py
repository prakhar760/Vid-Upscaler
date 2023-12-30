import os.path as osp
import os
import glob
import cv2
import numpy as np
import torch
from tqdm import tqdm
from RRDBNet_arch import RRDBNet

model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_video_path = 'LR/test1.mp4'
output_video_path = 'results/output_video.avi'
# frames_folder = 'frames'
os.makedirs(frames_folder, exist_ok=True)



model = RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))

cap = cv2.VideoCapture(test_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# print(fps)
# print(frame_count)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')

out = cv2.VideoWriter(output_video_path, fourcc, fps, (width * 4, height * 4))

print('Processing video...')
with torch.no_grad():
    first_frame_skipped = False

    for frame_idx in tqdm(range(frame_count), desc='Frames Processed'):
        ret, frame = cap.read()
        if not ret:
            break

        if not first_frame_skipped:
            first_frame_skipped = True
            continue

        frame = frame * 1.0 / 255
        frame = torch.from_numpy(np.transpose(frame[:, :, [2, 1, 0]], (2, 0, 1))).float()
        frame_LR = frame.unsqueeze(0)
        frame_LR = frame_LR.to(device)

        output = model(frame_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)

        out.write(output)
        
        # Save the processed frame as an image in the 'frames' folder
        # frame_filename = 'frame_{:04d}.png'.format(frame_idx)
        # frame_filepath = os.path.join(frames_folder, frame_filename)
        # cv2.imwrite(frame_filepath, output)

# Release video capture and writer
cap.release()
out.release()
cv2.destroyAllWindows()

print('Video processing complete. Output saved to', output_video_path)

