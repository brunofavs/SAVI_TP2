import os
import os.path as osp
import glob
import cv2
import numpy as np
import torch

# import RRDBNet_arch as arch

import lib.ESRGAN_upscaler.RRDBNet_arch as arch


class imageUpscaler():

    def __init__(self):
        
        cwd = os.getcwd()
        os.chdir(f'{os.getenv("SAVI_TP2")}/src/lib/ESRGAN_upscaler/models')

        
        self.model_path = 'RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
        # model_path = 'models/RRDB_PSNR_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
        self.device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
        # device = torch.device('cpu')



        self.model = arch.RRDBNet(3, 3, 64, 23, gc=32)
        self.model.load_state_dict(torch.load(self.model_path), strict=True)
        self.model.eval()
        self.model = self.model.to(self.device)

        os.chdir(cwd)

    def upscale(self,img):

        # cv2.imwrite("./image.png",img)
        # img = cv2.imread("./image.png",cv2.IMREAD_COLOR)


        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(self.device)

        with torch.no_grad():
            output = self.model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()

        output_image = output
        return output_image
