import numpy as np
import sys
import cv2
from utils import batch_psnr, normalize, init_logger_ipol, \
				variable_to_cv2_image, remove_dataparallel_wrapper, is_rgb
import torch

import os
from models import FFDNet
from torch.autograd import Variable
import matplotlib.image as mpimg
from PIL import Image

class State_de():
    def __init__(self, size, move_range, model):
        self.image = np.zeros(size,dtype=np.float32)
        self.move_range = move_range
        self.net = model
        self.image_freq = np.zeros(size,dtype=np.float32)
        self.image_mag = np.zeros(size,dtype=np.float32)
        self.image_pha = np.zeros(size,dtype=np.float32)
    def reset(self, x):
        self.image = x
        self.raw = x * 255
        self.raw[np.where(self.raw <= 2)] = 3
        self.image_freq = torch.fft.fftshift(torch.fft.fft2(torch.from_numpy(self.image*255), dim=[-2, -1], norm='backward'))
        self.image_mag = torch.abs(self.image_freq)
        self.image_pha = torch.angle(self.image_freq)
        
    def step_el(self, act):
        #neutral = 4
        #print(act)
        #moves = transform_array(act.astype(np.float32)-neutral)
        #act = np.array(act, dtype=np.float32)
        moves = (act.astype(np.float32)-10)*0.01
        #moves = (move - neutral) / 20
        moves = torch.from_numpy(moves)
        print("moves:  ",moves[:, 0, :, :].max(),moves[:, 0, :, :].min(),moves[:, 0, :, :].mean())
        _,_,H,W = self.image.shape
        moved_image_mag = torch.from_numpy(np.zeros(self.image.shape, dtype=np.float32))
        image_mag = self.image_mag
        moves[:, 0, :, :] = moves[:, 0, :, :]
        moves[:, 1, :, :] = moves[:, 0, :, :]
        moves[:, 2, :, :] = moves[:, 0, :, :]
        moved_image_mag = np.exp(moves) * image_mag
        self.image_mag = moved_image_mag
        real = moved_image_mag * torch.cos(self.image_pha)
        imag = moved_image_mag * torch.sin(self.image_pha)
        image_out = torch.complex(real, imag)
        image_out = torch.fft.ifft2(torch.fft.ifftshift(image_out), s=(H, W), dim=[-2, -1], norm='backward')
        image_out = image_out.real.numpy()
        image_out = np.clip(image_out,0,255)
        print(image_out.mean(),image_out.max(),image_out.min())
        image_out = image_out/255
        
        self.image = 0.8 * image_out + 0.2 * self.image

    def step_de(self, act_b):
        pix_num = act_b.shape[1] * act_b.shape[2]
        threshold = pix_num
        checker = act_b.sum(1)
        checker = checker.sum(1)
        
        for i in range(len(checker)):
            sh_im = self.image[i].shape  # Note the shape of the ith image
            imorig = np.expand_dims(self.image[i], 0)
            imorig_float = imorig * 255
            lowimg = np.expand_dims(self.raw[i], 0)
            
            expanded_h = False
            expanded_w = False

            # Check and expand height if necessary
            if sh_im[1] % 2 == 1:
                expanded_h = True
                imorig = np.concatenate((imorig, imorig[:, :, -1:, :]), axis=2)
            
            # Check and expand width if necessary
            if sh_im[2] % 2 == 1:
                expanded_w = True
                imorig = np.concatenate((imorig, imorig[:, :, :, -1:]), axis=3)

            # Convert to Tensor
            imorig_tensor = torch.Tensor(imorig)

            # Sets data type according to CPU or GPU modes
            dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

            # Noise level map
            nsigma = (imorig_float - lowimg) / lowimg
            nsigma = 0 + (np.max(nsigma) * 2 - 0) * (nsigma - np.min(nsigma)) / (np.max(nsigma) - np.min(nsigma))
            nsigma = np.clip(nsigma, 0, 255) / 255

            nsigma = nsigma.astype('float32')
            nsigma_shape = (imorig_tensor.shape[2] // 2, imorig_tensor.shape[3] // 2)
            nsigma = nsigma[:, :, :nsigma_shape[0], :nsigma_shape[1]]  # Ensure nsigma dimensions match expected shape

            # Test mode
            with torch.no_grad():
                imorig_tensor = Variable(imorig_tensor.type(dtype))
                nsigma_tensor = Variable(torch.FloatTensor(nsigma).type(dtype))


            # Estimate noise and subtract it from the input image
            im_noise_estim = self.net(imorig_tensor, nsigma_tensor)
            outim = torch.clamp(imorig_tensor - im_noise_estim, 0., 1.)

            outim_np = outim.cpu().detach().numpy()

            # Remove expanded parts to keep the original shape
            if expanded_h:
                outim_np = outim_np[:, :, :-1, :]
            if expanded_w:
                outim_np = outim_np[:, :, :, :-1]

            # Store the processed image back
            self.image[i] = outim_np.squeeze(0) 
