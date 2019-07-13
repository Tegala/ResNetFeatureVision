import os

import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

import cv2
from config import cfg
from resnet import ResNet


class ResNetFeatureVision:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = ResNet(cfg).cuda().eval()
        pretrained = cfg.OUTPUTS.PRETRAINED
        self.model.load_checkpoint(pretrained)
        
    def preprocess_image(self, img_path, input_size=(1000, 1000)):
        # mean and std list for channels (Imagenet)
        mean = [102.9801, 115.9465, 122.7717]
        std = [1., 1., 1.]
        cv2im = cv2.imread(img_path)
        # Resize image
        if input_size:
            cv2im = cv2.resize(cv2im, input_size)
        im_as_arr = np.float32(cv2im)
        im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
        im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
        # Normalize the channels
        for channel, _ in enumerate(im_as_arr):
            im_as_arr[channel] -= mean[channel]
            im_as_arr[channel] /= std[channel]
        # Convert to float tensor
        im_as_ten = torch.from_numpy(im_as_arr).float()
        # Add one more channel to the beginning. Tensor shape = 1,3,224,224
        im_as_ten.unsqueeze_(0)
        # Convert to Pytorch variable
        im_as_var = Variable(im_as_ten, requires_grad=True)
        return im_as_var

    def save_image_features(self, img_path, output_size=(112, 112)):
        input_size = cfg.MODEL.INPUT_SIZE
        results_root = cfg.OUTPUTS.RESULTS

        img = self.preprocess_image(img_path, input_size).cuda()
        outputs = self.model(img)

        for i, output in tqdm(enumerate(outputs, 1)):
            output = output.cpu().data.numpy().squeeze(0)
            output= 1.0/(1+np.exp(-1*output))
            output = np.round(output*255)
            
            save_path = os.path.join(results_root, str(i))
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            for j, out in enumerate(output, 1):
                if output_size:
                    out = cv2.resize(out, output_size)
                cv2.imwrite(os.path.join(save_path, 'layer{}_{}.png'.format(i, j)), out)

        print('image features saved at', results_root)

if __name__ == '__main__':
    img_path = 'datasets/lung/000002.png'
    rfv = ResNetFeatureVision(cfg)
    rfv.save_image_features(img_path, output_size=(256, 256))