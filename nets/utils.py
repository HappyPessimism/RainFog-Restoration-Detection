import os
import cv2
import skimage
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
import math
import random
import network
import torch.nn.functional as F

# ----------------------------------------
#                 Network
# ----------------------------------------
def create_generator(opt):
    # Initialize the network
    generator = network.LarKPN(16, opt.color, opt.burst_length, opt.blind_est, opt.kernel_size, opt.sep_conv, \
        opt.channel_att, opt.spatial_att, opt.upMode, opt.core_bias)
    if opt.load_name == '':
        # Init the network
        network.weights_init(generator, init_type = opt.init_type, init_gain = opt.init_gain)
        print('Generator is created!')
    else:
        # Load a pre-trained network
        # pretrained_net = torch.load(opt.load_name)
        # load_dict(generator, pretrained_net)
        mload(generator, opt.load_name)
        print('Generator is loaded!')
    return generator

def mload(model, path):
    return model.load_state_dict(torch.load(path))

def load_dict(process_net, pretrained_net):
    # Get the dict from pre-trained network
    pretrained_dict = pretrained_net
    # Get the dict from processing network
    process_dict = process_net.state_dict()
    # Delete the extra keys of pretrained_dict that do not belong to process_dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in process_dict}
    # Update process_dict using pretrained_dict
    process_dict.update(pretrained_dict)
    # Load the updated dict to processing network
    process_net.load_state_dict(process_dict)
    return process_net

# ----------------------------------------
#    Validation and Sample at training
# ----------------------------------------
def save_sample_png(sample_folder, sample_name, img_list, name_list, pixel_max_cnt = 255, height = -1, width = -1):
    # Save image one-by-one
    for i in range(len(img_list)):
        img = img_list[i]
        # Recover normalization
        img = img * 255.0
        # Process img_copy and do not destroy the data of img
        #print(img.size())
        img_copy = img.clone().data.permute(0, 2, 3, 1).cpu().numpy()
        img_copy = np.clip(img_copy, 0, pixel_max_cnt)
        img_copy = img_copy.astype(np.uint8)[0, :, :, :]
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        # if (height != -1) and (width != -1):
        #     img_copy = cv2.resize(img_copy, (width, height))
        # Save to certain path
        save_img_name = sample_name + '_' + name_list[i] + '.png'
        save_img_path = os.path.join(sample_folder, save_img_name)
        cv2.imwrite(save_img_path, img_copy)

def save_sample_png2(sample_folder, Denoise_folder, sample_name, img_list, name_list, pixel_max_cnt = 255, height = -1, width = -1):
    # Save image one-by-one
    for i in range(len(img_list)):
        img = img_list[i]
        # Recover normalization
        img = img * 255.0
        # Process img_copy and do not destroy the data of img
        #print(img.size())
        img_copy = img.clone().data.permute(0, 2, 3, 1).cpu().numpy()
        img_copy = np.clip(img_copy, 0, pixel_max_cnt)
        img_copy = img_copy.astype(np.uint8)[0, :, :, :]
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        # if (height != -1) and (width != -1):
        #     img_copy = cv2.resize(img_copy, (width, height))
        # Save to certain path
        save_img_name = sample_name[0][0:8] + '_' + name_list[i] + '.png'
        if name_list[i] == 'pred':
            save_denoise_name = sample_name[0][0:8] + '.jpg'
            save_denoise_path = os.path.join(Denoise_folder, save_denoise_name)
            cv2.imwrite(save_denoise_path, img_copy)
        save_img_path = os.path.join(sample_folder, save_img_name)
        cv2.imwrite(save_img_path, img_copy)

def save_sample_png_test(sample_folder, sample_name, img_list, name_list, pixel_max_cnt = 255):
    # Save image one-by-one
    for i in range(len(img_list)):
        img = img_list[i]
        # Recover normalization
        img = img * 255.0
        # Process img_copy and do not destroy the data of img
        img_copy = img.clone().data.permute(0, 2, 3, 1).cpu().numpy()
        img_copy = np.clip(img_copy, 0, pixel_max_cnt)
        img_copy = img_copy.astype(np.uint8)[0, :, :, :]
        img_copy = img_copy.astype(np.float32)
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        # Save to certain path
        save_img_name = sample_name + '_' + name_list[i] + '.png'
        save_img_path = os.path.join(sample_folder, save_img_name)
        cv2.imwrite(save_img_path, img_copy)

def recover_process(img, height = -1, width = -1):
    img = img * 255.0
    img_copy = img.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    img_copy = np.clip(img_copy, 0, 255)
    img_copy = img_copy.astype(np.uint8)[0, :, :, :]
    img_copy = img_copy.astype(np.float32)
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    if (height != -1) and (width != -1):
        img_copy = cv2.resize(img_copy, (width, height))
    return img_copy

def psnr(pred, target):
    # print(pred.shape)
    # print(target.shape)
    mse = F.mse_loss(pred*255, target*255)
    if mse == 0:
        return 100
    PIXEL_MAX = torch.tensor(255.0).float().to(device=pred.device)
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))


'''
def psnr(pred, target, pixel_max_cnt = 255):
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    p = 20 * np.log10(pixel_max_cnt / rmse_avg)
    return p
'''

def grey_psnr(pred, target, pixel_max_cnt = 255):
    pred = torch.sum(pred, dim = 0)
    target = torch.sum(target, dim = 0)
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    p = 20 * np.log10(pixel_max_cnt * 3 / rmse_avg)
    return p

def ssim(pred, target):
    pred = pred.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    target = target.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    target = target[0]
    pred = pred[0]
    ssim = skimage.measure.compare_ssim(target, pred, multichannel = True)
    return ssim

# ----------------------------------------
#             PATH processing
# ----------------------------------------
def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def savetxt(name, loss_log):
    np_loss_log = np.array(loss_log)
    np.savetxt(name, np_loss_log)


#rain100H/L / SPA
def get_files(path):
    ret = []
    path_mix = path + "/mix_images"
    path_gt = path + "/images"
    path_name = path + "/ImageSets/Main/all.txt"
    with open(path_name, 'r') as f:
        files = [line.strip() for line in f.readlines()]
    for file in files:
        name = file
        rand_num = random.random()
        if rand_num <= 0.5:
            mix_name = '{}_f{}_r{}.jpg'.format(name, 0, 1)
        elif rand_num > 0.5 and rand_num <= 0.6:
            mix_name = '{}_f{}_r{}.jpg'.format(name, 1, 0)
        elif rand_num > 0.6 and rand_num <= 0.7:
            mix_name = '{}_f{}_r{}.jpg'.format(name, 2, 0)
        elif rand_num > 0.7 and rand_num <= 0.8:
            mix_name = '{}_f{}_r{}.jpg'.format(name, 3, 0)
        elif rand_num > 0.8 and rand_num <= 0.9:
            mix_name = '{}_f{}_r{}.jpg'.format(name, 4, 0)
        else:
            mix_name = '{}_f{}_r{}.jpg'.format(name, 5, 0)
        file_mix = path_mix + '/' + mix_name
        file_gt = path_gt + '/' + file + '.jpg'
        ret.append([file_mix, file_gt]) 
    return ret

def get_testfiles(path):
    ret = []
    file_rainy = '/home/shaoyl/Dataset/Vehicle_Car/mix_images/00133858_f3_r0.jpg'
    file_gt = '/home/shaoyl/Dataset/Vehicle_Car/images/00133858.jpg'
    ret.append([file_rainy, file_gt]) 
    return ret

def get_mixfiles(path, mode):
    ret = []
    path_mix = path + "/mix_images"
    path_gt = path + "/images"
    if mode == "train":
        path_name = path + "/ImageSets/Main/train.txt"
    elif mode == "val":
        path_name = path + "/ImageSets/Main/val.txt"
    else :
        path_name = ''
    with open(path_name, 'r') as f:
        files = [line.strip() for line in f.readlines()]
    for file in files:
        name = file
        rand_num = random.random()
        if rand_num <= 0.5:
            mix_name = '{}_f{}_r{}.jpg'.format(name, 0, 1)
        elif rand_num > 0.5 and rand_num <= 0.6:
            mix_name = '{}_f{}_r{}.jpg'.format(name, 1, 0)
        elif rand_num > 0.6 and rand_num <= 0.7:
            mix_name = '{}_f{}_r{}.jpg'.format(name, 2, 0)
        elif rand_num > 0.7 and rand_num <= 0.8:
            mix_name = '{}_f{}_r{}.jpg'.format(name, 3, 0)
        elif rand_num > 0.8 and rand_num <= 0.9:
            mix_name = '{}_f{}_r{}.jpg'.format(name, 4, 0)
        else:
            mix_name = '{}_f{}_r{}.jpg'.format(name, 5, 0)
        file_mix = path_mix + '/' + mix_name
        file_gt = path_gt + '/' + file + '.jpg'
        ret.append([file_mix, file_gt]) 
    return ret

'''
#rain1400
def get_files(path):
    ret = []

    path_rainy = path + "/rainy_image"
    path_gt = path + "/ground_truth"
    
    for root, dirs, files in os.walk(path_gt):
        files.sort()
        for name in files:
            if name.split('.')[1] != "jpg":
                continue
            id = name.split('.')[0]
            file_gt = path_gt + "/" + id + ".jpg"
            for i in range(1, 15):
                file_rainy = path_rainy + "/" + id + "_" + str(i) + ".jpg"
                ret.append([file_rainy, file_gt])
    return ret
'''

'''
#real
def get_files(path):
    # read a folder, return the complete path of rainy files and ground truth files
    ret=[]
    
    path_rainy = path + "/rainy_image"
    path_gt = path + "/ground_truth"

    for root, dirs, files in os.walk(path):
        files.sort()
        for name in files:
            if name.split('.')[1] != 'png':
                continue
            id = name.split('.')[0]
            file_gt = path + '/' + id + '.png'
            file_rainy = file_gt
            ret.append([file_rainy, file_gt])

    return ret
'''

'''
def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret
'''

def get_jpgs(path):
    # read a folder, return the image name
    ret = [] 
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(filespath)
    return ret
    
def get_last_2paths(path):
    # read a folder, return the image name
    ret = [] 
    for root, dirs, files in os.walk(path):
        for filespath in files:
            if filespath[-4:] == '.png':
                wholepath = os.path.join(root, filespath)
                last_2paths = os.path.join(wholepath.split('/')[-2], wholepath.split('/')[-1])
                ret.append(last_2paths)
    return ret
    
def text_readlines(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i])-1]
    file.close()
    return content

def text_save(content, filename, mode = 'a'):
    # save a list to a txt
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]))
    file.close()

'''
a = torch.randn(1, 3, 4, 4)
b = torch.randn(1, 3, 4, 4)
c = (a, b)
d = repackage_hidden(c)
print(d)
'''
'''
class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight = 1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class GradLoss(nn.Module):
    def __init__(self, GradLoss_weight = 1):
        super(GradLoss, self).__init__()
        self.GradLoss_weight = GradLoss_weight
        self.MSEloss = nn.MSELoss()

    def forward(self, x, y):
        h_x = x.size()[2]
        w_x = x.size()[3]

        x_h_grad = x[:, :, 1:, :] - x[:, :, :h_x - 1, :]
        x_w_grad = x[:, :, :, 1:] - x[:, :, :, :w_x - 1]
        y_h_grad = y[:, :, 1:, :] - y[:, :, :h_x - 1, :]
        y_w_grad = y[:, :, :, 1:] - y[:, :, :, :w_x - 1]
        
        h_loss = self.MSEloss(x_h_grad, y_h_grad)
        w_loss = self.MSEloss(x_w_grad, y_w_grad)
        
        return self.GradLoss_weight * (h_loss + w_loss)
'''
