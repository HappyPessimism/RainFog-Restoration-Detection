import datetime
import os

import torch
import matplotlib
matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import shutil
import numpy as np
import random
import cv2
from PIL import Image
from tqdm import tqdm
from .utils import cvtColor, preprocess_input, resize_image
from .utils_bbox import DecodeBox
from .utils_map import get_coco_map, get_map

import pytorch_ssim
import torch.nn.functional as F


def PSNR(pred, target):
    # print(pred.shape)
    # print(target.shape)
    mse = F.mse_loss(pred*255, target*255)
    if mse == 0:
        return 100
    PIXEL_MAX = torch.tensor(255.0).float().to(device=pred.device)
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))


def visualize_detection(image,image_shape, input_shape, boxes, labels, scores=None, class_names=None, score_threshold=0.1, color=(0, 255, 0)):
    """
    可视化目标检测结果

    Parameters:
    - image: 输入的图像 (numpy array, BGR 格式)
    - boxes: 边界框 (list of tuples/lists, 每个边界框 [x_min, y_min, x_max, y_max])
    - labels: 每个边界框对应的类别标签 (list of ints)
    - scores: 每个边界框的置信度得分 (list of floats, 可选)
    - class_names: 类别标签的名称 (list of strings, 可选)
    - score_threshold: 置信度得分的阈值，低于此值的边界框将被忽略 (float, 默认值 0.5)
    - color: 边界框的颜色 (tuple, 默认值为绿色 (0, 255, 0))
    """
    
    # 创建副本以避免修改原图像
    image_copy = image.copy()
    # print(image_shape)

    ih, iw  = image_shape
    h, w    = input_shape

    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    dx = (w-nw)//2
    dy = (h-nh)//2

    boxes[:, [1,3]] = boxes[:, [1,3]]*nw/iw + dx
    boxes[:, [0,2]] = boxes[:, [0,2]]*nh/ih + dy
    boxes[:, 1:3][boxes[:, 1:3]<0] = 0
    boxes[:, 3][boxes[:, 3]>w] = w
    boxes[:, 2][boxes[:, 2]>h] = h
    box_w = boxes[:, 3] - boxes[:, 1]
    box_h = boxes[:, 2] - boxes[:, 0]
    boxes = boxes[np.logical_and(box_w>1, box_h>1)] # discard invalid box
    # 遍历每个边界框
    for i, box in enumerate(boxes):
        # print(box)
        # 如果有置信度得分并且低于阈值，则跳过
        if scores is not None and scores[i] < score_threshold:
            continue
        # box = map(int,box)

        
        # 提取边界框坐标
        y_min, x_min, y_max, x_max  = map(int, box)
        
        # 绘制边界框
        cv2.rectangle(image_copy, (x_min, y_min), (x_max, y_max), color, 2)
        
        # 准备标签文本
        label_text = str(labels[i])
        if class_names is not None:
            label_text = class_names[labels[i]]
        if scores is not None:
            label_text += f' {scores[i]:.2f}'
        
        # 设置标签文本的显示位置
        label_position = (x_min, y_min - 10 if y_min - 10 > 10 else y_min + 10)
        
        # 绘制标签
        cv2.putText(image_copy, label_text, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image_copy


class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir    = log_dir
        self.losses     = []
        self.val_loss   = []
        
        os.makedirs(self.log_dir)
        self.writer     = SummaryWriter(self.log_dir)
        # try:
        #     dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1])
        #     self.writer.add_graph(model, dummy_input)
        # except:
        #     pass

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        # try:
        #     if len(self.losses) < 25:
        #         num = 5
        #     else:
        #         num = 15
            
        #     plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
        #     plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        # except:
        #     pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

class EvalCallback2():
    def __init__(self, net, input_shape, class_names, num_classes, val_lines, log_dir, cuda, \
            map_out_path=".temp_map_out", max_boxes=100, confidence=0.05, nms_iou=0.5, letterbox_image=True, MINOVERLAP=0.5, eval_flag=True, period=1):
        super(EvalCallback2, self).__init__()
        
        self.net                = net
        self.input_shape        = input_shape
        self.class_names        = class_names
        self.num_classes        = num_classes
        self.val_lines          = val_lines
        self.log_dir            = log_dir
        self.cuda               = cuda
        self.map_out_path       = map_out_path
        self.max_boxes          = max_boxes
        self.confidence         = confidence
        self.nms_iou            = nms_iou
        self.letterbox_image    = letterbox_image
        self.MINOVERLAP         = MINOVERLAP
        self.eval_flag          = eval_flag
        self.period             = period

        self.ssim = 0
        self.image_num = 0
        
        self.bbox_util          = DecodeBox(self.num_classes, (self.input_shape[0], self.input_shape[1]))
        
        self.maps       = [0]
        self.epoches    = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")

    def save_sample_png(self, image_shape, sample_folder, image_id, img_gt, boxes, labels, scores, class_names, pixel_max_cnt = 255, height = -1, width = -1):
        if os.path.exists('./{}/{}_results'.format(sample_folder, 'img_result'))==False:
            os.mkdir('./{}/{}_results'.format(sample_folder, 'img_result'))

        img_gt = img_gt * 255.0
        img_copy_gt = img_gt.clone().data.permute(0, 2, 3, 1).cpu().numpy()
        img_copy_gt = np.clip(img_copy_gt, 0, pixel_max_cnt)
        img_copy_gt = img_copy_gt.astype(np.uint8)[0, :, :, :]
        img_copy_gt = cv2.cvtColor(img_copy_gt, cv2.COLOR_BGR2RGB)
        save_img_name_gt = image_id + '_gt' + '.png'
        save_img_path_gt = os.path.join('./{}/{}_results'.format(sample_folder, 'img_result'), save_img_name_gt)
        cv2.imwrite(save_img_path_gt, img_copy_gt)

        # img_pre = img_pre * 255.0
        # img_copy_pre = img_pre.clone().data.permute(0, 2, 3, 1).cpu().numpy()
        # img_copy_pre = np.clip(img_copy_pre, 0, pixel_max_cnt)
        # img_copy_pre = img_copy_pre.astype(np.uint8)[0, :, :, :]
        # img_copy_pre = cv2.cvtColor(img_copy_pre, cv2.COLOR_BGR2RGB)
        # save_img_name_pre = image_id + '_pre' + '.png'
        # save_img_path_pre = os.path.join('./{}/{}_results'.format(sample_folder, 'img_result'), save_img_name_pre)
        # cv2.imwrite(save_img_path_pre, img_copy_pre)

        img_det = visualize_detection(img_copy_gt,image_shape, self.input_shape, boxes, labels, scores, class_names)
        save_img_name_det = image_id + '_det' + '.png'
        save_img_path_det = os.path.join('./{}/{}_results'.format(sample_folder, 'img_result'), save_img_name_det)
        cv2.imwrite(save_img_path_det, img_det)



    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"), "w", encoding='utf-8') 
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            # out_tmp = outputs
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return 

            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]

        top_100     = np.argsort(top_conf)[::-1][:self.max_boxes]
        top_boxes   = top_boxes[top_100]
        top_conf    = top_conf[top_100]
        top_label   = top_label[top_100]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        self.save_sample_png(image_shape, self.log_dir,image_id,images,top_boxes,top_label,top_conf,class_names)
        return 
    
    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_eval
            # shutil.rmtree(self.map_out_path)
            if not os.path.exists(self.map_out_path):
                os.makedirs(self.map_out_path)
            if not os.path.exists(os.path.join(self.map_out_path, "ground-truth")):
                os.makedirs(os.path.join(self.map_out_path, "ground-truth"))
            if not os.path.exists(os.path.join(self.map_out_path, "detection-results")):
                os.makedirs(os.path.join(self.map_out_path, "detection-results"))
            print("Get map.")
            self.ssim = 0
            self.image_num = 0
            for annotation_line in tqdm(self.val_lines):
                line        = annotation_line.split()
                image_id    = os.path.basename(line[0]).split('.')[0]
                #------------------------------#
                #   读取图像并转换成RGB图像
                #------------------------------#
                image       = Image.open(line[0])
                #------------------------------#
                #   获得预测框
                #------------------------------#
                gt_boxes    = np.array([np.array(list(map(int,box.split(',')))) for box in line[2:]])
                #------------------------------#
                #   获得预测txt
                #------------------------------#
                self.get_map_txt(image_id, image, self.class_names, self.map_out_path)
                
                #------------------------------#
                #   获得真实框txt
                #------------------------------#
                with open(os.path.join(self.map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                    for box in gt_boxes:
                        left, top, right, bottom, obj = box
                        obj_name = self.class_names[obj]
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
                        
            print("Calculate Map.")
            result_path = os.path.join(self.log_dir, 'result.txt')
            with open(result_path, 'a') as f:
                f.write(str(epoch))
                f.write('\n')
            # try:
            #     temp_map = get_coco_map(class_names = self.class_names, path = self.map_out_path, result_path=self.log_dir)[1]
            # except:
            #     temp_map = get_map(self.MINOVERLAP, False, path = self.map_out_path)
            temp_map = get_coco_map(class_names = self.class_names, path = self.map_out_path, result_path=result_path)[1]
            self.maps.append(temp_map)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(temp_map))
                f.write("\n")
            
            plt.figure()
            plt.plot(self.epoches, self.maps, 'red', linewidth = 2, label='train map')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Map %s'%str(self.MINOVERLAP))
            plt.title('A Map Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_map.png"))
            plt.cla()
            plt.close("all")

            print("Get map done.")
            shutil.rmtree(self.map_out_path)


class EvalCallback2():
    def __init__(self, net, input_shape, class_names, num_classes, val_lines, log_dir, cuda, \
            map_out_path=".temp_map_out", max_boxes=100, confidence=0.05, nms_iou=0.5, letterbox_image=True, MINOVERLAP=0.5, eval_flag=True, period=1):
        super(EvalCallback2, self).__init__()
        
        self.net                = net
        self.input_shape        = input_shape
        self.class_names        = class_names
        self.num_classes        = num_classes
        self.val_lines          = val_lines
        self.log_dir            = log_dir
        self.cuda               = cuda
        self.map_out_path       = map_out_path
        self.max_boxes          = max_boxes
        self.confidence         = confidence
        self.nms_iou            = nms_iou
        self.letterbox_image    = letterbox_image
        self.MINOVERLAP         = MINOVERLAP
        self.eval_flag          = eval_flag
        self.period             = period

        self.ssim = 0
        self.criterion_ssim = pytorch_ssim.SSIM()
        self.psnr = 0
        self.image_num = 0
        
        self.bbox_util          = DecodeBox(self.num_classes, (self.input_shape[0], self.input_shape[1]))
        
        self.maps       = [0]
        self.epoches    = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")

    def save_sample_png(self, image_shape, sample_folder, image_id, img_in, img_pre, img_gt, boxes, labels, scores, class_names, pixel_max_cnt = 255, height = -1, width = -1):
        if os.path.exists('./{}/{}_results'.format(sample_folder, 'img_result'))==False:
            os.mkdir('./{}/{}_results'.format(sample_folder, 'img_result'))

        ssim_loss = self.criterion_ssim(img_pre, img_gt)
        psnr = PSNR(img_gt, img_pre)
        self.ssim += ssim_loss
        self.psnr += psnr
        self.image_num += 1
        

        # if random.random() < 0.05:
        img_in = img_in * 255.0
        img_copy_in = img_in.clone().data.permute(0, 2, 3, 1).cpu().numpy()
        img_copy_in = np.clip(img_copy_in, 0, pixel_max_cnt)
        img_copy_in = img_copy_in.astype(np.uint8)[0, :, :, :]
        img_copy_in = cv2.cvtColor(img_copy_in, cv2.COLOR_BGR2RGB)
        save_img_name_in = image_id + '_in' + '.png'
        save_img_path_in = os.path.join('./{}/{}_results'.format(sample_folder, 'img_result'), save_img_name_in)
        cv2.imwrite(save_img_path_in, img_copy_in)

        img_pre = img_pre * 255.0
        img_copy_pre = img_pre.clone().data.permute(0, 2, 3, 1).cpu().numpy()
        img_copy_pre = np.clip(img_copy_pre, 0, pixel_max_cnt)
        # img_pre_ssim = img_copy_pre
        img_copy_pre = img_copy_pre.astype(np.uint8)[0, :, :, :]
        img_copy_pre = cv2.cvtColor(img_copy_pre, cv2.COLOR_BGR2RGB)
        save_img_name_pre = image_id + '_pre' + '.png'
        save_img_path_pre = os.path.join('./{}/{}_results'.format(sample_folder, 'img_result'), save_img_name_pre)
        cv2.imwrite(save_img_path_pre, img_copy_pre)

        img_det = visualize_detection(img_copy_pre,image_shape, self.input_shape, boxes, labels, scores, class_names)
        save_img_name_det = image_id + '_det' + '.png'
        save_img_path_det = os.path.join('./{}/{}_results'.format(sample_folder, 'img_result'), save_img_name_det)
        cv2.imwrite(save_img_path_det, img_det)

        img_gt = img_gt * 255.0
        img_copy_gt = img_gt.clone().data.permute(0, 2, 3, 1).cpu().numpy()
        img_copy_gt = np.clip(img_copy_gt, 0, pixel_max_cnt)
        # img_ori_ssim = img_copy_ori
        img_copy_gt = img_copy_gt.astype(np.uint8)[0, :, :, :]
        img_copy_gt = cv2.cvtColor(img_copy_gt, cv2.COLOR_BGR2RGB)
        save_img_name_gt = image_id + '_gt' + '.png'
        save_img_path_gt = os.path.join('./{}/{}_results'.format(sample_folder, 'img_result'), save_img_name_gt)
        cv2.imwrite(save_img_path_gt, img_copy_gt)

        



    def get_map_txt(self, image_id, image, image_gt, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"), "w", encoding='utf-8') 
        image_shape = np.array(np.shape(image)[0:2])
        # image_shape = np.array([1080,1920])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        image_gt       = cvtColor(image_gt)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_gt_data  = resize_image(image_gt, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        image_gt_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_gt_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            images_gt = torch.from_numpy(image_gt_data)
            if self.cuda:
                images = images.cuda()
                images_gt = images_gt.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images, images_gt)
            out_tmp = outputs
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return 

            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]

        top_100     = np.argsort(top_conf)[::-1][:self.max_boxes]
        top_boxes   = top_boxes[top_100]
        top_conf    = top_conf[top_100]
        top_label   = top_label[top_100]

        for i, c in list(enumerate(top_label)):

            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        # print(images)
        self.save_sample_png(image_shape, self.log_dir,image_id,images,out_tmp[6],images_gt,top_boxes,top_label,top_conf,class_names)
        return 
    
    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_eval
            # shutil.rmtree(self.map_out_path)
            if not os.path.exists(self.map_out_path):
                os.makedirs(self.map_out_path)
            if not os.path.exists(os.path.join(self.map_out_path, "ground-truth")):
                os.makedirs(os.path.join(self.map_out_path, "ground-truth"))
            if not os.path.exists(os.path.join(self.map_out_path, "detection-results")):
                os.makedirs(os.path.join(self.map_out_path, "detection-results"))
            print("Get map.")
            self.ssim = 0
            self.image_num = 0
            self.psnr = 0
            for annotation_line in tqdm(self.val_lines):
                line        = annotation_line.split()
                image_id    = os.path.basename(line[0]).split('.')[0]
                #------------------------------#
                #   读取图像并转换成RGB图像
                #------------------------------#
                file_name = line[0].split('/')[-2]
                gt_path = line[0].replace(file_name,'images')
                for i in range(1, 6):
                    gt_path = gt_path.replace('_f{}_r0'.format(i), '')
                gt_path = gt_path.replace('_f0_r1', '')
                image_gt = Image.open(gt_path)
                image       = Image.open(line[0])
                #------------------------------#
                #   获得预测框
                #------------------------------#
                gt_boxes    = np.array([np.array(list(map(int,box.split(',')))) for box in line[2:]])
                #------------------------------#
                #   获得预测txt
                #------------------------------#
                self.get_map_txt(image_id, image,image_gt, self.class_names, self.map_out_path)
                
                #------------------------------#
                #   获得真实框txt
                #------------------------------#
                with open(os.path.join(self.map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                    for box in gt_boxes:
                        left, top, right, bottom, obj = box
                        obj_name = self.class_names[obj]
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
                        
            print("Calculate Map.")
            result_path = os.path.join(self.log_dir, 'result.txt')
            with open(result_path, 'a') as f:
                f.write(str(epoch))
                f.write('SSIM:{}'.format(self.ssim/self.image_num))
                f.write('PSNR:{}'.format(self.psnr/self.image_num))
                f.write('\n')
            # try:
            #     temp_map = get_coco_map(class_names = self.class_names, path = self.map_out_path, result_path=self.log_dir)[1]
            # except:
            #     temp_map = get_map(self.MINOVERLAP, False, path = self.map_out_path)
            temp_map = get_coco_map(class_names = self.class_names, path = self.map_out_path, result_path=result_path)[1]
            self.maps.append(temp_map)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(temp_map))
                f.write("\n")
            
            plt.figure()
            plt.plot(self.epoches, self.maps, 'red', linewidth = 2, label='train map')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Map %s'%str(self.MINOVERLAP))
            plt.title('A Map Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_map.png"))
            plt.cla()
            plt.close("all")
            shutil.rmtree('.temp_map_out')
            print("Get map done.")


class EvalCallback3():
    def __init__(self, net, input_shape, class_names, num_classes, val_lines, log_dir, cuda, \
            map_out_path=".temp_map_out", max_boxes=100, confidence=0.05, nms_iou=0.5, letterbox_image=True, MINOVERLAP=0.5, eval_flag=True, period=1):
        super(EvalCallback3, self).__init__()
        
        self.net                = net
        self.input_shape        = input_shape
        self.class_names        = class_names
        self.num_classes        = num_classes
        self.val_lines          = val_lines
        self.log_dir            = log_dir
        self.cuda               = cuda
        self.map_out_path       = map_out_path
        self.max_boxes          = max_boxes
        self.confidence         = confidence
        self.nms_iou            = nms_iou
        self.letterbox_image    = letterbox_image
        self.MINOVERLAP         = MINOVERLAP
        self.eval_flag          = eval_flag
        self.period             = period

        self.ssim = 0
        self.criterion_ssim = pytorch_ssim.SSIM()
        self.psnr = 0
        self.image_num = 0
        
        self.bbox_util          = DecodeBox(self.num_classes, (self.input_shape[0], self.input_shape[1]))
        
        self.maps       = [0]
        self.epoches    = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")

    def save_sample_png(self, image_shape, sample_folder, image_id, img_in, img_pre, img_gt, boxes, labels, scores, class_names, pixel_max_cnt = 255, height = -1, width = -1):
        if os.path.exists('./{}/{}_results'.format(sample_folder, 'img_result'))==False:
            os.mkdir('./{}/{}_results'.format(sample_folder, 'img_result'))

        ssim_loss = self.criterion_ssim(img_pre, img_gt)
        psnr = PSNR(img_gt, img_pre)
        self.ssim += ssim_loss
        self.psnr += psnr
        self.image_num += 1
        

        # if random.random() < 0.05:
        img_in = img_in * 255.0
        img_copy_in = img_in.clone().data.permute(0, 2, 3, 1).cpu().numpy()
        img_copy_in = np.clip(img_copy_in, 0, pixel_max_cnt)
        img_copy_in = img_copy_in.astype(np.uint8)[0, :, :, :]
        img_copy_in = cv2.cvtColor(img_copy_in, cv2.COLOR_BGR2RGB)
        save_img_name_in = image_id + '_in' + '.png'
        save_img_path_in = os.path.join('./{}/{}_results'.format(sample_folder, 'img_result'), save_img_name_in)
        cv2.imwrite(save_img_path_in, img_copy_in)

        img_pre = img_pre * 255.0
        img_copy_pre = img_pre.clone().data.permute(0, 2, 3, 1).cpu().numpy()
        img_copy_pre = np.clip(img_copy_pre, 0, pixel_max_cnt)
        # img_pre_ssim = img_copy_pre
        img_copy_pre = img_copy_pre.astype(np.uint8)[0, :, :, :]
        img_copy_pre = cv2.cvtColor(img_copy_pre, cv2.COLOR_BGR2RGB)
        save_img_name_pre = image_id + '_pre' + '.png'
        save_img_path_pre = os.path.join('./{}/{}_results'.format(sample_folder, 'img_result'), save_img_name_pre)
        cv2.imwrite(save_img_path_pre, img_copy_pre)

        img_det = visualize_detection(img_copy_pre,image_shape, self.input_shape, boxes, labels, scores, class_names)
        save_img_name_det = image_id + '_det' + '.png'
        save_img_path_det = os.path.join('./{}/{}_results'.format(sample_folder, 'img_result'), save_img_name_det)
        cv2.imwrite(save_img_path_det, img_det)

        img_gt = img_gt * 255.0
        img_copy_gt = img_gt.clone().data.permute(0, 2, 3, 1).cpu().numpy()
        img_copy_gt = np.clip(img_copy_gt, 0, pixel_max_cnt)
        # img_ori_ssim = img_copy_ori
        img_copy_gt = img_copy_gt.astype(np.uint8)[0, :, :, :]
        img_copy_gt = cv2.cvtColor(img_copy_gt, cv2.COLOR_BGR2RGB)
        save_img_name_gt = image_id + '_gt' + '.png'
        save_img_path_gt = os.path.join('./{}/{}_results'.format(sample_folder, 'img_result'), save_img_name_gt)
        cv2.imwrite(save_img_path_gt, img_copy_gt)

        



    def get_map_txt(self, image_id, image, image_gt, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"), "w", encoding='utf-8') 
        image_shape = np.array(np.shape(image)[0:2])
        # image_shape = np.array([1080,1920])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        image_gt       = cvtColor(image_gt)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_gt_data  = resize_image(image_gt, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        image_gt_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_gt_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            images_gt = torch.from_numpy(image_gt_data)
            if self.cuda:
                images = images.cuda()
                images_gt = images_gt.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images, images_gt)
            out_tmp = outputs
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return 

            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]

        top_100     = np.argsort(top_conf)[::-1][:self.max_boxes]
        top_boxes   = top_boxes[top_100]
        top_conf    = top_conf[top_100]
        top_label   = top_label[top_100]

        for i, c in list(enumerate(top_label)):
            if int(c) > 2:
                c = 2
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        # print(images)
        self.save_sample_png(image_shape, self.log_dir,image_id,images,out_tmp[6],images_gt,top_boxes,top_label,top_conf,class_names)
        return 
    
    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_eval
            # shutil.rmtree(self.map_out_path)
            if not os.path.exists(self.map_out_path):
                os.makedirs(self.map_out_path)
            if not os.path.exists(os.path.join(self.map_out_path, "ground-truth")):
                os.makedirs(os.path.join(self.map_out_path, "ground-truth"))
            if not os.path.exists(os.path.join(self.map_out_path, "detection-results")):
                os.makedirs(os.path.join(self.map_out_path, "detection-results"))
            print("Get map.")
            self.ssim = 0
            self.image_num = 0
            self.psnr = 0
            for annotation_line in tqdm(self.val_lines):
                line        = annotation_line.split()
                image_id    = os.path.basename(line[0]).split('.')[0]
                #------------------------------#
                #   读取图像并转换成RGB图像
                #------------------------------#
                file_name = line[0].split('/')[-2]
                gt_path = line[0].replace(file_name,'images')
                for i in range(1, 6):
                    gt_path = gt_path.replace('_f{}_r0'.format(i), '')
                gt_path = gt_path.replace('_f0_r1', '')
                image_gt = Image.open(line[0])
                image       = Image.open(line[0])
                #------------------------------#
                #   获得预测框
                #------------------------------#
                gt_boxes    = np.array([np.array(list(map(int,box.split(',')))) for box in line[2:]])
                #------------------------------#
                #   获得预测txt
                #------------------------------#
                self.get_map_txt(image_id, image,image_gt, self.class_names, self.map_out_path)
                
                #------------------------------#
                #   获得真实框txt
                #------------------------------#
                with open(os.path.join(self.map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                    for box in gt_boxes:
                        left, top, right, bottom, obj = box
                        obj_name = self.class_names[obj]
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
                        
            print("Calculate Map.")
            result_path = os.path.join(self.log_dir, 'result.txt')
            with open(result_path, 'a') as f:
                f.write(str(epoch))
                f.write('SSIM:{}'.format(self.ssim/self.image_num))
                f.write('PSNR:{}'.format(self.psnr/self.image_num))
                f.write('\n')
            # try:
            #     temp_map = get_coco_map(class_names = self.class_names, path = self.map_out_path, result_path=self.log_dir)[1]
            # except:
            #     temp_map = get_map(self.MINOVERLAP, False, path = self.map_out_path)
            temp_map = get_coco_map(class_names = self.class_names, path = self.map_out_path, result_path=result_path)[1]
            self.maps.append(temp_map)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(temp_map))
                f.write("\n")
            
            plt.figure()
            plt.plot(self.epoches, self.maps, 'red', linewidth = 2, label='train map')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Map %s'%str(self.MINOVERLAP))
            plt.title('A Map Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_map.png"))
            plt.cla()
            plt.close("all")
            shutil.rmtree('.temp_map_out')
            print("Get map done.")
     