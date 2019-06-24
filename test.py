from __future__ import print_function
import os
import argparse
import math
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg
from layers.functions.prior_box import PriorBox
from utils.nms_wrapper import nms
#from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.faceboxes import FaceBoxes
from utils.box_utils import decode
from utils.timer import Timer

parser = argparse.ArgumentParser(description='FaceBoxes')


parser.add_argument('-m', '--trained_model', default='weights2-256-NoOcculusion/FaceBoxes_epoch_499.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str, help='Dir to save results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--dataset', default='WIDER_val', type=str, choices=['AFW', 'PASCAL', 'FDDB','WIDER_val'], help='dataset')
parser.add_argument('--confidence_threshold', default=0.05, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys,
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    # net and model
    net = FaceBoxes(phase='test', size=None, num_classes=2)    # initialize detector
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    #print('Finished loading model!')
    #print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)


    # save file
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    fw = open(os.path.join(args.save_folder, args.dataset + '_detsNoOcclusion-499.txt'), 'w')

    # testing dataset
    testset_folder = os.path.join('data', args.dataset, 'images/')
    testset_list = os.path.join('data', args.dataset, 'img_list.txt')
    with open(testset_list, 'r') as fr:
        test_dataset = fr.read().split()
    num_images = len(test_dataset)
    #print(test_dataset)

    # testing scale
    if args.dataset == "FDDB":
        resize = 3
    elif args.dataset == "PASCAL":
        resize = 1
    elif args.dataset == "AFW" or args.dataset == "WIDER_val" :
        resize = 1.5

    _t = {'forward_pass': Timer(), 'misc': Timer()}

    # testing begin
    for i, img_name in enumerate(test_dataset):
        detection_dimensions = list()
    
        image_path = testset_folder + img_name
        #image_path = cv2.resize(image_path,(500,600))
        #print(image_path)
        img = np.float32(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
        #img = np.float32(cv2.imread(image_path, 0))
        #img = cv2.resize(img,(640,480))
        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
            rows = np.size(img,0)
            cols = np.size(img,1)

             
        img = np.expand_dims(img,axis=2)
        featureR = math.ceil(rows/32)
        featureC = math.ceil(cols/32)
        feaSize = torch.Size([featureR, featureC])
        detection_dimensions.append(feaSize)
        featureR = math.ceil(rows/64)
        featureC = math.ceil(cols/64)
        feaSize = torch.Size([featureR, featureC])
        detection_dimensions.append(feaSize)
        featureR = math.ceil(rows/128)
        featureC = math.ceil(cols/128)
        feaSize = torch.Size([featureR, featureC])
        detection_dimensions.append(feaSize)
        detection_dimensions = torch.tensor(detection_dimensions, device=device)

        im_height, im_width, _ = img.shape
        #print(img.shape)
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        #img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        _t['forward_pass'].tic()
        loc, conf = net(img)  # forward pass
        #out = net(img)  # forward pass

        #print(out)
        _t['forward_pass'].toc()
        _t['misc'].tic()

       # priorbox = PriorBox(cfg, out[2], (im_height, im_width), phase='test')
        priorbox = PriorBox(cfg, detection_dimensions, (im_height, im_width), phase='test')

        priors = priorbox.forward()
        priors = priors.to(device)

        #loc, conf, _ = out
        
        prior_data = priors.data
        #for i in loc.data.squeeze(0):
            #print(i)
        
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        conf = conf.data.squeeze(0)
        #print(boxes.shape)
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.data.cpu().numpy()[:, 1]
        # print(boxes.shape)
        # print(conf.shape)
        # print(scores.shape)
        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]
         
        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        #keep = py_cpu_nms(dets, args.nms_threshold)
        keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        _t['misc'].toc()

        # save dets
        if args.dataset == "WIDER_val" or args.dataset == "FDDB" :
            fw.write('{:s}\n'.format(img_name))
            fw.write('{:.1f}\n'.format(dets.shape[0]))
            for k in range(dets.shape[0]):
                xmin = dets[k, 0]
                ymin = dets[k, 1]
                xmax = dets[k, 2]
                ymax = dets[k, 3]
                score = dets[k, 4]
                w = xmax - xmin + 1
                h = ymax - ymin + 1
                fw.write('{:.3f} {:.3f} {:.3f} {:.3f} {:.10f}\n'.format(xmin, ymin, w, h, score))
        else:
            for k in range(dets.shape[0]):
                xmin = dets[k, 0]
                ymin = dets[k, 1]
                xmax = dets[k, 2]
                ymax = dets[k, 3]
                ymin += 0.2 * (ymax - ymin + 1)
                score = dets[k, 4]
                fw.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(img_name, score, xmin, ymin, xmax, ymax))
        print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(i + 1, num_images, _t['forward_pass'].average_time, _t['misc'].average_time))
    fw.close()
