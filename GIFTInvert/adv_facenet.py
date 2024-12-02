# -*- coding:utf-8 -*-

import sys
sys.path.append('..')

import os, inspect, shutil, json
import argparse
import subprocess
import csv
import cv2
import numpy as np
import random
import lpips
import torch
import torch.nn as nn
from tqdm import tqdm
from matplotlib import pyplot as plt
from glob import glob

from models import MODEL_ZOO
from models import build_generator
from models import parse_gan_type
from utils.misc import bool_parser
from utils.visualizer import HtmlPageVisualizer
from utils.visualizer import save_image
from utils.visualizer import load_image

from image_tools import preprocess, postprocess, Lanczos_resizing
import argparse

import torch
import numpy as np
import sys
import os
import os.path as osp

import torchvision.transforms as transforms
from torch.utils.data import DataLoader




from PIL import Image
from face_models import irse, ir152, facenet
import torch.nn.functional as F
from util import *
import torch.optim as optim
from faceparsing.model import BiSeNet

def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='')

    # StyleGAN
    parser.add_argument('--seed', type=int, default=1,
                        help='Seed for sampling. (default: %(default)s)')
    parser.add_argument('--model_name', type=str, default='stylegan2_ffhq1024',
                        help='Name to the pre-trained model.')
    parser.add_argument('--trunc_psi', type=float, default=0.7,
                        help='Psi factor used for truncation. This is '
                             'particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')
    parser.add_argument('--trunc_layers', type=int, default=8,
                        help='Number of layers to perform truncation. This is '
                             'particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')
    parser.add_argument('--randomize_noise', type=bool_parser, default=False,
                        help='Whether to randomize the layer-wise noise. This '
                             'is particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')

    # IO
    parser.add_argument('--inversion_dir', type=str, default='./results/stylegan2_ffhq1024', help='Latent head dir directory generated from invert.py')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory to save the results. If not specified, '
                             'the results will be saved to '
                             '`work_dirs/inversion/` by default. '
                             '(default: %(default)s)')
    parser.add_argument('--job_name', type=str, default='', help='Sub directory to save the results. If not specified, the result will be saved to {save_dir}/{model_name}')
    # Settings
    parser.add_argument('--use_FW_space', type=bool_parser, default=True)
    parser.add_argument('--basecode_spatial_size', type=int, default=16, help='spatial resolution of basecode.')
    parser.add_argument("--target_image_path", type=str, default="./target/085807.jpg", help="path to target image")
    parser.add_argument("--train_model_name_list", type=list, default=['ir152'], help="path to train model name list")
    parser.add_argument("--resize_rate", type=float, default=0.9, help="resize rate")
    parser.add_argument("--diversity_prob", type=float, default=0.5, help="diversity_prob")
    parser.add_argument("--diversity",type=int, default=5, help="diversity")
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()

    # Set adverarial args
    device = "cuda"
    target_image = read_img(args.target_image_path, 0.5, 0.5, device)
    targe_models = {}
    CEloss = torch.nn.CrossEntropyLoss()
    epoch = 50 #100
    for model in args.train_model_name_list:
        if model == 'ir152':
            targe_models[model] = []
            targe_models[model].append((112, 112))
            fr_model = ir152.IR_152((112, 112))
            fr_model.load_state_dict(torch.load('face_models/ir152.pth'))
            fr_model.to(device)
            fr_model.eval()
            targe_models[model].append(fr_model)
        if model == 'irse50':
            targe_models[model] = []
            targe_models[model].append((112, 112))
            fr_model = irse.Backbone(50, 0.6, 'ir_se')
            fr_model.load_state_dict(torch.load('face_models/irse50.pth'))
            fr_model.to(device)
            fr_model.eval()
            targe_models[model].append(fr_model)
        if model == 'facenet':
            targe_models[model] = []
            targe_models[model].append((160, 160))
            fr_model = facenet.InceptionResnetV1(num_classes=8631, device=device)
            fr_model.load_state_dict(torch.load('face_models/facenet.pth'))
            fr_model.to(device)
            fr_model.eval()
            targe_models[model].append(fr_model)
        if model == 'mobile_face':
            targe_models[model] = []
            targe_models[model].append((112, 112))
            fr_model = irse.MobileFaceNet(512)
            fr_model.load_state_dict(torch.load('face_models/mobile_face.pth'))
            fr_model.to(device)
            fr_model.eval()
            targe_models[model].append(fr_model)
    
    def input_noise(args, x):
        rnd = torch.rand(1).to(device)
        noise = torch.randn_like(x).to(device)
        x_noised = x + rnd * (0.1 ** 0.5) * noise
        x_noised.to(device)
        return x_noised if torch.rand(1) < args.diversity_prob else x

    def input_diversity(args, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * args.resize_rate)

        if args.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False).to(device)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left
        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0).to(
            device)
        return padded if torch.rand(1) < args.diversity_prob else x
    
    def parse_matrix(img):
        cp = '79999_iter.pth'
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        save_pth = osp.join('faceparsing/res/cp', cp)
        net.load_state_dict(torch.load(save_pth))
        net.eval()

        img = F.interpolate(img, size=(512, 512), mode='bilinear')
        img = torch.clamp((img +1)/2.0, 0, 1)   
        img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
        out = net(img)[0]
        # parsing = out.squeeze(0).argmax(0)
        return out
    
    def cal_adv_loss(source, target, model_name, target_models):
        input_size = target_models[model_name][0]
        fr_model = target_models[model_name][1]
        source_resize = F.interpolate(source, size=input_size, mode='bilinear')
        target_resize = F.interpolate(target, size=input_size, mode='bilinear')
        # print(source_resize.shape, target_resize.shape)
        emb_source = fr_model(source_resize)
        emb_target = fr_model(target_resize).detach()
        cos_loss = 1 - cos_simi(emb_source, emb_target)
        return cos_loss
    
    def cos_simi(emb_1, emb_2):
        return torch.mean(torch.sum(torch.mul(emb_2, emb_1), dim=1) / emb_2.norm(dim=1) / emb_1.norm(dim=1))
    
    # Set random seed.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Get work directory and job name.
    if args.save_dir:
        work_dir = args.save_dir
    else:
        work_dir = os.path.join('work_dirs', 'inversion')
    os.makedirs(work_dir, exist_ok=True)
    job_name = args.job_name
    if job_name == '':
        job_name = f'{args.model_name}'
    os.makedirs(os.path.join(work_dir, job_name), exist_ok=True)
    
    # Parse model configuration.
    if args.model_name not in MODEL_ZOO:
        raise SystemExit(f'Model `{args.model_name}` is not registered in 'f'`models/model_zoo.py`!')
    model_config = MODEL_ZOO[args.model_name].copy()
    url = model_config.pop('url')  # URL to download model if needed.
    # Build generation and get synthesis kwargs.
    print(f'Building generator for model `{args.model_name}` ...')
    generator = build_generator(**model_config)
    synthesis_kwargs = dict(trunc_psi=args.trunc_psi,
                            trunc_layers=args.trunc_layers,
                            randomize_noise=args.randomize_noise)
    print(f'Finish building generator.')

    # Load StyleGAN
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = os.path.join('checkpoints', args.model_name + '.pth')
    print(f'Loading StyleGAN checkpoint from `{checkpoint_path}` ...')
    if not os.path.exists(checkpoint_path):
        print(f'  Downloading checkpoint from `{url}` ...')
        subprocess.call(['wget', '--quiet', '-O', checkpoint_path, url])
        print(f'  Finish downloading checkpoint.')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'generator_smooth' in checkpoint:
        generator.load_state_dict(checkpoint['generator_smooth'])
    else:
        generator.load_state_dict(checkpoint['generator'])
    generator = generator.cuda()
    generator.eval()
    generator.requires_grad_(False)
    print(f'Finish loading StyleGAN checkpoint.')
    # Get GAN type
    stylegan_type = parse_gan_type(generator) # stylegan or stylegan2

    # Define layers used for base code
    basecode_layer = int(np.log2(args.basecode_spatial_size) - 2) * 2
    if stylegan_type == 'stylegan2':
        basecode_layer = f'x{basecode_layer-1:02d}'
    elif stylegan_type == 'stylegan':
        basecode_layer = f'x{basecode_layer:02d}'
    print('basecode_layer : ', basecode_layer)

    inversion_dir = args.inversion_dir
    os.makedirs(os.path.join(inversion_dir, "facenet"), exist_ok=True)
    print('inversion_dir : ', inversion_dir)
    print(glob(f'{inversion_dir}/invert_results'))

    image_list = []
    for filename in glob(f'{inversion_dir}/invert_results/*.png'):
        print('file_name : ', filename)
        image_basename = os.path.splitext(os.path.basename(filename))[0]
        image_list.append(image_basename)
        detailcode = np.load(f'{inversion_dir}/invert_detailcode/{image_basename}.npy')
        if args.use_FW_space:
            basecode = np.load(f'{inversion_dir}/invert_basecode/{image_basename}.npy')
        
        detailcode = torch.from_numpy(detailcode).type(torch.FloatTensor).cuda()
        basecode = torch.from_numpy(basecode).type(torch.FloatTensor).cuda()

        detailcode = detailcode.clone().detach().to('cuda')
        basecode = basecode.clone().detach().to('cuda')
        detailcode.requires_grad = True
        basecode.requires_grad = False
        inputs = read_img(os.path.join(f'{inversion_dir}/target_images',image_basename+'.png'), 0.5, 0.5, device)
        optimizer = optim.Adam([detailcode], lr=0.002)
        for ep in range(epoch):
            image = generator.synthesis(detailcode, randomize_noise=args.randomize_noise,
                                            basecode_layer=basecode_layer, basecode=basecode)['image']
            targeted_loss_list = []
            p_diversity = []
            for i in range(args.diversity):
                p_diversity.append(input_diversity(args, input_noise(args, image)).to(device))
            for model_name in targe_models.keys():
                for i in range(args.diversity):
                    adv_loss = cal_adv_loss(p_diversity[i], target_image, model_name,
                                                        targe_models)
                    targeted_loss_list.append(adv_loss)
            loss_adv = torch.mean(torch.stack(targeted_loss_list))
            loss_sem = CEloss(parse_matrix(inputs), parse_matrix(image)) * 0.01 
            loss = loss_adv + loss_sem
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            image = generator.synthesis(detailcode, randomize_noise=args.randomize_noise,
                                            basecode_layer=basecode_layer, basecode=basecode)['image']
            adv_image = postprocess(image.clone())[0]
            cv2.imwrite(os.path.join(work_dir, job_name, 'facenet', image_basename+'.png'), adv_image)
        
if __name__ == '__main__':
    main()

