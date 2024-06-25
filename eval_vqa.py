# -*- coding: utf-8 -*-
import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from models.mile_vqa import mile_vqa
from models.mile_lora_eval import mile_lora_vqa
from models.mile_ia3_eval import mile_ia3_vqa
from models.mile_prefix_eval import mile_prefix_vqa
from models.mile_ptv2_eval import mile_ptv2_vqa

import utils
from data.utils import save_result
from data.vqa_dataset import Slake_dataset
from torchvision import transforms
from torch.nn.parallel import DataParallel


def get_MILE_model(args, config):
    if args.lora_MILE:
        model = mile_lora_vqa(pretrained=config['pretrained'], image_size=config['image_size'], 
                        vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])
        
    elif args.ia3_MILE:
        model = mile_ia3_vqa(pretrained=config['pretrained'], image_size=config['image_size'], 
                        vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])
        
    elif args.prefix_MILE:
        model = mile_prefix_vqa(pretrained=config['pretrained'], image_size=config['image_size'], 
                        vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])
        
    elif args.PTv2_MILE:
        model = mile_ptv2_vqa(pretrained=config['pretrained'], image_size=config['image_size'], 
                        vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])
    else: ## Full FineTuning
        model = mile_vqa(pretrained=config['pretrained'], image_size=config['image_size'], 
                        vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])
    return model

@torch.no_grad()
def evaluation(model, data_loader, device, config) :
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50
    
    result = []
        
    for n, (image, question,answer,qid) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):        
        image = image.to(device,non_blocking=True)             

        if config['inference']=='generate':
            answers = model(image, question, train=False, inference='generate') 
            qid = qid.tolist()
            for answer,q in zip(answers,qid):
                #ques_id = int(ques_id.item())       
                result.append({ "answer":answer,"qid":q})             

    return result


def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    #### Dataset #### 
    print("Creating vqa datasets")
    transform_train = transforms.Compose([
            transforms.Resize((480,480)),
            transforms.ToTensor(),
            ])

    ## set your dataset path
    datadir = r'.../.../.../Slake/Slake1.0/imgs'
    test_labeldir = r'.../.../.../Slake/Slake1.0/slake_test.jsol'
    test_datasets = Slake_dataset(data_dir=datadir, label_dir= test_labeldir, transform= transform_train)
    test_loader = DataLoader(dataset=test_datasets,batch_size=100, num_workers=26, drop_last=False)
    
    #### Model #### 
    print("Creating model")
    model = get_MILE_model(args, config)    
    model = model.to(device)      

    vqa_result = evaluation(model, test_loader, device, config)        
    result_file = save_result(vqa_result, args.result_dir, 'eval_result')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/vqa.yaml') 
    parser.add_argument('--output_dir', default='.../.../.../MILE_eval_result_save')
    parser.add_argument('--checkpoint', default='.../.../.../MILE_checkpoint.pth')
    parser.add_argument('--evaluate', action='store_true')      
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    
    parser.add_argument('--lora_MILE', default=False, type=bool)
    parser.add_argument('--ia3_MILE', default=False, type=bool)
    parser.add_argument('--prefix_MILE', default=False, type=bool)
    parser.add_argument('--PTv2_MILE', default=False, type=bool)
    
    args = parser.parse_args()

    yaml = yaml.YAML(typ='rt')  
    config = yaml.load(open(args.config, 'r'))

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)