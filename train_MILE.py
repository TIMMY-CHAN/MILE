# -*- coding: utf-8 -*-
import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.mile_vqa import mile_vqa

import utils
from utils import cosine_lr_schedule
from data.vqa_dataset import Slake_dataset
from torchvision import transforms
from torch.nn.parallel import DataParallel

from peft import TaskType, get_peft_model
from peft import LoraConfig, IA3Config, PrefixTuningConfig

def train(model, data_loader, optimizer, epoch, device):
    # train
    model.train() 
       
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50    
    
    for i,(image, question, answer,qid) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device,non_blocking=True),      
        loss = model(image[0], question, answer, train=True, n=[1])        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} 


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

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return list(lora_module_names)

def get_MILE_peft_config(peft):
    if peft=='lora':
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1)
        
    elif peft=='ia3':
        return IA3Config(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["key", "value", "output.dense"],
            inference_mode=False, 
            feedforward_modules=["output.dense"])
    
    elif peft=='prefix':
        return PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False, 
            num_virtual_tokens=10, 
            prefix_projection=True)
    
    elif peft=='ptv2':
        return PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False, 
            num_virtual_tokens=10, 
            prefix_projection=False)
    
    
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
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    ## set your dataset path
    datadir = r'.../.../.../Slake/Slake1.0/imgs'
    train_labeldir = r'.../.../.../Slake/Slake1.0/slake_train.jsol'
    
    # set your dataloader of your dataset
    train_datasets = Slake_dataset(data_dir=datadir,label_dir=train_labeldir,transform=transform_train) 
    train_loader = DataLoader(dataset=train_datasets,batch_size=20,shuffle=True, num_workers=26, drop_last=False)

    #### Model #### 
    print("Creating model")
    model = mile_vqa(pretrained=config['pretrained'], image_size=config['image_size'], 
                       vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])
    model = model.to(device)   
    
    # Check if at least two PEFT methods are True
    peft_flags = [args.lora_MILE, args.ia3_MILE, args.prefix_MILE, args.PTv2_MILE]
    if sum(peft_flags) >= 2:
        raise ValueError("Only one PEFT method can be used in MILE model.")

    if args.lora_MILE or args.ia3_MILE or args.prefix_MILE or args.PTv2_MILE: # (MILE with best performance: F-F-PEFT)
        ## freeze: vit + text_encoder: 
        for name, param in model.named_parameters():
            if 'visual_encoder' in name or 'text_encoder' in name:
                param.requires_grad = False

        if args.lora_MILE:
            decoder_peft_config = get_MILE_peft_config('lora')
            print('######## MILE_LoRA ########')
        elif args.ia3_MILE:
            decoder_peft_config = get_MILE_peft_config('ia3')
            print('######## MILE_IA3 ########')
        elif args.prefix_MILE:
            decoder_peft_config = get_MILE_peft_config('prefix')
            print('######## MILE_Prefix ########')
        elif args.PTv2_MILE:
            decoder_peft_config = get_MILE_peft_config('ptv2')
            print('######## MILE_PTuning-v2 ########')
        
        # text_decoder + PEFT
        model.text_decoder = get_peft_model(model.text_decoder, decoder_peft_config) 
        print('text_decoder parameters:')
        model.text_decoder.print_trainable_parameters()
    else:
        print('######## Full FineTuning ########')
    
    model_without_ddp = model
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
    
    print("Start training")
    start_time = time.time()   

    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:        
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])              
            train_stats = train(model, train_loader, optimizer, epoch, device) 
        else:         
            break        
        
        if utils.is_main_process():     
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                        }                
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")                        
                    
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
            }

            # save model checkpoint
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))
            
            # save MILE_peft_weight
            peft_model_id = os.path.join(args.output_dir, f'MILE_PEFT_text_decoder_checkpoint_{epoch}')
            model.text_decoder.save_pretrained(peft_model_id)
            


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    

    parser.add_argument('--config', default='configs/vqa.yaml')
    parser.add_argument('--output_dir', default='.../.../.../MILE_result_save') 
    parser.add_argument('--checkpoint', default='.../.../.../MISS_checkpoint.pth')
    
    parser.add_argument('--evaluate', action='store_true')      
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    
    # MILE-PEFT
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