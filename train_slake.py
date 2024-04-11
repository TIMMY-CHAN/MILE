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

from models.miss_vqa import miss_vqa
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
from data.vqa_dataset import vqa_collate_fn
from data.utils import save_result
from data.vqarad_dataset import VQA_dataset,Slake_dataset
from torchvision import transforms


def train(model, data_loader, optimizer, epoch, device):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50    
    
    for i,(image, question, answer,qid) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        #print('type of input', type(image))
        image = image.to(device,non_blocking=True),      
        #print('type of input', type(image[0]))
        loss = model(image[0], question, answer, train=True, n=[1])        
        #print(question, answer,qid)
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
    
    #if config['inference']=='rank':   
    #    answer_list = data_loader.dataset.answer_list
    #    answer_candidates = model.tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)    
    #    answer_candidates.input_ids[:,0] = model.tokenizer.bos_token_id
        
    for n, (image, question,answer,qid) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):        
        image = image.to(device,non_blocking=True)             

        if config['inference']=='generate':
            answers = model(image, question, train=False, inference='generate') 
            qid = qid.tolist()
            for answer,q in zip(answers,qid):
                #ques_id = int(ques_id.item())       
                result.append({ "answer":answer,"qid":q})             
          
        #elif config['inference']=='rank':    
        #    answer_ids = model(image, question, answer_candidates, train=False, inference='rank', k_test=config['k_test'])      

            #for ques_id, answer_id in zip(question_id, answer_ids):
            #    result.append({"question_id":int(ques_id.item()), "answer":answer_list[answer_id]})  

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
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    #datasets = create_dataset('vqa', config)  
    datadir = r'/Dataset1/cjw/Slake/Slake1.0/imgs'
    train_labeldir = r'/Dataset1/cjw/Slake/Slake1.0/slake_train.jsol'
    test_labeldir = r'/Dataset1/cjw/Slake/Slake1.0/slake_test.jsol'
    train_datasets = Slake_dataset(data_dir=datadir,label_dir=train_labeldir,transform=transform_train) 
    test_datasets = Slake_dataset(data_dir=datadir, label_dir= test_labeldir, transform= transform_train)
    
    train_loader = DataLoader(dataset=train_datasets,batch_size=20,shuffle=True, num_workers=26, drop_last=False)
    test_loader = DataLoader(dataset=test_datasets,batch_size=100, num_workers=26, drop_last=False)
    #### Model #### 
    print("Creating model")
    model = miss_vqa(pretrained=config['pretrained'], image_size=config['image_size'], 
                       vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])

    model = model.to(device)   
    for name, param in model.named_parameters():
        if 'visual_encoder' in name:
            param.requires_grad = True
        #elif 'attention' in name:          
        #    param.requires_grad = False
        else:
            if 'attention' in name:
                param.requires_grad = True
            #elif 'output.dense' in name:
            #    param.requires_grad = True
            else:
                param.requires_grad = False
    model_without_ddp = model
    if False:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
    
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
    """    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu')        
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']+1                
        print('resume checkpoint from %s'%args.checkpoint)"""
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total_params',total_params)
    print('trainable_params', trainable_params)
    print('全局训练参数比', trainable_params/total_params)
    total_params = sum(p.numel() for p in model.text_encoder.parameters())
    trainable_params = sum(p.numel() for p in model.text_encoder.parameters() if p.requires_grad)
    print('encoderPEFT_total_params',total_params)
    print('encoderPEFT_trainable_params', trainable_params)
    print('encoder peft 参数比', trainable_params/total_params)
    total_params = sum(p.numel() for p in model.text_decoder.parameters())
    trainable_params = sum(p.numel() for p in model.text_decoder.parameters() if p.requires_grad)
    print('decoderPEFT_total_params',total_params)
    print('decoderPEFT_trainable_params', trainable_params)
    print('decoder peft参数比', trainable_params/total_params)
    best = 0
    best_epoch = 0 
       
    print("Start training")
    start_time = time.time()   
    start_epoch = 0
    for epoch in range(start_epoch, config['max_epoch']):
        if not args.evaluate:        
            #            if args.distributed:
            #    train_loader.sampler.set_epoch(epoch)
                
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
                #'optimizer': optimizer.state_dict(),
                'config': config,
                #'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))

        #dist.barrier()         
    
    vqa_result = evaluation(model_without_ddp, test_loader, device, config)        
    #result_file = save_result(vqa_result, args.result_dir, 'vqa_result_%02d'%epoch)  
    result_file = save_result(vqa_result, args.result_dir, 'vqa_test_epoch129instruction_result')
                      
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
    
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/home/cjw/code/VQA/miss-main/configs/slake.yaml') 
    parser.add_argument('--output_dir', default='/Dataset1/cjw/pretrained/miss/ffn_a')
    parser.add_argument('--checkpoint', default='/Dataset1/cjw/pretrained/MISS/missslake_freeze_PRE/checkpoint_59.pth')
    parser.add_argument('--evaluate', action='store_true')      
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    yaml = yaml.YAML(typ='rt')  
    config = yaml.load(open(args.config, 'r'))

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)