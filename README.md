# [MICCAI 2024] Can LLMs' Tuning Methods Work in Medical Multimodal Domain?

## Abstract
While Large Language Models (LLMs) excel in world knowledge understanding, adapting them to specific subfields requires precise adjustments. Due to the model's vast scale, traditional global fine-tuning methods for large models can be computationally expensive and impact generalization. To address this challenge, a range of innovative Parameters-Efficient Fine-Tuning (PEFT) methods have emerged and achieved remarkable success in both LLMs and Large Vision-Language Models (LVLMs). In the medical domain, fine-tuning a medical Vision-Language Pretrained (VLP) model is essential for adapting it to specific tasks. Can the fine-tuning methods for large models be transferred to the medical field to enhance transfer learning efficiency? In this paper, we delve into the fine-tuning methods of LLMs and conduct extensive experiments to investigate the impact of fine-tuning methods for large models on existing multimodal models in the medical domain from the training data level and the model structure level. We show the different impacts of fine-tuning methods for large models on medical VLMs and develop the most efficient ways to fine-tune medical VLP models. We hope this research can guide medical domain researchers in optimizing VLMs' training costs, fostering the broader application of VLMs in healthcare fields.

<img src="MILE.png" width="700">

This is the PyTorch code of the <a href="https://arxiv.org/abs/2403.06407"> paper</a>. To install the dependencies, run: <pre/> pip install -r requirements.txt</pre> 


## Datasets:
1. Downloading links of some medical datasets:

| Dataset Name | Link |
|--------------|------|
| VQA-RAD | https://osf.io/89kps/|
| SLAKE | https://www.med-vqa.com/slake/ |


2. The instruction-format dataset proposed in this paper is constructed based on SLAKE and VQA-RAD and stored in [instruction_data.json](https://github.com/TIMMY-CHAN/MILE/blob/main/Instruction_dataset/instruction_data.json). Please prepare the SLAKE and VQA-RAD image datasets, and you can follow the directory setting below:

```
MILE/
├── Instruction_dataset/
│   ├── image/
│   │   ├── slake/
│   │   └── vqa-rad/
│   └── instruction_data.json/
└── ...
```

## Train:
1. This code is based on the SLAKE dataset for demonstration. Prepare your training dataset. Follow [vqa_dataset.py](https://github.com/TIMMY-CHAN/MILE/blob/main/data/vqa_dataset.py) and add a loader for your dataset.

2. [train_MILE.py](https://github.com/TIMMY-CHAN/MILE/blob/main/train_MILE.py) is used for MILE training. Four PEFT methods are provided in this file: LoRA, IA3, Prefix, and P-Tuning (v2). You can add more PEFT methods based on this file. 
3. Modify the paths of your input, output, pre-trained checkpoint, and training config in [train_MILE.py](https://github.com/TIMMY-CHAN/MILE/blob/main/train_MILE.py) and [vqa.yaml](https://github.com/TIMMY-CHAN/MILE/blob/main/configs/vqa.yaml) according to your needs.
4. To fine-tune the baseline model MISS by method MILE, run:
<pre/> python train_MILE.py --lora_MILE False --ia3_MILE False --prefix_MILE False --PTv2_MILE False</pre>

Set the parameter to **True** corresponding to one PEFT method you need to use . Or, it is full parameter fine-tuning if the corresponding parameters of all PEFT methods are **False**.

## Evaluation:
[eval_vqa.py](https://github.com/TIMMY-CHAN/MILE/blob/main/eval_vqa.py) is used for evaluation. 
1. Modify some file paths based on your needs. 
2. Depending on the new PEFT method you use, create a new MILE_model file for evaluation, refering to [mile_lora_eval.py](https://github.com/TIMMY-CHAN/MILE/blob/main/models/mile_lora_eval.py)
3. For evaluation, run:
<pre/> python eval_vqa.py --lora_MILE False --ia3_MILE False --prefix_MILE False --PTv2_MILE False</pre>

## Experiment Results:
The experiment results are shown in Table 1 to Table 4 in the paper.
1. Table 1: Results of MILE-LoRA

| ViT | JTM | Dec | Rank | \#Params | Memory | Opened | Closed | Global |
|-----|-----|-----|------|----------|--------|--------|--------|--------|
| F   | LoRA| LoRA | 4    | 0.163%   | 5.19GB | 3.57   | 50.70  | 20.34  |
| F   | LoRA| LoRA | 8    | 0.325%   | 5.21GB | 3.57   | 50.70  | 20.34  |
| LoRA| LoRA| LoRA | 4    | 0.327%   | 26.63GB| 48.65  | 50.70  | 49.34  |
| LoRA| LoRA| LoRA | 8    | 0.652%   | 26.75GB| 48.93  | 50.70  | 49.57  |
| F   | T   | LoRA | 4    | 38.022%  | 7.26GB | 47.76  | 70.70  | 55.53  |
| F   | T   | LoRA | 8    | 38.072%  | 7.45GB | 50.21  | 70.99  | 57.18  |
| T   | LoRA| LoRA | 4    | 24.009%  | 26.96GB| 68.14  | 50.70  | 62.29  |
| T   | LoRA| LoRA | 8    | 24.133%  | 27.29GB| 68.28  | 50.70  | 62.38  |
| T   | T   | LoRA | 4    | 61.887%  | 27.60GB| 78.52  | 79.44  | 78.83  |
| T   | T   | LoRA | 8    | 61.919%  | 28.11GB| 78.66  | 80.56  | 79.30  |

2. Table 2: Results of MILE-Prefix

| ViT | JTM   | Dec    | \#Params  | Memory  | Opened | Closed | Global |
|-----|-------|--------|-----------|---------|--------|--------|--------|
| F   | F     | Prefix | 3.926%    | 4.62GB  | 0      | 50.7   | 17.3   |
| F   | Prefix| Prefix | 7.556%    | 4.67GB  | 0      | 50.7   | 17.3   |
| T   | Prefix| Prefix | 29.636%   | 26.41GB | 41.50  | 32.95  | 38.61  |
| T   | T     | Prefix | 63.354%   | 27.97GB | 76.82  | **82.25** | 78.65  |

3. Table 3: Results of MILE-IA3

| ViT | JTM | Dec | \#Params  | Memory  | Opened | Closed | Global |
|-----|-----|-----|-----------|---------|--------|--------|--------|
| F   | IA3 | IA3 | 0.051%    | 6.35GB  | 0      | 1.69   | 0.57   |
| IA3 | IA3 | IA3 | 0.061%    | 23.01GB | 0      | 50.70  | 16.98  |
| T   | IA3 | IA3 | 23.924%   | 26.83GB | 12.77  | 28.17  | 17.92  |
| F   | T   | IA3 | 37.987%   | 7.52GB  | 46.24  | 50.70  | 47.74  |
| T   | T   | IA3 | 61.866%   | 27.90GB | 72.20  | 47.04  | 63.77  |


4. Table 4: Results of MILE-PTV2

| ViT | JTM  | Dec  | \#Params  | Memory  | Opened | Closed | Global |
|-----|------|------|-----------|---------|--------|--------|--------|
| F   | PTV2 | PTV2 | 0.102%    | 4.52GB  | 0      | 0      | 0      |
| F   | F    | PTV2 | 0.051%    | 4.57GB  | 7.10   | 0      | 4.72   |
| T   | PTV2 | PTV2 | 23.963%   | 25.41GB | 13.62  | 29.30  | 18.87  |
| T   | T    | PTV2 | 61.876%   | 27.46GB | 74.18  | 49.86  | 66.04  |


5. Table 5: Extensive Results of BiomedGPT-Tiny

|Method          | \#Params  | Opened | Closed | Global |
|-----|-----------|--------|--------|--------|
|Full Fine-tuning|             100%    | 71.84  | 64.46  & 68.97  
|Decoder-LoRA    |            50.76%   | 66.82  | 63.48  & 65.52  
|Decoder-Prefix  |           51.05%    | 69.94  | 60.54  & 66.29  
|Decoder-IA3     |            50.49%   | 64.95  | 52.21  & 60.01  
|Decoder-PTV2    |            50.92%   | 68.07  | 48.78  & 60.57 

## Citation
If you find this code to be useful for your research, please consider citing.
<pre>
@misc{chen2024llms,
      title={Can LLMs' Tuning Methods Work in Medical Multimodal Domain?}, 
      author={Jiawei Chen and Yue Jiang and Dingkang Yang and Mingcheng Li and Jinjie Wei and Ziyun Qian and Lihua Zhang},
      year={2024},
      booktitle={MICCAI}
}
</pre>


## Related Projects

- [MISS](https://github.com/TIMMY-CHAN/MISS)
- [BLIP](https://github.com/salesforce/BLIP)
- [SLAKE](https://www.med-vqa.com/slake/)
- [VQA-RAD](https://osf.io/89kps/)
