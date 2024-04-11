import json  
import random 
import numpy as np 
import sys
import pandas as pd
import csv
path1 = r"/Dataset1/cjw/VQA_RAD/test.jsol"
with open(path1, 'r', encoding='utf-8') as f:
    data_list = json.load(f)


# path2 = '/Dataset1/cjw/pretrained/eval_result_3.8/result/vqa_result_jy_instruction_FT_checkpoint129.json'
path2 = '/Dataset1/cjw/pretrained/eval_result_3.8/result/vqa_result_instruction_tuning_checkpoint129.json'
with open(path2, 'r', encoding='utf-8') as f:
    answer_list = json.load(f)

# data_list: 测试集
# answer_list：推理结果

#print(data_list)
x = ['yes','Yes','No','no']
y = ['yes','Yes','people','person']
z = ['No','no']
def deli(y,z,st = None):
    if st in y:
        return 0
    if st in z:
        return 1 

i = 0
j = 0
q = []
e = []

# slake_list = []
# for x in data_list:
#     qid = str(x['qid'])
#     gt = x['answer']
#     slake_list.append({
#         'qid':qid,
#         'gt': gt
#     })

# eval_list = []
# for x in answer_list:
#     qid = str(x['qid'])
#     answer = x['answer']
#     eval_list.append({
#         'qid':qid,
#         'answer': answer
#     })

# match_list = []
# for eval_d in eval_list:
#     for slake_d in slake_list:
#         if eval_d['qid'] == slake_d['qid']:
#             print(1)
#             match_list.append({
#                 'gt': slake_d['gt'],
#                 'answer': eval_d['answer']
#             })
# print(len(eval_list))
# print(len(slake_list))

for test, result in zip(data_list,answer_list):
    # print(type(test), type(result))
    # print(test)
    a = test['answer']
    b = result['answer']
    if a in x:
        i = i+1
        ans_0 = deli(y, z, st = a)
        ans_1 = deli(y,z,st = b)  
        if ans_0 == ans_1:
            j = j+1
    q.append(a)
    e.append(b)
print(j)
print(i)
print(j/i)
# print(q)
# print(e)

with open('/home/cjw/code/VQA/BLIP-main/ACC_test/answer.txt', 'w') as f:  
    # 遍历列表中的每个元素，并将其写入文件  
    for item in q:  
        f.write("%s\n" % item)

with open('/home/cjw/code/VQA/BLIP-main/ACC_test/result.txt', 'w') as f:  
    # 遍历列表中的每个元素，并将其写入文件  
    for item in e:  
        f.write("%s\n" % item)

