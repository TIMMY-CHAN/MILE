import json  
import random 
import numpy as np 
path1 = r"/Dataset1/cjw/VQA_RAD/test.jsol"
# 打开文件  
with open(path1, 'r') as f:  
    for line in f:  
        data = [json.loads(line) for line in f]    # 读取一行json数据  

#path = r"C:\Users\Administrator\Desktop\新建文件夹\VQA试验记录\vaq_rad\vqa_result_80.json"
#path = r"C:\Users\Administrator\Desktop\新建文件夹\VQA试验记录\rsna_slake_vqa\vqa_test_result.json"
path2 = '/Dataset1/cjw/pretrained/eval_result_3.8/result/vqa_result_jy_instruction_FT_checkpoint129.json'
# path2 = '/Dataset1/cjw/pretrained/eval_result_3.8/result/vqa_result_instruction_tuning_checkpoint129.json'
# 打开文件  
with open(path2, 'r', encoding='utf-8') as f:  
    json_string = f.read()  
answer = json.loads(json_string)
# print('answer',answer)
'''path = r"G:\code\med_blip\test.jsol"
# 打开文件  
with open(path, 'r', encoding='utf-8') as f:  
    json_string = f.read()  
data = json.loads(json_string)'''
with open(path1, 'r', encoding='utf-8') as f:
    # 读取文件内容
    json_string = f.read()

# 将JSON字符串转换为包含所有对象的列表
data = [json.loads(line) for line in json_string.split('\n') if line.strip()]

#print(data_list)
x = ['yes','Yes','No','no']
y = ['yes','Yes','people','person']
z = ['No','no']
def deli(y,z,st = None):
    if st in y:
        return 0
    if st in z:
        return 1 
    
# 使用json.loads()解析JSON字符串  
i = 0
j = 0
q = []
e = []
for test, result in zip(data,answer):
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

with open(r'/home/cjw/code/VQA/BLIP-main/ACC_test/answer.txt', 'w') as f:  
    # 遍历列表中的每个元素，并将其写入文件  
    for item in q:  
        f.write("%s\n" % item)

with open(r'/home/cjw/code/VQA/BLIP-main/ACC_test/result.txt', 'w') as f:  
    # 遍历列表中的每个元素，并将其写入文件  
    for item in e:  
        f.write("%s\n" % item)