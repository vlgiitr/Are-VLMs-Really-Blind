import torch
from tqdm import tqdm
import os
from PIL import Image
import requests
import time
import re

def get_accuracy(l1,l2):
    count = 0
    for x,y in zip(l1,l2):
        if x==y:
            count+=1
    return count/len(l1)*100

def generate_cqa(model,model_qa,processor,dataset,prompt):
    token_count = 0
    results = []
    for sample in tqdm(dataset):
        
        if token_count%15==0 and token_count!=0:
            time.sleep(60)
            
        image = sample['image']
        
        model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
        input_len = model_inputs["input_ids"].shape[-1]
        
        decoded = ""
        with torch.inference_mode():
            generation = model.generate(**model_inputs, max_new_tokens=1024, do_sample=False)
            generation = generation[0][input_len:]
            decoded = processor.decode(generation, skip_special_tokens=True)
            
        prompt = "From the description of the image answer the question strictly in the asked format." + \
                 "Description :" + decoded + \
                 "Question : " + sample['prompt']
        
        response = model_qa.generate_content(prompt)       
        response = response.text
        results.append(response)
        token_count+=1
        
        return results

def generate(model,processor,tokenizer,dataset,prompt=None):
    results = []
    for sample in tqdm(dataset):
        image = sample['image']
        if prompt==None:
            prompt = sample['prompt']
        length = len(prompt)
        #print(prompt)
        pixel_values = processor(images=image, return_tensors="pt").to(model.device)["pixel_values"]

        messages = prompt
        
        input_ids = tokenizer.encode(messages, return_tensors="pt")
        image_token_id = tokenizer.convert_tokens_to_ids("<image>")
        image_prefix = torch.empty((1, getattr(processor, "image_seq_length")), dtype=input_ids.dtype).fill_(image_token_id)
        input_ids = torch.cat((image_prefix, input_ids), dim=-1).to(model.device)

        generation = model.generate(input_ids, pixel_values=pixel_values, max_new_tokens=10)
        generation = generation[0]
        decoded = processor.decode(generation, skip_special_tokens=True)[length:]
        #print(decoded)
        results.append(decoded)
        
    return results

def extract_numbers(s):
    numbers = re.findall(r'\d+', s)
    return f'{numbers[0]},{numbers[1]}' if len(numbers) >= 2 else None