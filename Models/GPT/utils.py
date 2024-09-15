from datasets import load_dataset
import base64
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image
import os
from openai import OpenAI
from tqdm import tqdm
import re


def get_accuracy(l1,l2):
    count = 0
    for x,y in zip(l1,l2):
        if x==y:
            count+=1
    return count/len(l1)*100

def get_image_url(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{base64_image}"

def generate(client,dataset,prompt=None):
    results = []
    model_name = "gpt-4o-mini"
    
    for sample in tqdm(dataset):
        image_url = get_image_url(sample['image'])
        if prompt==None:
            prompt = sample['prompt']
        
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": f"{prompt}"},
                {"type": "image_url", "image_url": {"url": f"{image_url}"},}
                       ]
                    }
                 ]
        response = client.chat.completions.create(model = model_name,messages = messages,seed = 300,max_tokens = 100)
        response = response.choices[0].message.content

        #print(response)
        results.append(response)
        
    return results

def extract_row_column(entry):
    if 'rows=' in entry and 'columns=' in entry:
        row = entry.split('rows={')[1].split('}')[0]
        col = entry.split('columns={')[1].split('}')[0]
        return f"{row},{col}"
    elif '(' in entry and ')' in entry:
        return entry.strip('()')
    
def generate_circled_letters(client,dataset):
    results = []
    model_name = "gpt-4o-mini"
    
    for sample in tqdm(dataset):
        image_url = get_image_url(sample['image'])
        prompt = sample['prompt'] + "Answer just a single character."
        
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": f"{prompt}"},
                {"type": "image_url", "image_url": {"url": f"{image_url}"},}
                       ]
                    }
                 ]
        response = client.chat.completions.create(model = model_name,messages = messages,seed = 300,max_tokens = 100)
        response = response.choices[0].message.content

        #print(response)
        results.append(response)
        
    return results

def generate_cqa(client,dataset,prompt=None):
    results_a = []
    model_name = "gpt-4o-mini"
    
    for sample in tqdm(dataset):
        image_url = get_image_url(sample['image'])

        if prompt==None:
            prompt = sample['prompt']
        
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": f"{prompt}"},
                {"type": "image_url", "image_url": {"url": f"{image_url}"},},
                       ],
                    }
                 ]
        response = client.chat.completions.create(model = model_name,messages = messages,seed = 300,max_tokens = 100)
        response = response.choices[0].message.content
        #print(response)
        qa = "From the description about the image answer the Question strictly in the asked format. " + \
             "Description : " + response + \
             "Question : " + sample['prompt']
        
        message_qa=[{
            "role": "user",
            "content": [{"type": "text", "text": f"{qa}"}]
                   }]
        
        response_a = client.chat.completions.create(model = model_name,messages = message_qa,seed = 300,max_tokens = 100)
        response_a = response_a.choices[0].message.content
        #print(response_a)
        results_a.append(response_a)
        
    return results_a

def generate_mod_cqa(client,dataset,prompt=None):
    results_a = []
    model_name = "gpt-4o-mini"
    
    for sample in tqdm(dataset):
        image_url = get_image_url(sample['image'])

        if prompt==None:
            prompt = sample['prompt']
        
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": f"{prompt}"},
                {"type": "image_url", "image_url": {"url": f"{image_url}"},},
                       ],
                    }
                 ]
        response = client.chat.completions.create(model = model_name,messages = messages,seed = 300,max_tokens = 75)
        response = response.choices[0].message.content
        #print(response)
        qa = "From the description about the image answer the Question strictly in the asked format. " + \
             "Description : " + response + \
             "Question : " + sample['prompt'] + "Answer in just a single character."
        
        message_qa=[{
            "role": "user",
            "content": [{"type": "text", "text": f"{qa}"}]
                   }]
        
        response_a = client.chat.completions.create(model = model_name,messages = message_qa,seed = 300,max_tokens = 100)
        response_a = response_a.choices[0].message.content
        #print(response_a)
        results_a.append(response_a)
        
    return results_a