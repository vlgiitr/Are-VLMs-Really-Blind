from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import google.generativeai as genai
import torch
from datasets import load_dataset
from huggingface_hub import login
from tqdm import tqdm
import os
from PIL import Image
import requests
import time
import re
from utils import get_accuracy,generate,extract_numbers


ds = load_dataset("Ayush-Singh/llms-are-blind-captions_florence2-large")
model_id = "BUAADreamer/PaliGemma-3B-Chat-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(model_id, device_map="cuda")

### Direct QA

### Circled Letters
Sample_Circled_Letter = ds['Circled_Letter'].select(range(200))
result_cl = generate(model,processor,tokenizer,Sample_Circled_Letter)
result_cl = [item.strip() for item in result_cl]
count = 0
for x,y in zip(result_cl,Sample_Circled_Letter['groundtruth']):
    if (x==y):
        count+=1
print("Circled Letters Accuracy is = ", count/len(result_cl)*100)

### Touching Circles 
Sample_Touching_Circles = ds['Touching_Circles'].select(range(200))
result_tc = generate(model,processor,tokenizer,Sample_Touching_Circles)
count = 0
for x,y in zip(result_tc,Sample_Touching_Circles['groundtruth']):
    if (x=='yes' and y=='Yes') or (x=='no' and y=='No'):
        count+=1
print("Touching Circles Accuracy is = ", count/len(result_tc)*100)

### Line Intersecting
Sample_Line_Intersecting = ds['Line_Plot_Intersections'].select(range(200))
prompt = "Are the two line plots intersecting? Answer with Yes/No. "
result_lp = generate(model,processor,tokenizer,Sample_Line_Intersecting,prompt)
count = 0
for x,y in zip(result_lp,Sample_Line_Intersecting['groundtruth']):
    if (x=="yes" and y!=0 or x=="no" and y==0):
        count+=1
print("Line Intersecting Accuracy is = ", count/len(result_lp)*100)

### No. of olympic rings

Sample_Olympic_Ring = ds['Olympic_Counting_Circles'].select(range(200))
prompt = "Count the total number of circles in the image? "
result_or = generate(model,processor,tokenizer,Sample_Olympic_Ring)
print("olympic rings Accuracy is = ", get_accuracy(result_or,Sample_Olympic_Ring['groundtruth']))

### No. of Line of Intersections
Sample_Line_Point_Int = ds['Line_Plot_Intersections'].select(range(200))
prompt = "Count the number of points of intersection of red and blue lines?"
result_lpi = generate(model,processor,tokenizer,Sample_Line_Point_Int,prompt)
print("Line of Intersections Accuracy is = ", get_accuracy(result_lpi,Sample_Line_Point_Int['groundtruth']))

### Count Rows and Cols
Sample_Rows = ds['Counting_Grid_Blank_Grids'].select(range(200))
prompt = "Count number of rows and number columns in grid "
result_cr = generate(model,processor,tokenizer,Sample_Rows,prompt)
extracted_numbers = [extract_numbers(s) for s in result_cr if extract_numbers(s)]
print("Rows and Cols Count Accuracy is = ", get_accuracy(extracted_numbers,Sample_Rows['groundtruth']))

### Nested Squares

Sample_Nested = ds['Nested_Squares'].select(range(200))
prompt = "Count total number of squares in the image."
result_ns = generate(model,processor,tokenizer,Sample_Nested,prompt)
print("Nested Squares Accuracy is = ",get_accuracy(result_ns, Sample_Nested['groundtruth']))

### Subway Conncections 
Sample_Subway = ds['Subway_Connections'].select(range(200))
prompt = "Count number of subway connections"
result_sc = generate(model,processor,tokenizer,Sample_Subway,prompt)
print("Subway Conncections Accuracy is = ",get_accuracy(result_sc, Sample_Subway['groundtruth']))

