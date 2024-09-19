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
from utils import get_accuracy,generate_cqa

os.environ["GOOGLE_API_KEY"]="YOUR_API_KEY"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
login("YOUR_API_KEY")
ds = load_dataset("Ayush-Singh/llms-are-blind-captions_florence2-large")
model_qa = genai.GenerativeModel(model_name="gemini-1.5-flash")
model_id = "google/paligemma-3b-mix-448"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id).cuda().eval()
processor = AutoProcessor.from_pretrained(model_id)

####### QA With Captions############
#### Sample_Olympic_Counting_Circles
Sample_Olympic_Counting_Circles = ds['Olympic_Counting_Circles'].select(range(200))
prompt = "caption on the geometry of number of circles in the image en"
results_occ = generate_cqa(model,model_qa,processor,Sample_Olympic_Counting_Circles,prompt)
results_occ = [re.findall(r'\d+', item)[0] for item in results_occ]
print("Counting Circles Accuracy is : ", get_accuracy(results_occ,Sample_Olympic_Counting_Circles['groundtruth']))

#### Nested Squares
Sample_Nested_Squares = ds['Nested_Squares'].select(range(200))
prompt = "describe on the count and geometry of squares in the image "
results_ns = generate_cqa(model,model_qa,processor,Sample_Nested_Squares,prompt)
results_ns = [re.findall(r'\d+', item)[0] for item in results_ns]
print("Nested Squares Accuracy is : ", get_accuracy(results_ns,Sample_Nested_Squares['groundtruth']))

##### Line_Plot_Intersections
Sample_Line_Plot_Intersections = ds['Line_Plot_Intersections'].select(range(200))
prompt = "describe the geometry of intersection of blue and red line in image"
results_lpi = generate_cqa(model,model_qa,processor,Sample_Line_Plot_Intersections,prompt)
results_lpi = [re.findall(r'\d+', item)[0] for item in results_lpi]
print("Line Plot Intersections Accuracy is : ", get_accuracy(results_lpi,Sample_Line_Plot_Intersections['groundtruth']))

###### Sample_Subway_Connections

Sample_Subway_Connections = ds['Subway_Connections'].select(range(200))
prompt = "prompt : describe in detail on path connecting differnt stations in the image en" 
results_sc = generate_cqa(model,model_qa,processor,Sample_Subway_Connections,prompt)
results_sc = [re.findall(r'\d+', item)[0] if re.findall(r'\d+', item) else -1 for item in results_sc]
print("Subway Connections Accuracy is : ", get_accuracy(results_sc,Sample_Subway_Connections['groundtruth']))

##### Sample_Counting_Grid_Blank_Grids

Sample_Counting_Grid_Blank_Grids = ds['Counting_Grid_Blank_Grids'].select(range(200))
prompt = "describe the count rows and columns in table correctly "
results_cg = generate_cqa(model,model_qa,processor,Sample_Counting_Grid_Blank_Grids,prompt) 
results_cg = [re.search(r'\{([^}]*)\}', item).group(1) if re.search(r'\{([^}]*)\}', item) else "NA" for item in results_cg]
print("Counting Grid Blank Grids Accuracy is : ", get_accuracy(results_cg,Sample_Counting_Grid_Blank_Grids['groundtruth']))

#### Line_Plot_Intersections

Sample_Lines = ds['Line_Plot_Intersections'].select(range(200))
prompt = "describe the geometry of intersection of blue and red line in image "
results_l = generate_cqa(model,model_qa,processor,Sample_Lines,prompt)
results_l = [re.search(r'\{(Yes|No)\}', item).group(1) if re.search(r'\{(Yes|No)\}', item) else NA for item in results_l]
count = 0
for x,y in zip(results_l,Sample_Lines['groundtruth']):
    if (x=='Yes' and y!='0') or (x=='No' and y==0):
        count+=1
print("Line Plot Intersections Accuracy is ", count/len(results_l)*100)

##### Touching_Circles
Sample_Touching_Circles = ds['Touching_Circles'].select(range(200))
prompt = "caption on the geometry of two circles in the image en"
results_tc = generate_cqa(model,model_qa,processor,Sample_Touching_Circles,prompt)    
results_tc = [re.search(r'\{(Yes|No)\}', item).group(1) if re.search(r'\{(Yes|No)\}', item) else NA for item in results_tc]
print("Touching Circles Accuracy is : ", get_accuracy(results_tc,Sample_Touching_Circles['groundtruth']))

####### Circled_Letter

Sample_Circled_Letter = ds['Circled_Letter'].select(range(200))
prompt = "caption on what is the letter within the red oval in the text in the image "
results_cl = generate_cqa(model,model_qa,processor,Sample_Circled_Letter,prompt)
extracted_chars = [re.sub(r'\s+', '', item) for item in results_cl]
print("Circled Letter Accuracy is : ", get_accuracy(extracted_chars,Sample_Circled_Letter['groundtruth']))

