from utils import *
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
###### Counting ###########
#Number of Intersection
api = "YOUR_API_KEY"
os.environ["OPENAI_API_KEY"] = api
client = OpenAI()
ds = load_dataset("Ayush-Singh/llms-are-blind-captions_florence2-large")
Sample_Line_Point_Int = ds['Line_Plot_Intersections'].select(range(100))
result_lpi = generate(client,Sample_Line_Point_Int)
results = [re.findall(r'\d+', item)[0] for item in result_lpi]
print("Intersection Accuracy is = ", get_accuracy(results,Sample_Line_Point_Int['groundtruth']))
# Counting Cirlces
Sample_Olympic_Ring = ds['Olympic_Counting_Circles'].select(range(100))
result_or = generate(client,Sample_Olympic_Ring)
results = [re.findall(r'\d+', item)[0] for item in result_or]
print("Counting circles accuracy is = ",get_accuracy(results,Sample_Olympic_Ring['groundtruth']))
# Number of rows/columns
Sample_Rows = ds['Counting_Grid_Blank_Grids'].select(range(100))
result_cr = generate(client,Sample_Rows)
new_data = [extract_row_column(item) for item in result_cr]
print("Rows and Cols Counting Accuracy is = ",get_accuracy(new_data,Sample_Rows['groundtruth']))
# Subway Connections
Sample_Subway = ds['Subway_Connections'].select(range(100))
result_sc = generate(client,Sample_Subway)
results = [re.findall(r'\d+', item)[0] if re.findall(r'\d+', item) else -1 for item in result_sc]
print("Subway Connections Accuracy is = ",get_accuracy(results,Sample_Subway['groundtruth']))
# Nested Squares
Sample_Nested = ds['Nested_Squares'].select(range(100))
result_ns = generate(client,Sample_Nested)
results = [re.findall(r'\d+', item)[0] for item in result_ns]
print("Nested Squares Accuracy is = ",get_accuracy(results,Sample_Nested['groundtruth']))
########### Geometry ##############
# Touching circles
Sample_Touching_Circles = ds['Touching_Circles'].select(range(100))
result_tc = generate(client,Sample_Touching_Circles)
cleaned_data = [re.sub(r'\.$', '', item) for item in result_tc]
print("Geometry Accuracy is = ",get_accuracy(cleaned_data,Sample_Touching_Circles['groundtruth']))
# Are lines intersecting
Sample_Line_Intersecting = ds['Line_Plot_Intersections'].select(range(100))
prompt = "Are the two lines intersecting. Answer in just a single word Yes/No."
result_li = generate(client,Sample_Line_Intersecting,prompt)
results_lp = generate(client,Sample_Line_Intersecting,prompt)
cleaned_data = [re.sub(r'\.$', '', item) for item in results_lp]
count = 0
for x,y in zip(results_lp,Sample_Line_Intersecting['groundtruth']):
    if (x=='Yes' and y!='0') or (x=='No' and y==0):
        count+=1
print("Lines intersecting accuracy is ", count/len(results_lp)*100)
## Circled Letters
Sample_Circled_Letter = ds['Circled_Letter'].select(range(100))
results_cl = generate_circled_letters(client,Sample_Circled_Letter)
print("Circled Letters Accurcy is = ", get_accuracy(results_cl,Sample_Circled_Letter['groundtruth']))

########## CAPTION ##############
# Circles touching 
prompt = "Caption on the position of each circle with respect to each other. "
result_tc = generate_cqa(client,Sample_Touching_Circles,prompt)
cleaned_data = [re.sub(r'\.$', '', item) for item in result_tc]
print("Touching Circles Caption Accuracy is = ",get_accuracy(cleaned_data,Sample_Touching_Circles['groundtruth']))
## Intersecting Lines
prompt = "Describe whether the two lines are intersecting or not"
results_lp = generate_cqa(client,Sample_Line_Intersecting,prompt)
cleaned_data = [re.sub(r'\.$', '', item) for item in results_lp]
print("Circled Letters Accurcy is = ", get_accuracy(results_cl,Sample_Line_Intersecting['groundtruth']))
## Circled Letters
prompt = "Caption the image focusing on what letter is in the red oval"
results_cl = generate_mod_cqa(client,Sample_Circled_Letter,prompt)
print("Word Oval Accuracy is = ",get_accuracy(results_cl,Sample_Circled_Letter['groundtruth']))

## Counting
## Number of Circles
prompt = "Caption the image describing the total number of circles in the image? "
result_or = generate_cqa(client,Sample_Olympic_Ring,prompt)
results = [re.findall(r'\d+', item)[0] for item in result_or]
print("Number of Circles Accuracy is = ",get_accuracy(results,Sample_Olympic_Ring['groundtruth']))
## Line Intersections
prompt = "Caption the image describing the intersection of blue and red line plots. "
result_lpi = generate_cqa(client,Sample_Line_Point_Int,prompt)
results = [re.findall(r'\d+', item)[0] for item in result_lpi]
print("Line Intersections Accuracy is ",get_accuracy(results,Sample_Line_Point_Int['groundtruth']))

## Counting Grids
prompt = "Caption on the number of rows and number of columns in the image of grid. "
result_cr = generate_cqa(client,Sample_Rows,prompt)
new_data = [extract_row_column(item) for item in result_cr]
results = [re.search(r'\{([^}]*)\}', item).group(1) if re.search(r'\{([^}]*)\}', item) else "NA" for item in results_cr]
print("Grid Accuracy is = ",get_accuracy(new_data,Sample_Rows['groundtruth']))

## Subway Count 
prompt = "Describe the paths connecting 4 subway stations located at the 4 edge centers of a square."
result_sc = generate_cqa(client,Sample_Subway,prompt)
results = [re.findall(r'\d+', item)[0] if re.findall(r'\d+', item) else -1 for item in result_sc]
print("Subway Station Accuracy is = ",get_accuracy(results,Sample_Subway['groundtruth']))

## Nested Squares 
prompt = "Describe the total number of squares in the image. "
result_ns = generate(client,Sample_Nested,prompt)
results = [re.findall(r'\d+', item)[0] for item in result_ns]
print("Nested Square Accuracy is = ",get_accuracy(results,Sample_Nested['groundtruth']))
