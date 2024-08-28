import transformers
import torch

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

Questions = ["Count the intersection points where the blue and red lines meet ","Are the lines intersecting","Count the total number of squares in the image.","How many shapes are in the image","Which character is being highlighted with a red oval?","How many single-colored paths go from A to C?","Count the number of rows and columns.","Are the two circles overlapping? Answer with Yes/No."]
for question in Questions:
    messages = [
        {"role": "system", "content": "You will be given some questions you need to just extract most important 1-2 word keywords from the question.JUST RETURN THEE KEYWORDS AND NOTHING ELSE"},
        {"role": "user", "content": question},
        
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=256,
    )
    print(outputs[0]["generated_text"][-1])
