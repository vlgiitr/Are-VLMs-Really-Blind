# Are VLMs Really Blind

Vision-Language Models excel at complex tasks like OCR, VQA, and geometric reasoning but struggle with simple visual tasks that humans find easy. This work explores whether their limitations in geometric reasoning are inherent or can be improved. We propose an automatic pipeline that enhances image understanding by generating question-based keywords. These keywords are used to generate a caption highlighting relevant image details, which is then processed by a language model to deliver precise answers without requiring additional fine-tuning.

## Overview

### Initial Experiment - Zero-Shot Question Answering:
- In the first experiment, we evaluated the  Vision-Language Model (VLM) on zero-shot question answering .
- The model was provided only with an image and a corresponding question, without any prior fine-tuning or additional information.
- This setup tested the model’s innate ability to answer the question based solely on the visual input and the posed question.

### Enhanced Approach - Keyword-Based Captioning and Question Answering
- In the second experiment, we used LLaMA, to extract important keywords from the question. These keywords highlighted the essential elements that the model should focus on in the image.
- We then used the extracted keywords to prompt the model to generate a caption that describes the image.
- The caption, now acting as a detailed textual summary of the image, was fed into a language model to answer the original question.

## Task 
- Counting the number of intersections between 2 lines(Task 1)
- Checking whether two lines intersect(Task 2) 
- Counting the number of nested squares in an image(Task 3)
- Counting the number of Olympic rings in an image. (Task 4)
- Finding the letter circled in red in a word(Task 5)
- Counting the number of paths in the subway line task(Task 6) 
- Finding the number of rows and columns in a grid(Task 7)
- Determining if 2 circles are intersecting(Task 8)

## Results 
| Model      | Task 1 QnA | Task 1 QnA+Captions | Task 2 QnA | Task 2 QnA+Captions | Task 3 QnA | Task 3 QnA+Captions | Task 4 QnA | Task 4 QnA+Captions | Task 5 QnA | Task 5 QnA+Captions | Task 6 QnA | Task 6 QnA+Captions | Task 7 QnA | Task 7 QnA+Captions | Task 8 QnA | Task 8 QnA+Captions | Average QnA | Average QnA+Captions |
|------------|------------|--------------------|------------|--------------------|------------|--------------------|------------|--------------------|------------|--------------------|------------|--------------------|------------|--------------------|------------|--------------------|--------------|----------------------|
| **Gemini** | 40         | 43                 | 78         | 88                 | 49         | 51                 | 24         | 26 | 33               | 41         | 15                 | 14         | 13  | 15                | 65         | 74                 | 39.63   | **44.00**           |
| **Paligemma** | 3        | 49                 | 90         | 84                 | 60         | 40                 | 21         | 34  |3               | 34         | 0                  | 24         | 6       |6       |37.5    | 55              | 27.56  | **40.75**           |
| **GPT4omni** | 46       | 71                 | 58         | 96                 | 18         | 41                 | 39         | 46      |40           | 56         | 22                 | 21         | 48     |54            | 80         | 79                 | 43.88  | **58.00**           |


## Installation
```
git clone https://github.com/vlgiitr/Are-VLMs-Really-Blind.git
```

## Contributing
Contributions are welcome!

## License
See the LICENSE file for details.






