# Model Test Runner
### CTCS 2024/25 - UIBK

Investigating Text Processing Strategies for Decreasing Token Count in LLM Inputs


## Topic Requirements

Pick 1-2 tasks with their datasets where each request to LLM contains relatively large amount of text:

- QA, summarization, sentiment estimation, story understanding, etc.
- Choose optimal prompts

Experiment with some of the potential, diverse approaches for decreasing token count:

- [x] Random word removal
- [x] Short word removal
- [ ] Redundant word removal (Arno)
- [x] Stop word removal
- [ ] Text simplification (using existing library) (Bernd)
- [ ] Shortening words by abbreviating them (Sonja)
- [x] Any other idea…
- Measure the token count save vs. the accuracy drop


## Goals

### Sentiment Analysis
- Dataset: https://www.kaggle.com/datasets/priyamchoksi/rotten-tomato-movie-reviews-1-44m-rows
- Models: ChatGPT (& another one from HuggingFace)
- Validation: use `scoreSentiment` from dataset to compare to our results 

### Question-Answering
- Dataset: SimpleQuestion dataset from https://github.com/ad-freiburg/large-qa-datasets?tab=readme-ov-file#simplequestions
- Models: ChatGPT & facebook/blenderbot-3B
- Validation: ask ChatGPT if the questions have been answered correctly

- Alternatively: use prompts from https://github.com/jujumilk3/leaked-system-prompts/

### Spam Detection
- Dataset: https://www.kaggle.com/datasets/ashfakyeafi/spam-email-classification
- Models: ChatGPT & mrm8488/bert-tiny-finetuned-enron-spam-detection
- Validation: 


## Running the test runner

`⚠️ Note: always make sure you are in the root directory of the project`

It is recommended to use a Python virtual environment.
The following should set it up, activate it and install the requirements.

```
python3 -m venv .venv
source ./.venv/bin/activate
pip3 install -r requirements.txt # this may take some time
```

After setting up all of that you can run it by simply calling `main.py`.
To install Cuda on Windows you need to download the CudaToolKit https://developer.nvidia.com/cuda-downloads and set up Torch for your system https://pytorch.org/get-started/locally/
If you don't have Cuda you should also be able to run on CPU which you can do by simply not passing any argument for `--device`.

```
python3 main.py --device cuda
```
OR
```
python3 main.py --device cpu # cpu is default
```

When you run the program like that you will notice that every result is directly printed to the console.
Since this is not very useful for evaluating the results it is also possible to store them in JSON files.

Simply put there are three modes:

```
python3 main.py --mode print # print is default, prints results to the console
```
OR
```
python3 main.py --mode persistent # writes results to JSONs in the "data/results" directory
```
OR
```
python3 main.py --mode persistent-skip # writes results to JSONs in the "data/results" directory, skipping existing results
```


## Using ChatGPT

In order to use the OpenAI API you need to add a ".env" file with you API key.

```
CHATGPT_API_SECRET=YOUR_SECRET_KEY
```

You can set this up on https://platform.openai.com/ -
if you don't specify an API key the test runner will simply skip the ChatGPT models and only run the local ones.


## Adding new models

Please make sure your model class is placed in the `models` package and extends the `BaseModel` class.
The `BaseModel` class is rather simple to allow flexibility as it only specifies two functions:

- `__init__()` could initialize a library (e.g. HuggingFace) or an API endpoint (e.g. ChatGPT) or whatever it needs to run 
- `run(prompt)` takes a prompt and runs it using whatever it set up in the `__init__` function

In order to add models to the test runner simply put them into the models dictionary in the `__init__` of the `TestRunner` class.

This is purposefully done in the `__init__` and not the class definition since it allows to pass parameters like `device` for the HuggingFace models.


## Adding new processors

Please make sure your processor class is placed in the `textprocessors` package and extends the `BaseProcessor` class.
The `BaseProcessor` class is rather simple to allow flexibility as it only specifies two functions:

- `__init__()` could initialize a library (e.g. nltk) or whatever it needs to run 
- `tokenize(prompt)` takes a prompt and returns a processed version of it using whatever it set up in the `__init__` function

In order to add processors to the test runner simply put them into the processors dictionary in the `__init__` of the `TestRunner` class.

This is purposefully done in the `__init__` and not the class definition since it allows to pass parameters (even though that isn't used currently).
