# Model Test Runner
### CTCS 2024/25 - UIBK

Investigating Text Processing Strategies for Decreasing Token Count in LLM Inputs

## Topic Requirements

Pick 1-2 tasks with their datasets where each request to LLM contains relatively large amount of text:

- QA, summarization, sentiment estimation, story understanding, etc.
- Choose optimal prompts

Experiment with some of the potential, diverse approaches for decreasing token count:

- Random word removal
- Short word removal
- Redundant word removal
- Stop word removal
- Text simplification (using existing library)
- Shortening words by abbreviating them
- Any other idea…
- Measure the token count save vs. the accuracy drop

## Running the test runner

It is recommended to use a Python virtual environment.
The following should set it up, activate it and install the requirements.

```
python3 -m venv .venv
source ./.venv/bin/activate
pip3 install -r requirements.txt # this may take some time
```

After setting up all of that you can run it by simply calling `main.py`.
If you don't have Cuda you can also use Torch but the models should be able to run on CPU which you can do by simply not passing any argument for `--device`.

```
python3 main.py --device cuda
```
OR
```
python3 main.py --device torch
```
OR
```
python3 main.py --device cpu
```

## Using ChatGPT

In order to use the OpenAI API you need to add a ".env" file with you API key.

```
CHATGPT_API_SECRET=YOUR_SECRET_KEY
```

You can set this up on https://platform.openai.com/ -
if you don't specify an API key the test runner will simply skip the ChatGPT models and only run the local ones.



## Adding new models

// TODO
