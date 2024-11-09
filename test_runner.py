from safetensors import torch

from models import HuggingFaceModel
from textprocessors import StemmingProcessor, LemmatizationProcessor


def pretty_print_answer(method, prompt, answer):
    print(f"- {method}")
    print(f"\tPrompt:")
    print(f"\t{prompt}")
    print(f"\tAnswer:")
    print(f"\t{answer}")


class TestRunner:

    # mode in which the tests should run
    # "print" just prints the results to the console
    # "persistent" saves output to files
    mode = "print"

    # dictionary of model names and instantiated models
    models = {}

    # processors, currently hard coded but could be list
    stemmer = StemmingProcessor()
    lemmatizer = LemmatizationProcessor()

    def __init__(self, mode="print", device="cpu"):
        self.mode = mode
        self.models["facebook/blenderbot-400M-distill"] = HuggingFaceModel(
            task="text2text-generation",
            model="facebook/blenderbot-400M-distill",
            device=device
        )

    def run_tests(self, prompt):
        processed_prompts = {
            "stemmer": self.stemmer.tokenize(prompt),
            "lemmatizer": self.lemmatizer.tokenize(prompt),
        }

        for model_name, model in self.models.items():
            print(f"Running model '{model_name}':")
            answer = model.run(prompt)
            pretty_print_answer("unmodified", prompt, answer)
            for processor, processed_prompt in processed_prompts.items():
                answer = model.run(processed_prompt)
                pretty_print_answer(processor, processed_prompt, answer)
