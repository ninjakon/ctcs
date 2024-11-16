from models import HuggingFaceModel, ChatGPTModel
from textprocessors import StemmingProcessor, LemmatizationProcessor, RandomRemovalProcessor, ShortWordRemovalProcessor, StopWordRemovalProcessor
from dotenv import dotenv_values


def pretty_print_answer(method, prompt, answer):
    print(f"- {method}")
    print(f"\tPrompt:")
    print(f"\t{prompt}")
    print(f"\tAnswer:")
    print(f"\t{answer}")


def get_openai_key():
    try:
        openai_api_key = dotenv_values(".env")['CHATGPT_API_SECRET']
        return openai_api_key
    except KeyError:
        return None


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
    random_remover = RandomRemovalProcessor()
    short_word_remover = ShortWordRemovalProcessor()
    stop_word_remover = StopWordRemovalProcessor()

    def __init__(self, mode="print", device="cpu"):
        self.mode = mode
        self.models["facebook/blenderbot-400M-distill"] = HuggingFaceModel(
            task="text2text-generation",
            model="facebook/blenderbot-400M-distill",
            device=device
        )
        openai_api_key = get_openai_key()
        if openai_api_key is not None:
            self.models["gpt-3.5-turbo"] = ChatGPTModel(
                model="gpt-3.5-turbo",
                openai_api_key=openai_api_key
            )
        else:
            print("No OpenAI API key found - Skipping ChatGPT model(s)")

    def run_tests(self, prompt):
        processed_prompts = {
            "stemmer": self.stemmer.tokenize(prompt),
            "lemmatizer": self.lemmatizer.tokenize(prompt),
            "random_remover": self.random_remover.tokenize(prompt),
            "short_word_remover": self.short_word_remover.tokenize(prompt),
            "stop_word_remover": self.stop_word_remover.tokenize(prompt)
        }

        for model_name, model in self.models.items():
            print(f"Running model '{model_name}':")
            answer = model.run(prompt)
            pretty_print_answer("unmodified", prompt, answer)
            for processor, processed_prompt in processed_prompts.items():
                answer = model.run(processed_prompt)
                pretty_print_answer(processor, processed_prompt, answer)
