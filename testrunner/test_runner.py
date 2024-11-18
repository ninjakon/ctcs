import os
import shutil

from models import HuggingFaceModel, ChatGPTModel
from textprocessors import (
    StemmingProcessor, LemmatizationProcessor, RandomRemovalProcessor,
    ShortWordRemovalProcessor, StopWordRemovalProcessor
)
from .utils import (
    get_openai_key, print_model_answer, write_model_answer_to_json_file, get_missing_tests
)


DATA_DIR = "./data/results"


class TestRunner:

    # mode in which the tests should run
    # "print" just prints the results to the console
    # "persistent" saves output to files
    # "persistent-skip" saves output to files, skips models/processors if result already exists
    mode = "print"

    # dictionary of model names and instantiated models, initialized in __init__
    models = {}

    # dictionary of processor names and instantiated processors, initialized in __init__
    processors = {}

    def __init__(self, mode="print", prompts=None, device="cpu"):
        self.mode = mode
        self.prompts = prompts if prompts else {}

        # Dev Note: add new processors here
        self.processors["stemmer"] = StemmingProcessor()
        self.processors["lemmatizer"] = LemmatizationProcessor()
        self.processors["random_remover"] = RandomRemovalProcessor()
        self.processors["short_word_remover"] = ShortWordRemovalProcessor()
        self.processors["stop_word_remover"] = StopWordRemovalProcessor()

        # Dev Note: add new models here
        self.models["facebook/blenderbot-3B"] = HuggingFaceModel(
            task="text2text-generation",
            model="facebook/blenderbot-3B",
            device=device
        )
        openai_api_key = get_openai_key()
        if openai_api_key is not None:
            self.models["ChatGPT/gpt-3.5-turbo"] = ChatGPTModel(
                model="gpt-3.5-turbo",
                openai_api_key=openai_api_key
            )
        else:
            print("No OpenAI API key found - Skipping ChatGPT model(s)")

    def run_tests(self):
        if self.mode == "print":
            self.__test_in_print_mode()
        elif self.mode == "persistent":
            self.__test_in_persistent_mode()
        elif self.mode == "persistent-skip":
            self.__test_in_persistent_skip_mode()
        else:
            print("Invalid mode")

    def __test_in_print_mode(self):
        # iterate over all prompts
        for prompt_file_name, prompt in self.prompts.items():
            # process prompts first since it is not necessary to repeat that for each model
            processed_prompts = {"unmodified": prompt}
            for processor_name, processor in self.processors.items():
                processed_prompts[processor_name] = processor.tokenize(prompt)
            # iterate over all models and run the processed prompts through them
            for model_name, model in self.models.items():
                print(f"Running model '{model_name}'...")
                # run the processed prompts through the model and print the results
                for processor_name, processed_prompt in processed_prompts.items():
                    answer = model.run(processed_prompt)
                    print_model_answer(processor_name, processed_prompt, answer)

    def __test_in_persistent_mode(self):
        # iterate over all prompts
        for prompt_file_name, prompt in self.prompts.items():
            # process prompts first since it is not necessary to repeat that for each model
            processed_prompts = {"unmodified": prompt}
            for processor_name, processor in self.processors.items():
                processed_prompts[processor_name] = processor.tokenize(prompt)
            # iterate over all models and run the processed prompts through them
            for model_name, model in self.models.items():
                print(f"Running model '{model_name}'...")
                # remove result directory if it exists since this mode re-runs all tests
                result_dir = os.path.join(DATA_DIR, model_name, prompt_file_name)
                if os.path.exists(result_dir):
                    shutil.rmtree(result_dir)
                os.makedirs(result_dir)
                # run the processed prompts through the model and save the results to json
                for processor_name, processed_prompt in processed_prompts.items():
                    answer = model.run(processed_prompt)
                    write_model_answer_to_json_file(
                        result_dir=result_dir,
                        processor=processor_name,
                        prompt=processed_prompt,
                        answer=answer,
                    )

    def __test_in_persistent_skip_mode(self):
        # first check which tests need to be repeated
        missing_tests = get_missing_tests(
            data_dir=DATA_DIR,
            prompts=self.prompts,
            models=self.models,
            processors=self.processors
        )
        # iterate over all missing tests
        for prompt_file_name, missing_test in missing_tests.items():
            # always run the "unmodified" prompt for sanity reasons
            prompt = self.prompts[prompt_file_name]
            processed_prompts = {}
            # process prompts (with missing processors) first since it is not necessary to repeat that for each model
            missing_processors = [p for p, _ in missing_test]
            for processor_name in missing_processors:
                if processor_name not in processed_prompts.keys():
                    if processor_name == "unmodified":
                        processed_prompts["unmodified"] = prompt
                    else:
                        processed_prompts[processor_name] = self.processors[processor_name].tokenize(prompt)
            # iterate over missing models and run missing prompts through them
            missing_models = [m for _, m in missing_test]
            for model_name in missing_models:
                print(f"Running model '{model_name}'...")
                # make sure the result directory exists and create it if not
                result_dir = os.path.join(DATA_DIR, model_name, prompt_file_name)
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)
                # run the missing prompts through the model and save the results to json
                for processor_name, processed_prompt in processed_prompts.items():
                    if (processor_name, model_name) in missing_test:
                        model = self.models[model_name]
                        answer = model.run(processed_prompt)
                        write_model_answer_to_json_file(
                            result_dir=result_dir,
                            processor=processor_name,
                            prompt=processed_prompt,
                            answer=answer,
                        )
