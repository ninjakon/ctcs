import os
import json
from dotenv import dotenv_values


def get_openai_key():
    try:
        openai_api_key = dotenv_values(".env")['CHATGPT_API_SECRET']
        return openai_api_key
    except KeyError:
        return None


def print_model_answer(processor, prompt, answer):
    print(f"- {processor}")
    print(f"\tPrompt:")
    print(f"\t{prompt}")
    print(f"\tAnswer:")
    print(f"\t{answer}")


def write_model_answer_to_json_file(result_dir, processor, prompt, answer):
    result_dict = {
        "processor": processor,
        "prompt": prompt,
        "answer": answer
    }
    json_object = json.dumps(result_dict, indent=4)
    json_path = os.path.join(result_dir, f"{processor}")
    with open(f"{json_path}.json", "w") as json_file:
        json_file.write(json_object)


def add_test_if_missing(missing_tests, json_path, prompt_file_name, processor_name, model_name):
    # if the file exists the test does not need to run again
    if os.path.isfile(json_path):
        print(f"> Found '{json_path}'")
        print(f"> - Skipping processing prompt '{prompt_file_name}' with '{processor_name}' for model '{model_name}'.")
    # if the file exists declare the combination of prompt, processor and model as missing
    else:
        print(f"> Could not find '{json_path}'")
        print(f"> - Test for '{prompt_file_name}' with '{processor_name}' for model '{model_name}' will be repeated.")
        missing_test = (processor_name, model_name)
        if prompt_file_name not in missing_tests.keys():
            missing_tests[prompt_file_name] = [missing_test]
        else:
            missing_tests[prompt_file_name].append(missing_test)


def get_missing_tests(data_dir, prompts, models, processors):
    missing_tests = {}
    # iterate over all prompts
    for prompt_file_name, prompt in prompts.items():
        # iterate over all models
        for model_name, model in models.items():
            print(f"Checking results for model '{model_name}' and prompt '{prompt_file_name}'...")
            result_dir = str(os.path.join(data_dir, model_name, prompt_file_name))
            # results are checked by looking for "DATA_DIR/model-name/prompt-name/processor.json"
            for processor_name in processors.keys():
                add_test_if_missing(
                    missing_tests=missing_tests,
                    json_path=os.path.join(result_dir, f"{processor_name}.json"),
                    prompt_file_name=prompt_file_name,
                    processor_name=processor_name,
                    model_name=model_name
                )
            # also check if results for the unmodified prompt exist
            add_test_if_missing(
                missing_tests=missing_tests,
                json_path=os.path.join(result_dir, "unmodified.json"),
                prompt_file_name=prompt_file_name,
                processor_name="unmodified",
                model_name=model_name
            )
    return missing_tests
