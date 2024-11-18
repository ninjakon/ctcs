import os
import json


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
