import argparse
import os.path
import torch
from os import listdir
from os.path import isfile, join

from testrunner import TestRunner


DEFAULT_MODE = "print"
DEFAULT_DEVICE = "cpu"


def process_arguments():
    print("Welcome to the model test runner!")

    parser = argparse.ArgumentParser("Model Test Runner")
    parser.add_argument("--mode", help="Whether to print results or write them to files.", type=str)
    parser.add_argument("--device", help="The device to run HuggingFace models on.", type=str)
    args = parser.parse_args()

    mode = args.mode
    device = args.device

    if mode is None:
        print("> You did not specify a mode ('print' | 'persistent' | 'persistent-skip').")
        print(f"> So by default the pipeline will use '{DEFAULT_MODE}'.")
        mode = DEFAULT_MODE
    else:
        print(f"> Running in '{mode}' mode as specified.")
    if device is None:
        print("> You did not specify a device to run HuggingFace models on.")
        print(f"> So by default the pipeline will use '{DEFAULT_DEVICE}'.")
        device = "cpu"
    else:
        print(f"> The device to run HuggingFace models on is '{device}' as specified.")
        if device == "cuda" and not torch.cuda.is_available():
            print("> You tried to set the device to 'cuda' but it is not available.")
            print("> Switching to 'cpu' instead.")
            device = "cpu"
    return mode, device


def read_prompt_files(prompt_path="./data/prompts"):
    prompt_file_paths = [
        os.path.join(prompt_path, f) for f in listdir(prompt_path) if isfile(join(prompt_path, f))
    ]
    prompts = {}
    for prompt_file_path in prompt_file_paths:
        with open(prompt_file_path) as prompt_file:
            prompt_file_name = os.path.basename(prompt_file_path)
            prompts[prompt_file_name] = prompt_file.read()
    return prompts


def main():
    mode, device = process_arguments()
    prompts = read_prompt_files()
    test_runner = TestRunner(mode=mode, prompts=prompts, device=device)
    test_runner.run_tests()


if __name__ == '__main__':
    main()
