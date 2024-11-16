import argparse

from test_runner import TestRunner


def main():
    parser = argparse.ArgumentParser("Model Test Runner")
    parser.add_argument("--device", help="The device to run HuggingFace models on.", type=str)
    args = parser.parse_args()
    device = args.device

    print("Welcome to the model test runner!")
    if device is None:
        print("> You did not specify a device to run HuggingFace models on.")
        print("> So by default the pipeline will use CPU.")
    else:
        print(f"> The device to run HuggingFace models on is '{device}' as specified.")

    prompt = "Who is the president of the USA?"
    test_runner = TestRunner(device=args.device)
    test_runner.run_tests(prompt)


if __name__ == '__main__':
    main()
