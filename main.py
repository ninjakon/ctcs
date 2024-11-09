import argparse

from test_runner import TestRunner


def main():
    parser = argparse.ArgumentParser("Model Test Runner")
    parser.add_argument("--device", help="The device to run HuggingFace models on.", type=str)
    args = parser.parse_args()

    prompt = "Who is the president of the USA?"
    test_runner = TestRunner(device=args.device)
    test_runner.run_tests(prompt)


if __name__ == '__main__':
    main()
