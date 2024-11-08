from test_runner import TestRunner


def main():
    prompt = "Who is the president of the USA?"
    test_runner = TestRunner()
    test_runner.run_tests(prompt)


if __name__ == '__main__':
    main()
