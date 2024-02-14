import argparse
from experiment import ExperimentFramework

def main():
    parser = argparse.ArgumentParser(description='Load the needed datasets')
    parser.add_argument(
        '-m',
        '--model',
        default=None,
        action='store',
        help='Specifies the model to test.',
    )  
    args = parser.parse_args()

    framework = ExperimentFramework()
    framework.start()

if __name__ == '__main__':
    main()