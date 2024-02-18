import argparse
from experiment import DEFAULT_EXPERIMENTS, ExperimentFramework
from model import ModelType

def main():
    parser = argparse.ArgumentParser(description='Load the needed datasets')
    parser.add_argument(
        '-m',
        '--model',
        default=None,
        action='store',
        help='Specifies the model to test.',
    )
    parser.add_argument(
        '-s',
        '--subject',
        default=None,
        action='store',
        help='(optional) Specifies the subject for experimental test.'
    )
    parser.add_argument(
        '-e',
        '--experiment',
        default=DEFAULT_EXPERIMENTS,
        action='store',
        help='Experiments path'
    )
    args = parser.parse_args()
    match args.model:
        case 'liquid':
            model_type = ModelType.ONLY_LIQUID
        case 'conv':
            model_type = ModelType.ONLY_CONV
        case 'lstm':
            model_type = ModelType.CONV_LSTM
        case 'conv_liquid':
            model_type = ModelType.CONV_LIQUID
        case None:
            model_type = None
        case _:
            raise ValueError('the --model value muest be one of liquid, conv, lstm or conv_liquid')

    print(f'>>>>>>>> Model type selected: {model_type}')
    framework = ExperimentFramework(model_type=model_type, experiments_path=args.experiment)
    framework.start(subject=args.subject)

if __name__ == '__main__':
    main()