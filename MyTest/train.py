from datasets import create_dataset
import argparse
import os
from utils import load_yaml


def main(config_file):
    configuration = load_yaml(config_file)  # parse_configuration(config_file)
    print('Initializing dataset...')
    train_dataset = create_dataset(configuration['train_dataset_params'])
    train_dataset_size = len(train_dataset)
    print('The number of training samples = {0}'.format(train_dataset_size))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform model training.')
    parser.add_argument('-configfile', default='config.yaml', help='path to the configfile')
    print(os.getcwd())
    args = parser.parse_args()

    main(args.configfile)
