import argparse
from run import run_evaluation


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-s", "--seed", type=int)
    args = parser.parse_args()
    dataset, seed = "electricity", 40
    if args.dataset is not None:
        dataset = args.dataset
    if args.seed is not None:
        seed = args.seed

    run_evaluation(data_name=dataset, seed=seed)


if __name__ == '__main__':
    main()
