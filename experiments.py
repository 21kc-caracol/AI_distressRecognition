from tqdm import tqdm

import evaluate_screamClf

def different_data_size():
    screamGlobals= evaluate_screamClf.global_For_Clf()

    for size in tqdm([150,200,250,300,350,400]):
        screamGlobals.try_lower_amount = size
        evaluate_screamClf.experiment_data_size(screamGlobals)

if __name__ == '__main__':
    different_data_size()
