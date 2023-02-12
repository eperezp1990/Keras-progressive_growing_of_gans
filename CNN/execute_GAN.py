import sys
import json
import os
import time
from multiprocessing import Pool

from utils import load_data
from execute_fold import evaluate_fold

#models
from models.MobileNet_model import MobileNet_imagenet, MobileNet_sinpesos
from models.DenseNet201_model import DenseNet201_imagenet, DenseNet201_sinpesos
from models.InceptionV3_model import InceptionV3_imagenet, InceptionV3_sinpesos
from models.Xception_model import Xception_imagenet, Xception_sinpesos
from models.NASNetMobile_model import NASNetMobile_imagenet, NASNetMobile_sinpesos

models = [
    MobileNet_imagenet(),

    DenseNet201_imagenet(),

    InceptionV3_imagenet(),

    Xception_imagenet(),

    NASNetMobile_imagenet(),
]
#end models

###################### start evaluation ######################

# config_file = str(sys.argv[1])
config_file = './config.json'

with open(config_file) as json_data:
    configuration = json.load(json_data)

reports_dir = configuration['reportsDir']
gpu = configuration['gpu']

for dataset in configuration['datasets']:
    data_folds = load_data(dataset, root='/home/eperezp1990/images/224-GAN-training-ready')
    for model in models:
        for optimizer in configuration['optimizers']:
            curr_report_dir = os.path.join(reports_dir, dataset['name'], model['name'], optimizer)
            
            os.makedirs(curr_report_dir, exist_ok=True)

            execute_configs = []
            for fold, train_x_f, train_y_f, test_x_f, test_y_f, train_extra_x_f in data_folds:
                curr_report = os.path.join(curr_report_dir, str(fold) + '_fold.csv')
                if os.path.exists(curr_report):
                    print('>> done', curr_report)
                    continue
                execute_configs.append((
                    curr_report,
                    model, 
                    optimizer, 
                    train_x_f, 
                    train_y_f, 
                    test_x_f, 
                    test_y_f, 
                    train_extra_x_f,
                    # pre-locate each gpu 
                    gpu[len(execute_configs)%len(gpu)]
                ))
            # execute in each gpu the configurations
            for i in range(0, len(execute_configs), len(gpu)):
                with Pool(processes=len(gpu)) as pool:
                    pool.starmap(evaluate_fold, execute_configs[i:i+len(gpu)])
                    time.sleep(20)

