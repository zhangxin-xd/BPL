from models.stagewise_resnet import *

MODEL_MAP_21 = {
    'sres50': create_SResNet50_21,
    'sres101': create_SResNet101_21,

}
MODEL_MAP_45 = {
    'sres50': create_SResNet50,
    'sres101': create_SResNet101,

}


DATASET_TO_MODEL_MAP = {
    'nwpu-45': MODEL_MAP_45,
    'ucml-21': MODEL_MAP_21,
}


#   return the model creation function
def get_model_fn(dataset_name, model_name):
    # print(DATASET_TO_MODEL_MAP[dataset_name.replace('_blank', '_standard')].keys())
    return DATASET_TO_MODEL_MAP[dataset_name][model_name]

def get_dataset_name_by_model_name(model_name):
    for dataset_name, model_map in DATASET_TO_MODEL_MAP.items():
        if model_name in model_map:
            return dataset_name
    return None