from os import stat
import torch
import torch.nn as nn

def load_model(model,logger,activate=False,stat_path=""):
    logger.info("Loading model")
    dic=torch.load(stat_path)
    model.load_state_dict(dic['state_dict'])
    logger.info("Loading Finish")
    if activate==True:
        model.mode=1
        logger.info("Model's activation is prepared")
    return model