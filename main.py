import load_data
from dataset import *
from meta import *
from model import *
import json
import torch
import random
import numpy as np

if __name__ == "__main__":

    with open('config.json') as config_file:
        cfg = json.load(config_file)

    SEED = cfg["seed"]
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    TOP_WORDS = cfg["top_words"]
    N_WAY = cfg["n_way"]
    K_SHOT = cfg["k_shot"]
    QUERY_SIZE = cfg["query_size"]
    N_TASK = cfg["n_task"]
    VALIDATION = cfg["use_validation"]

    data = load_data._load_json("data/" + cfg["dataset"] + ".json")
    train, val, test = load_data.dataset_loader(cfg["dataset"])
    n_class = len(train+val+test)
    data = load_data.group_class(data, n_class)

    embedder = Embedder()
    sampler = Dataset(data, train, val, test, embedder=embedder,
                      use_validation=VALIDATION, top_words=TOP_WORDS)
    sampler.embed_counter()

    trainer = Trainer(cfg, sampler)
    trainer.train(N_WAY, K_SHOT, QUERY_SIZE, N_TASK)
