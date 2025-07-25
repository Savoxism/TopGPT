import torch

class GPTConfig:
    # Model archtecture
    vocab_size = None
    block_size = 256
    n_embd = 384
    n_head = 6
    n_layer = 6
    dropout = 0.2

    # Training hyperparameters
    batch_size = 64
    max_iters = 5000
    eval_interval = 500
    learning_rate = 3e-4
    eval_iters = 200

    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # data
    data_path = "data/shakespeare/shakespeare.txt"
    model_save_path = "checkpoints/model.pt"
    result_file = "result.txt"
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)