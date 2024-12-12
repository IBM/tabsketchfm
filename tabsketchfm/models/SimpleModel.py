import torch

class SimpleModel(torch.nn.Module):

    def __init__(self, config):
        super(SimpleModel, self).__init__(config)
        self.linear1 = torch.nn.Linear(config.task_specific_params['hash_input_size'], config.hidden_size)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

