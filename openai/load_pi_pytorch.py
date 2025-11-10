import torch
import safetensors
import jax

import openpi
from openpi.models_pytorch import pi0_pytorch
from openpi.training import config as _config
from openpi.training.config import LeRobotLiberoDataConfig, DataConfig

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f'using device {device}')        

def load_pytorch(model_config, weight_path):
    model = pi0_pytorch.PI0Pytorch(config=model_config).to(device)
    safetensors.torch.load_model(model, weight_path)
    return model


def traverse_torch_module(model):
    for name, child in model.named_children():
        print(f'name:{name}, child:{type(child)}')

        traverse_torch_module(child)


def get_data_loader(config_name):
    import dataclasses
    from openpi.training import data_loader as _data_loader
#    config = _config.get_config("pi0_aloha_sim")
    config = _config.get_config(config_name)
    config = dataclasses.replace(config, batch_size=1)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=1,
        shuffle=True,
        framework='pytorch'  # must specify.
    )

    return loader


if __name__ == "__main__":
    # 典型pytorch safetensors加载
    # from safetensors.torch import load_model, save_model

    # save the state dict
    # save_model(model, "resnet18.safetensors")

    # load the model without weights
    # model_st = resnet18(pretrained=False)
    # load_model(model_st, "resnet18.safetensors")
    import sys
    model_name = sys.argv[1]
    weight_tensor = sys.argv[2]
    config = _config.get_config(model_name)
    pi0_model = load_pytorch(config.model, weight_tensor)
    pi0_model.eval()

    loader = get_data_loader(model_name)
    count, step = 0, 0
    num_train_steps = 10
    with torch.no_grad():
        for observation, actions in loader:
            # Check if we've reached the target number of steps
            if step >= num_train_steps:
                break

            # The unified data loader returns (observation, actions) tuple
            observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901
            actions = actions.to(torch.float32)  # noqa: PLW2901
            actions = actions.to(device)  # noqa: PLW2901

            losses = pi0_model(observation, actions)
            print(f'result: {type(losses)}')
#    traverse_torch_module(pi0_model)
    # Combine all parameters (no prefix needed for our model structure)
    # all_params = {**paligemma_params, **gemma_params, **projection_params}
    # Load state dict
    # pi0_model.load_state_dict(all_params, strict=False)
