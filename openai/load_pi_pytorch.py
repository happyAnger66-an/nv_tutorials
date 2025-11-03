import safetensors

import openpi
from openpi.models_pytorch import pi0_pytorch
from openpi.training import config as _config

def load_pytorch(model_config, weight_path):
    model = pi0_pytorch.PI0Pytorch(config=model_config)
    safetensors.torch.load_model(model, weight_path)
    return model

def traverse_torch_module(model):
    for name, child in model.named_children():
        print(f'name:{name}, child:{type(child)}')

        traverse_torch_module(child)

if __name__ == "__main__":
    ## 典型pytorch safetensors加载
        #from safetensors.torch import load_model, save_model

        # save the state dict
        #save_model(model, "resnet18.safetensors")

        # load the model without weights
        #model_st = resnet18(pretrained=False) 
        #load_model(model_st, "resnet18.safetensors")
    import sys
    config = _config.get_config(sys.argv[1])    
    pi0_model = load_pytorch(config.model, sys.argv[2])
    traverse_torch_module(pi0_model)
    # Combine all parameters (no prefix needed for our model structure)
    # all_params = {**paligemma_params, **gemma_params, **projection_params}
    # Load state dict
    # pi0_model.load_state_dict(all_params, strict=False)
    