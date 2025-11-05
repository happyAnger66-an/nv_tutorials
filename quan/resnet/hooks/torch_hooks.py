import torch

def hook_module_outputs(model, hook_outputs):
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.ReLU)):
            print(f'do register forward hook. name:{name} {type(module)}')
            hook = module.register_forward_hook(
                lambda m, i, o, name=name: hook_outputs.update({name: o.detach()})
            )
            hooks.append(hook)
    return hooks