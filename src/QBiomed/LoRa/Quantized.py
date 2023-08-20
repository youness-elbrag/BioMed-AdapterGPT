import loratorch as LoraT 
import torch.nn as nn 
import numpy as np 

def make_lora_layer(layer, lora_r):
    new_layer = LoraT.Linear(
        in_features=layer.in_features,
        out_features=layer.out_features,
        bias=layer.bias is not None,  # Fixing the bias check
        r=lora_r
    )
    
    # Cloning the tensor
    new_layer.weight = nn.Parameter(layer.weight.detach().clone())  
    
    if layer.bias is not None:
        new_layer.bias = nn.Parameter(layer.bias.detach().clone())  
    
    return new_layer

def make_lora_replace(model, depth=1, path="", verbose=True):
    if depth > 10:
        return model
    
    if isinstance(model, nn.Linear) and ("self_attn" in path or "cross_attn" in path):
        if verbose:
            print(f"Find linear {path}:", type(model))
        return make_lora_layer(model)
    
    for key, module in model.named_children():  # Using named_children() for cleaner iteration
        if isinstance(module, nn.Linear) and ("self_attn" in path or "cross_attn" in path):
            layer = make_lora_layer(module)
            setattr(model, key, layer)
            if verbose:
                print(f"Find linear {path}:{key} :", type(module))
                
        elif isinstance(module, nn.ModuleList):
            for i, elem in enumerate(module):
                layer = make_lora_replace(elem, depth+1, f"{path}:{key}[{i}]", verbose=verbose)
                if layer is not None:
                    module[i] = layer
                
        elif isinstance(module, nn.ModuleDict):
            for module_key, item in module.items():
                layer = make_lora_replace(item, depth+1, f"{path}:{key}:{module_key}", verbose=verbose)
                if layer is not None:
                    module[module_key] = layer
                
        else:
            layer = make_lora_replace(module, depth+1, f"{path}:{key}", verbose=verbose)
            if layer is not None:
                setattr(model, key, layer)
    
    return model