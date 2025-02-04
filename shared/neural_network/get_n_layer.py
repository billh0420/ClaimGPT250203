# get_n_layer.py

from shared.neural_network.MathGPTLanguageModel import GPTLanguageModel

def get_n_layer(model: GPTLanguageModel) -> int:
    layer_count = 0
    for module_name, module in model.named_modules():
        if module_name.startswith('blocks'):
            split_module_name = module_name.split('.')
            if len(split_module_name) == 2:
                layer_count += 1
    return layer_count
