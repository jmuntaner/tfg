import torch


tensor_dict = torch.load('/home/usuaris/veu/joan.muntaner/experiments/simplenmt/simple-nmt/model_deen/model.deen.15.1.30-3.65.1.48-4.40.pth', map_location='cpu') # OrderedDict
tensor_list = list(tensor_dict.items())
for layer_tensor_name, tensor in tensor_list:
    print('Layer {}: {} elements'.format(layer_tensor_name, torch.numel(tensor)))