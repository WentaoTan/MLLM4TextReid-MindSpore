import torch
from mindspore import Tensor, save_checkpoint

def pytorch2mindspore(default_file = 'best0.pth'):
    # read pth file
    # print(torch.load(default_file)['model'].keys())
    par_dict = torch.load(default_file)['model']
    params_list = []
    for name in par_dict:
        param_dict = {}
        parameter = par_dict[name]
        if 'ln' in name:
            if 'weight' in name:
                name = name.replace('weight','gamma')
            if 'bias' in name:
                name = name.replace('bias','beta')
        if 'embedding' in name:
            name = name.replace('weight', 'embedding_table')
        param_dict['name'] = name
        param_dict['data'] = Tensor(parameter.cpu().float().numpy())
        params_list.append(param_dict)
    save_checkpoint(params_list,  'ms_clip.ckpt')

pytorch2mindspore('best0.pth')