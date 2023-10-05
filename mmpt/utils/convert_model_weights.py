# import torch

# models_keys = dict(
#     resnet50= ['conv1.conv.weight', 'conv1.bn.weight', 'conv1.bn.bias', 
#     'conv1.bn.running_mean', 'conv1.bn.running_var', 'conv1.bn.num_batches_tracked', 
#     'layer1.0.conv1.conv.weight', 'layer1.0.conv1.bn.weight', 'layer1.0.conv1.bn.bias', 
#     'layer1.0.conv1.bn.running_mean', 'layer1.0.conv1.bn.running_var', 'layer1.0.conv1.bn.num_batches_tracked',
#     'layer1.0.conv2.conv.weight', 'layer1.0.conv2.bn.weight', 'layer1.0.conv2.bn.bias', 
#     'layer1.0.conv2.bn.running_mean', 'layer1.0.conv2.bn.running_var', 'layer1.0.conv2.bn.num_batches_tracked', 
#     'layer1.0.conv3.conv.weight', 'layer1.0.conv3.bn.weight', 'layer1.0.conv3.bn.bias', 
#     'layer1.0.conv3.bn.running_mean', 'layer1.0.conv3.bn.running_var', 
#     'layer1.0.conv3.bn.num_batches_tracked', 'layer1.0.downsample.conv.weight', 
#     'layer1.0.downsample.bn.weight', 'layer1.0.downsample.bn.bias', 
#     'layer1.0.downsample.bn.running_mean', 'layer1.0.downsample.bn.running_var',
#     'layer1.0.downsample.bn.num_batches_tracked', 'layer1.1.conv1.conv.weight', 
#     'layer1.1.conv1.bn.weight', 'layer1.1.conv1.bn.bias', 'layer1.1.conv1.bn.running_mean', 
#     'layer1.1.conv1.bn.running_var', 'layer1.1.conv1.bn.num_batches_tracked', 'layer1.1.conv2.conv.weight', 
#     'layer1.1.conv2.bn.weight', 'layer1.1.conv2.bn.bias', 'layer1.1.conv2.bn.running_mean', 
#     'layer1.1.conv2.bn.running_var', 'layer1.1.conv2.bn.num_batches_tracked', 'layer1.1.conv3.conv.weight',
#     'layer1.1.conv3.bn.weight', 'layer1.1.conv3.bn.bias', 'layer1.1.conv3.bn.running_mean', 
#     'layer1.1.conv3.bn.running_var', 'layer1.1.conv3.bn.num_batches_tracked', 'layer1.2.conv1.conv.weight',
#     'layer1.2.conv1.bn.weight', 'layer1.2.conv1.bn.bias', 'layer1.2.conv1.bn.running_mean', 
#     'layer1.2.conv1.bn.running_var', 'layer1.2.conv1.bn.num_batches_tracked', 'layer1.2.conv2.conv.weight', 
#     'layer1.2.conv2.bn.weight', 'layer1.2.conv2.bn.bias', 'layer1.2.conv2.bn.running_mean', 
#     'layer1.2.conv2.bn.running_var', 'layer1.2.conv2.bn.num_batches_tracked', 'layer1.2.conv3.conv.weight',
#     'layer1.2.conv3.bn.weight', 'layer1.2.conv3.bn.bias', 'layer1.2.conv3.bn.running_mean', 
#     'layer1.2.conv3.bn.running_var', 'layer1.2.conv3.bn.num_batches_tracked', 'layer2.0.conv1.conv.weight',
#     'layer2.0.conv1.bn.weight', 'layer2.0.conv1.bn.bias', 'layer2.0.conv1.bn.running_mean', 
#     'layer2.0.conv1.bn.running_var', 'layer2.0.conv1.bn.num_batches_tracked', 'layer2.0.conv2.conv.weight',
#     'layer2.0.conv2.bn.weight', 'layer2.0.conv2.bn.bias', 'layer2.0.conv2.bn.running_mean', 
#     'layer2.0.conv2.bn.running_var', 'layer2.0.conv2.bn.num_batches_tracked', 'layer2.0.conv3.conv.weight',
#     'layer2.0.conv3.bn.weight', 'layer2.0.conv3.bn.bias', 'layer2.0.conv3.bn.running_mean', 
#     'layer2.0.conv3.bn.running_var', 'layer2.0.conv3.bn.num_batches_tracked', 'layer2.0.downsample.conv.weight',
#     'layer2.0.downsample.bn.weight', 'layer2.0.downsample.bn.bias', 'layer2.0.downsample.bn.running_mean',
#     'layer2.0.downsample.bn.running_var', 'layer2.0.downsample.bn.num_batches_tracked', 
#     'layer2.1.conv1.conv.weight', 'layer2.1.conv1.bn.weight', 'layer2.1.conv1.bn.bias',
#     'layer2.1.conv1.bn.running_mean', 'layer2.1.conv1.bn.running_var', 'layer2.1.conv1.bn.num_batches_tracked',
#     'layer2.1.conv2.conv.weight', 'layer2.1.conv2.bn.weight', 'layer2.1.conv2.bn.bias', 'layer2.1.conv2.bn.running_mean', 'layer2.1.conv2.bn.running_var', 'layer2.1.conv2.bn.num_batches_tracked', 'layer2.1.conv3.conv.weight', 'layer2.1.conv3.bn.weight', 'layer2.1.conv3.bn.bias', 'layer2.1.conv3.bn.running_mean', 'layer2.1.conv3.bn.running_var', 'layer2.1.conv3.bn.num_batches_tracked', 'layer2.2.conv1.conv.weight', 'layer2.2.conv1.bn.weight', 'layer2.2.conv1.bn.bias', 'layer2.2.conv1.bn.running_mean', 'layer2.2.conv1.bn.running_var', 'layer2.2.conv1.bn.num_batches_tracked', 'layer2.2.conv2.conv.weight', 'layer2.2.conv2.bn.weight', 'layer2.2.conv2.bn.bias', 'layer2.2.conv2.bn.running_mean', 'layer2.2.conv2.bn.running_var', 'layer2.2.conv2.bn.num_batches_tracked', 'layer2.2.conv3.conv.weight', 'layer2.2.conv3.bn.weight', 'layer2.2.conv3.bn.bias', 'layer2.2.conv3.bn.running_mean', 'layer2.2.conv3.bn.running_var', 'layer2.2.conv3.bn.num_batches_tracked', 'layer2.3.conv1.conv.weight', 'layer2.3.conv1.bn.weight', 'layer2.3.conv1.bn.bias', 'layer2.3.conv1.bn.running_mean', 'layer2.3.conv1.bn.running_var', 'layer2.3.conv1.bn.num_batches_tracked', 'layer2.3.conv2.conv.weight', 'layer2.3.conv2.bn.weight', 'layer2.3.conv2.bn.bias', 'layer2.3.conv2.bn.running_mean', 'layer2.3.conv2.bn.running_var', 'layer2.3.conv2.bn.num_batches_tracked', 'layer2.3.conv3.conv.weight', 'layer2.3.conv3.bn.weight', 'layer2.3.conv3.bn.bias', 'layer2.3.conv3.bn.running_mean', 'layer2.3.conv3.bn.running_var', 'layer2.3.conv3.bn.num_batches_tracked', 'layer3.0.conv1.conv.weight', 'layer3.0.conv1.bn.weight', 'layer3.0.conv1.bn.bias', 'layer3.0.conv1.bn.running_mean', 'layer3.0.conv1.bn.running_var', 'layer3.0.conv1.bn.num_batches_tracked', 'layer3.0.conv2.conv.weight', 'layer3.0.conv2.bn.weight', 'layer3.0.conv2.bn.bias', 'layer3.0.conv2.bn.running_mean', 'layer3.0.conv2.bn.running_var', 'layer3.0.conv2.bn.num_batches_tracked', 'layer3.0.conv3.conv.weight', 'layer3.0.conv3.bn.weight', 'layer3.0.conv3.bn.bias', 'layer3.0.conv3.bn.running_mean', 'layer3.0.conv3.bn.running_var', 'layer3.0.conv3.bn.num_batches_tracked', 'layer3.0.downsample.conv.weight', 'layer3.0.downsample.bn.weight', 'layer3.0.downsample.bn.bias', 'layer3.0.downsample.bn.running_mean', 'layer3.0.downsample.bn.running_var', 'layer3.0.downsample.bn.num_batches_tracked', 'layer3.1.conv1.conv.weight', 'layer3.1.conv1.bn.weight', 'layer3.1.conv1.bn.bias', 'layer3.1.conv1.bn.running_mean', 'layer3.1.conv1.bn.running_var', 'layer3.1.conv1.bn.num_batches_tracked', 'layer3.1.conv2.conv.weight', 'layer3.1.conv2.bn.weight', 'layer3.1.conv2.bn.bias', 'layer3.1.conv2.bn.running_mean', 'layer3.1.conv2.bn.running_var', 'layer3.1.conv2.bn.num_batches_tracked', 'layer3.1.conv3.conv.weight', 'layer3.1.conv3.bn.weight', 'layer3.1.conv3.bn.bias', 'layer3.1.conv3.bn.running_mean', 'layer3.1.conv3.bn.running_var', 'layer3.1.conv3.bn.num_batches_tracked', 'layer3.2.conv1.conv.weight', 'layer3.2.conv1.bn.weight', 'layer3.2.conv1.bn.bias', 'layer3.2.conv1.bn.running_mean', 'layer3.2.conv1.bn.running_var', 'layer3.2.conv1.bn.num_batches_tracked', 'layer3.2.conv2.conv.weight', 'layer3.2.conv2.bn.weight', 'layer3.2.conv2.bn.bias', 'layer3.2.conv2.bn.running_mean', 'layer3.2.conv2.bn.running_var', 'layer3.2.conv2.bn.num_batches_tracked', 'layer3.2.conv3.conv.weight', 'layer3.2.conv3.bn.weight', 'layer3.2.conv3.bn.bias', 'layer3.2.conv3.bn.running_mean', 'layer3.2.conv3.bn.running_var', 'layer3.2.conv3.bn.num_batches_tracked', 'layer3.3.conv1.conv.weight', 'layer3.3.conv1.bn.weight', 'layer3.3.conv1.bn.bias', 'layer3.3.conv1.bn.running_mean', 'layer3.3.conv1.bn.running_var', 'layer3.3.conv1.bn.num_batches_tracked', 'layer3.3.conv2.conv.weight', 'layer3.3.conv2.bn.weight', 'layer3.3.conv2.bn.bias', 'layer3.3.conv2.bn.running_mean', 'layer3.3.conv2.bn.running_var', 'layer3.3.conv2.bn.num_batches_tracked', 'layer3.3.conv3.conv.weight', 'layer3.3.conv3.bn.weight', 'layer3.3.conv3.bn.bias', 'layer3.3.conv3.bn.running_mean', 'layer3.3.conv3.bn.running_var', 'layer3.3.conv3.bn.num_batches_tracked', 'layer3.4.conv1.conv.weight', 'layer3.4.conv1.bn.weight', 'layer3.4.conv1.bn.bias', 'layer3.4.conv1.bn.running_mean', 'layer3.4.conv1.bn.running_var', 'layer3.4.conv1.bn.num_batches_tracked', 'layer3.4.conv2.conv.weight', 'layer3.4.conv2.bn.weight', 'layer3.4.conv2.bn.bias', 'layer3.4.conv2.bn.running_mean', 'layer3.4.conv2.bn.running_var', 'layer3.4.conv2.bn.num_batches_tracked', 'layer3.4.conv3.conv.weight', 'layer3.4.conv3.bn.weight', 'layer3.4.conv3.bn.bias', 'layer3.4.conv3.bn.running_mean', 'layer3.4.conv3.bn.running_var', 'layer3.4.conv3.bn.num_batches_tracked', 'layer3.5.conv1.conv.weight', 'layer3.5.conv1.bn.weight', 'layer3.5.conv1.bn.bias', 'layer3.5.conv1.bn.running_mean', 'layer3.5.conv1.bn.running_var', 'layer3.5.conv1.bn.num_batches_tracked', 'layer3.5.conv2.conv.weight', 'layer3.5.conv2.bn.weight', 'layer3.5.conv2.bn.bias', 'layer3.5.conv2.bn.running_mean', 'layer3.5.conv2.bn.running_var', 'layer3.5.conv2.bn.num_batches_tracked', 'layer3.5.conv3.conv.weight', 'layer3.5.conv3.bn.weight', 'layer3.5.conv3.bn.bias', 'layer3.5.conv3.bn.running_mean', 'layer3.5.conv3.bn.running_var', 'layer3.5.conv3.bn.num_batches_tracked', 'layer4.0.conv1.conv.weight', 'layer4.0.conv1.bn.weight', 'layer4.0.conv1.bn.bias', 'layer4.0.conv1.bn.running_mean', 'layer4.0.conv1.bn.running_var', 'layer4.0.conv1.bn.num_batches_tracked', 'layer4.0.conv2.conv.weight', 'layer4.0.conv2.bn.weight', 'layer4.0.conv2.bn.bias', 'layer4.0.conv2.bn.running_mean', 'layer4.0.conv2.bn.running_var', 'layer4.0.conv2.bn.num_batches_tracked', 'layer4.0.conv3.conv.weight', 'layer4.0.conv3.bn.weight', 'layer4.0.conv3.bn.bias', 'layer4.0.conv3.bn.running_mean', 'layer4.0.conv3.bn.running_var', 'layer4.0.conv3.bn.num_batches_tracked', 'layer4.0.downsample.conv.weight', 'layer4.0.downsample.bn.weight', 'layer4.0.downsample.bn.bias', 'layer4.0.downsample.bn.running_mean', 'layer4.0.downsample.bn.running_var', 'layer4.0.downsample.bn.num_batches_tracked', 'layer4.1.conv1.conv.weight', 'layer4.1.conv1.bn.weight', 'layer4.1.conv1.bn.bias', 'layer4.1.conv1.bn.running_mean', 'layer4.1.conv1.bn.running_var', 'layer4.1.conv1.bn.num_batches_tracked', 'layer4.1.conv2.conv.weight', 'layer4.1.conv2.bn.weight', 'layer4.1.conv2.bn.bias', 'layer4.1.conv2.bn.running_mean', 'layer4.1.conv2.bn.running_var', 'layer4.1.conv2.bn.num_batches_tracked', 'layer4.1.conv3.conv.weight', 'layer4.1.conv3.bn.weight', 'layer4.1.conv3.bn.bias', 'layer4.1.conv3.bn.running_mean', 'layer4.1.conv3.bn.running_var', 'layer4.1.conv3.bn.num_batches_tracked', 'layer4.2.conv1.conv.weight', 'layer4.2.conv1.bn.weight', 'layer4.2.conv1.bn.bias', 'layer4.2.conv1.bn.running_mean', 'layer4.2.conv1.bn.running_var', 'layer4.2.conv1.bn.num_batches_tracked', 'layer4.2.conv2.conv.weight', 'layer4.2.conv2.bn.weight', 'layer4.2.conv2.bn.bias', 'layer4.2.conv2.bn.running_mean', 'layer4.2.conv2.bn.running_var', 'layer4.2.conv2.bn.num_batches_tracked', 'layer4.2.conv3.conv.weight', 'layer4.2.conv3.bn.weight', 'layer4.2.conv3.bn.bias', 'layer4.2.conv3.bn.running_mean', 'layer4.2.conv3.bn.running_var', 'layer4.2.conv3.bn.num_batches_tracked']
# )


# def convert_to_mmcv(model_keys_list, file, dst_file, key_name=None):
#     ckpt = torch.load(file)
#     if 'state_dict' in ckpt.keys():
#         state_dict = ckpt['state_dict']
#     else:
#         state_dict = ckpt
    
#     if key_name is not None:
#         state_dict = {k:v for k,v in state_dict.items() if k.find(key_name) != -1}

#     revised_state_dict = {}

#     for idx, (k,v) in enumerate(state_dict.items()):
#         revised_state_dict[model_keys_list[idx]] = v

#     torch.save(revised_state_dict, dst_file)


# convert_to_mmcv(models_keys['resnet50'], file='/home/lr/models/vos/stcn.pth', dst_file='/home/lr/models/vos/stcn_revised_keys.pth', key_name='key_encoder.')

    
    
import torch
import torchvision
import torchvision.models.resnet as resnet
import torch.nn as nn

model_path = '/home/lr/models/mmpt/timecycle.pth.tar'
model_state = torch.load(model_path, map_location='cpu')['state_dict']

net = resnet.resnet50()
net_state = net.state_dict()


for k in [k for k in model_state.keys() if 'encoderVideo' in k]:
    kk = k.replace('module.encoderVideo.', '')
    tmp = model_state[k]
    if net_state[kk].shape != model_state[k].shape and net_state[kk].dim() == 4 and model_state[k].dim() == 5:
        tmp = model_state[k].squeeze(2)
    net_state[kk][:] = tmp[:]
    
net.load_state_dict(net_state)

afterconv1 = nn.Conv2d(1024, 512, kernel_size=1, bias=False)
relu_layer = nn.ReLU(inplace=True)

afterconv1_weight = model_state['module.afterconv1.weight']
afterconv1_weight = afterconv1_weight.squeeze(2)

x = afterconv1.state_dict()
x['weight'][:] = afterconv1_weight[:]

layers = list(net.children())[:-3]
layers.append(afterconv1)
layers.append(relu_layer)

net = nn.Sequential(*layers)

inp = torch.randn(1, 3, 240, 240)
out = net(inp)
print(out.size())


torch.save(net_state, '/home/lr/models/mmpt/timecycle_revised_keys.pth')