from mmcv.runner import load_state_dict

def copy_params(model, model_test):
    origin_params = {}
    
    for name, param in model.state_dict().items():
        if name in model_test.state_dict().keys():
            origin_params[name.replace('module.','')] = param.data.detach().cpu()
    load_state_dict(model_test,origin_params, strict=False)