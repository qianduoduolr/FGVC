from mmcv import runner
from mmcv.parallel.utils import is_module_wrapper
from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class EMAHook_MoCo(Hook):
    def __init__(self, source_name,
                       target_name,
                       momentum,
                       interval=1
                       ):
        """
        """
        assert isinstance(interval, int) and interval > 0
        self.momentum = momentum
        self.source_name = source_name
        self.target_name = target_name
        self.interval = interval
    
    def before_run(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            self.model = model.module
        self.source_module = self.get_module(self.source_name)
        self.target_module = self.get_module(self.target_name)


    def after_train_iter(self, runner):
        
        curr_step = runner.iter
        if curr_step % self.interval != 0:
            return
        else:
            self.momentum_update()
        

    def after_train_epoch(self, runner):
        pass

    def after_train_epoch(self, runner):
        pass

    def get_module(self, name):

        for key, module in self.model.named_children():
            if name == key:
                return module
        
        raise RuntimeError
    
    def momentum_update(self):
        """ model_ema = m * model_ema + (1 - m) model """
        for p1, p2 in zip(self.source_module.parameters(), self.target_module.parameters()):
            p2.data.mul_(self.momentum).add_(p1.detach().data, alpha=1 - self.momentum)

        
