from torch.optim.lr_scheduler import LambdaLR


class LinearScheduler(LambdaLR):
    def __init__(self, optimizer, total_epoch, warm_up, last_epoch=-1):
        def rate(epoch):
            x = epoch/total_epoch
            s = (x <= warm_up)
            return (s*(x/warm_up)+(1-s))*(1-x)
        super().__init__(optimizer, rate, last_epoch)


