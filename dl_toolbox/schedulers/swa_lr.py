import math

class SwaLR:
    def __init__(
        self, 
        cycle_len,
        cycle_min,
        cycle_max,
        mode
    ):
        self.cycle_len = cycle_len
        self.cycle_min = cycle_min
        self.cycle_max = cycle_max
        self.mode = mode
        
    def t(self, p):
        if self.mode=='lin':
            return 1 - p
        elif self.mode=='cos':
            return (1+math.cos(math.pi*p))/2
        elif self.mode=='cos2':
            return (1+math.cos(math.pi*p*p))/2
        
    def __call__(self, i):
        if i<0:
            return self.cycle_max
            #p = i/self.start_swa
            #return self.end_lr + self.t(p) * (self.start_lr - self.end_lr)
        else:
            p = (i%self.cycle_len)/(self.cycle_len-1)
            return self.cycle_min + self.t(p) * (self.cycle_max - self.cycle_min)