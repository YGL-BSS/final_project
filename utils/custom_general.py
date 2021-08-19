import time
from utils.torch_utils import time_sync


class TimeCheck():
    def __init__(self, out=True):
        self.start = '0'
        self.end = '0'
        t = time_sync()
        self.t_start = t
        self.t_end = t
        self.out = out
        if out:
            print()

    def initial(self, tag='start'):
        self.start = tag
        self.t_start = time_sync()
    
    def check(self, tag='end', ret=False):
        self.end = tag
        self.t_end = time_sync()
        time_interval = self.t_end - self.t_start
        if self.out: 
            print(f'[{self.start} ~ {self.end}] {time_interval:5.2f} sec', end='\t')
        if ret:
            return time_interval