import time


class TimeCheck():
    def __init__(self):
        self.start = '0'
        self.end = '0'
        self.t_start = time.time()
        self.t_end = time.time()
        print()

    def initial(self, tag='start'):
        self.start = tag
        self.t_start = time.time()
    
    def check(self, tag='end', out=True):
        self.end = tag
        self.t_end = time.time()
        if out:
            time_interval = self.t_end - self.t_start
            print(f'[{self.start} ~ {self.end}] {time_interval:5.2f} sec', end='\t')
            return time_interval