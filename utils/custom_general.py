import time
import numpy as np

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


class GestureBuffer():

    def __init__(self, names):
        self._names = names
        self.log_detect = np.empty((0, len(names)), int)
        self.log_time = np.empty((0, 1), float)
        self._now = time.time()
        self._buf_delay = 1  # 1 sec 동안의 데이터만 저장

    def update_buf(self, t_send, detect_num=None):
        '''
        새로운 detect가 발생할때마다 버퍼를 업데이트하는 함수
        detect 없이 갱신도 가능함
        '''
        log_detect = self.log_detect
        log_time = self.log_time
        
        if type(detect_num) != type(None):
            log_detect = np.append(log_detect, np.array([detect_num]), axis=0)
            log_time = np.append(log_time, np.array([[t_send]]), axis=0)

        # 현재 시간과 비교해서 날릴 내용은 날리기
        now = time.time()
        mask = np.where(now - log_time < self._buf_delay)[0]
        log_detect = log_detect[mask]
        log_time = log_time[mask]

        # 저장
        self.log_detect = log_detect
        self.log_time = log_time
        # self._now = now

    def get_action(self):
        '''
        인식된 action list 반환하기
        '''
        buf = self.log_detect
        detected_action = np.array([0] * len(self._names), dtype=np.int32)
        now = time.time()

        if now - self._now >= self._buf_delay:
            self._now = now

            if len(buf) >= 1:
                most_cnt = 0
                for action in np.unique(buf, axis=0):
                    cnt = self._find_same_row(buf, action)
                    if cnt > most_cnt:
                        detected_action = action
        return detected_action

    def _find_same_row(self, arr_2d, target_row):
        '''
        arr_2d의 row 중에서 target_row와 같은 row의 개수를 반환
        '''
        # MSE에서 오차 계산을 squared error로 하는 것을 응용
        return (((arr_2d - target_row) ** 2).sum(axis=1) == 0).sum()

