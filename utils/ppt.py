'''
gesture 감지 -> 로직 -> 조작
'''
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import os

def output_to_detect(s):
    '''
    감지된 gesture를 리스트로 반환한다.
    감지되지 않으면 빈 리스트를 반환한다.
    '''
    temp = s.split()
    label_dict = ['K', 'L', 'paper', 'rock', 'scissor', 'W']
    send_result = []
    
    if len(temp) > 2:
        label_list = temp[3::2]
        label_num = temp[2::2]
        # print(label_list, label_num)
        for n, label in zip(label_num, label_list):
            send_result += [label_dict[int(label[0])]] * int(float(n))
    # print(send_result)
    return send_result


class EncodeInput:
    def __init__(self,alpha):
        self.alpha = alpha
                
        self.hand_count = {
            'K': 0,
            'L': 0,
            'paper': 0,
            'rock': 0,
            'scissor': 0,
            'W': 0
        }
        self.none_count = 0

    def reset_count(self):
        self.hand_count = {
            'K': 0,
            'L': 0,
            'paper': 0,
            'rock': 0,
            'scissor': 0,
            'W': 0
        }
        self.none_count = 0

    def accumulate(self,hands):
        if len(hands) == 1:
            hand = hands[0]
            self.hand_count[hand] += 1
        
        else: 
            self.none_count += 1
        

    def verify(self):
        output = None
        if self.none_count == self.alpha:
            self.reset_count()

        elif max(self.hand_count.values()) == self.alpha:
            output = max(self.hand_count, key=self.hand_count.get)
            self.reset_count()
        
        return output

    def encode(self,hands):
        self.accumulate(hands)
        output = self.verify()
        if output == None: pass
        return output

# if __name__ == '__main__':
#     EI = EncodeInput(10)
#     while True:
#         hands = list(input().split())
#         command = EI.encode(hands)
#         if command: print(f'==={command}===')


class Gesture2Command:
    gestures = ['K', 'L', 'paper', 'rock', 'scissor', 'W']
    
    def __init__(self, gesture = None):
        self.gesture = gesture
        self.current_page = -1
        self.ppt_path = os.path.join(os.getcwd(), 'TestPPT') 
        self.total_page = self.choosePPT()
        self.all_ppt = [os.path.join(self.ppt_path, i) for i in self.ppt_list]

    def choosePPT(self):
        self.ppt_list = os.listdir(self.ppt_path)
        return len(self.ppt_list)

    def openFirstSlide(self):
        options = Options()
        options.add_argument('--start-maximized')
        self.main_driver = webdriver.Chrome(executable_path="./chromedriver", options = options)
        self.current_page = 0
        self.main_driver.get(self.all_ppt[self.current_page])

    def nextSlide(self):
        if self.current_page < self.total_page - 1:
            self.current_page += 1
            self.main_driver.get(self.all_ppt[self.current_page])

    def previousSlide(self):
        if self.current_page > 1:
            self.current_page -= 1
            self.main_driver.get(self.all_ppt[self.current_page])

    def openYouTube(self):
        options = Options()

        #전체 화면
        options.add_argument('--start-maximized')

        #pdf 위치
        get_youtube = "https://www.youtube.com/watch?v=X_jfbXA9mUM"

        self.link_driver = webdriver.Chrome(executable_path="./chromedriver", options = options)
        self.link_driver.implicitly_wait(1)
        self.link_driver.get(get_youtube)
        self.link_driver.find_element_by_css_selector('#movie_player > div.ytp-chrome-bottom > div.ytp-chrome-controls > div.ytp-left-controls > button').click()
    
    def closeYouTube(self):
        self.link_driver.quit()

    def endSlide(self):
        self.main_driver.close()
    
    def activate_command(self, gesture):
        self.gesture = gesture
        if self.gesture == Gesture2Command.gestures[0]:     # K
            # self.openFirstSlide()
            print('K')
        elif self.gesture == Gesture2Command.gestures[1]:   # L
            # self.nextSlide()
            print('L')
        elif self.gesture == Gesture2Command.gestures[2]:   # paper
            # self.previousSlide()
            print('paper')
        elif self.gesture == Gesture2Command.gestures[3]:   # rock
            # self.openYouTube()
            print('rock')
        elif self.gesture == Gesture2Command.gestures[4]:   # scissor
            # self.closeYouTube()
            print('scissor')
        elif self.gesture == Gesture2Command.gestures[5]:   # W
            # self.endSlide()
            print('W')
