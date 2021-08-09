import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import os

class Gesture2Command:
    gestures = ['K', 'L', 'paper', 'rock', 'scissor', 'W']
    
    def __init__(self, gesture = None):
        self.gesture = gesture
        self.current_page = -1
        self.ppt_path = os.path.join(os.getcwd(), 'TestPPT') 
        self.total_page = self.choosePPT()
        self.all_ppt = [os.path.join(self.ppt_path, i) for i in self.ppt_list]
        self.open_slide = False

    def choosePPT(self):
        self.ppt_list = [str(i) + '.png' for i in sorted([int(i.split('.')[0]) for i in os.listdir(self.ppt_path)])]
        return len(self.ppt_list)

    def openFirstSlide(self):
        if self.open_slide == False:
            self.open_slide = True
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
        if open_slide:
            self.main_driver.close()
            self.open_slide = False
        
    def activate_command(self, gesture):
        self.gesture = gesture
        if self.gesture == Gesture2Command.gestures[0]:
            self.openFirstSlide()
        elif self.gesture == Gesture2Command.gestures[1]:
            self.nextSlide()
        elif self.gesture == Gesture2Command.gestures[2]:
            self.previousSlide()
        elif self.gesture == Gesture2Command.gestures[3]:
            self.openYouTube()
        elif self.gesture == Gesture2Command.gestures[4]:
            self.closeYouTube()
        elif self.gesture == Gesture2Command.gestures[5]:
            self.endSlide()