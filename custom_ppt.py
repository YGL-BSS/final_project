import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import os
import pyautogui

class StartEnd:
    def __init__(self, motion, exepath, pptpath):
        self.motion = motion
        self.exepath = exepath
        self.pptpath = pptpath
        self.is_open_ppt = False

    def g2c(self,given_motion):
        if self.motion == given_motion:
            if self.is_open_ppt:
                pyautogui.press('esc')
                pyautogui.hotkey('alt', 'f4')
                self.is_open_ppt = False
            else:
                batch_ppt = f'start "{self.exepath}" "{self.pptpath}"'
                os.system(batch_ppt)
                time.sleep(4)
                pyautogui.press('f5')
                self.is_open_ppt = True

class NextSlide:
    def __init__(self, motion):
        self.motion = motion

    def g2c(self,given_motion):
        if self.motion == given_motion:
            pyautogui.press('right')

class PreviousSlide:
    def __init__(self, motion):
        self.motion = motion

    def g2c(self, given_motion):
        if self.motion == given_motion:
            pyautogui.press('left')

class AddMedia:
    def __init__(self, motion, exepath, mediapath):
        self.motion = motion
        self.exepath = exepath
        self.mediapath = mediapath
        self.is_open_media = False

    def g2c(self, given_motion):
        if self.motion == given_motion:
            if self.is_open_media:
                pyautogui.hotkey('alt', 'f4')
                self.is_open_media = False
            else:
                batch_media = f'start "{self.exepath}" "{self.mediapath}"'
                os.system(batch_media)
                self.is_open_media = True

class AddLink:
    def __init__(self, motion, link):
        self.motion = motion
        self.link = link
        self.is_open_link = False
    
    def g2c(self, given_motion):
        if self.motion == given_motion:
            if self.is_open_link:
                pyautogui.hotkey('alt', 'f4')
                self.is_open_link = False
            else:
                batch_link = 'start ' + self.link
                os.system(batch_link)
                self.is_open_link = True