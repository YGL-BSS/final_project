import os

import sys

from PyQt5.QtWidgets import *

from PyQt5.QtGui import *

from PyQt5.QtCore import *


class CustomPPT(QWidget):
    commands = ['COMMAND', 'StartEnd', 'NextSlide', 'PreviousSlide', 'AddMedia',
                'AddLink']
    
    motions = ['mtn', 'zero', 'one', 'two', 'three', 'four', 'five',
                'L', 'K']

    def __init__(self):

        super().__init__()

        self.current_class = None

        self.current_motion1 = None

        self.current_motion2 = None

        self.current_exepath = None

        self.current_mediapath = None

        self.current_link = None

        self.initUI()

    def initUI(self):

        self.lbl = QLabel('Custom your Presentation', self)

        self.lbl.setGeometry(70, 40, 350, 60)

        self.lbl.setFont(QFont("Arial Black", 14))

        self.subtitle = QLabel('Select command', self)

        self.subtitle.setGeometry(120, 160, 150, 30)

        self.subtitle.setFont(QFont("Arial", 11))


        # command list

        self.choose_class = QComboBox(self)

        self.choose_class.setGeometry(120, 200, 150, 30)

        for c in CustomPPT.commands:

            self.choose_class.addItem(c)


        self.choose_class.activated[str].connect(self.temp_command)


        # motion1 list

        self.choose_motion1 = QComboBox(self)

        self.choose_motion1.setGeometry(120, 250, 60, 25)

        for motion in CustomPPT.motions:

            self.choose_motion1.addItem(motion)


        self.choose_motion1.activated[str].connect(self.temp_motion1)


        self.motion1_lbl = QLabel('motion1', self)

        self.motion1_lbl.setGeometry(190,250, 60, 25)


        # motion2 list

        self.choose_motion2 = QComboBox(self)

        self.choose_motion2.setGeometry(120, 290, 60, 25)

        for motion in CustomPPT.motions:

            self.choose_motion2.addItem(motion)

        self.choose_motion2.activated[str].connect(self.temp_motion2)

        self.motion2_lbl = QLabel('motion2', self)

        self.motion2_lbl.setGeometry(190, 290, 60, 25)   

        # add extra settings

        self.extra_settings = []

        # current motion lists

        self.saved_commands = []

        self.motion_sub = QLabel('Motion lists',self)

        self.motion_sub.setGeometry(490, 160, 100, 16)

        self.motion_sub.setFont(QFont("Arial", 10))

        self.motion_list = QListView(self)

        self.motion_list.setGeometry(490, 180, 260, 370)

        self.motion_list.viewMode()

        self.motion_list.clicked.connect(self.want_del)

        # push button

        self.push_btn = QPushButton('Push', self)

        self.push_btn.setGeometry(340, 280, 75, 25)

        self.push_btn.clicked.connect(self.push_command)
        

        # delete button

        self.del_btn = QPushButton('Delete', self)

        self.del_btn.setGeometry(340, 390, 75, 25)

        self.del_btn.clicked.connect(self.del_command)

        # save button

        self.save_btn = QPushButton('Save', self)

        self.save_btn.setFont(QFont("Arial", 12))

        self.save_btn.setGeometry(120, 520, 60, 30)

        self.save_btn.clicked.connect(self.save_custom)


        # Basic settings

        self.setWindowTitle("Custom PPT")

        self.setGeometry(200, 200, 800, 600)

        self.show()

    def save_custom(self):
        with open("customPPT.txt", "w") as f:
            for saved_command in self.saved_commands:
                f.write(saved_command + '\n')
        
        self.save_confirm()
    
    def want_del(self, index):
        self.wantdel = index.row()

    def del_command(self):
        if self.saved_commands:
            del self.saved_commands[self.wantdel]
        
        self.model = QStandardItemModel()
        i = 1
        for saved_command in self.saved_commands:
            self.model.appendRow(QStandardItem(f'{i}. '+ saved_command))
            i += 1
        self.motion_list.setModel(self.model)


    def push_command(self):
        if self.valid_test():
            self.saved_commands.append(self.get_current_command())
            self.model = QStandardItemModel()
            i = 1
            for saved_command in self.saved_commands:
                self.model.appendRow(QStandardItem(f'{i}. '+ saved_command))
                i += 1
            self.motion_list.setModel(self.model)

    def get_current_command(self):
        command = self.current_class
        motion1 = self.current_motion1
        motion2 = self.current_motion2
        exepath = self.current_exepath
        mediapath = self.current_mediapath
        link = self.current_link

        current_result = [command, motion1, motion2, exepath, mediapath, link]
        current_result = [i for i in current_result if i != None]
        return "|".join(current_result)

    def valid_test(self):
        if self.current_class == None:
            self.push_alert_box()
            return False
        else:
            if self.current_motion1 == None and self.current_motion2 == None:
                self.push_alert_box()
                return False
        return True
            
    def temp_command(self, text):

        self.reset_currents()

        if text == CustomPPT.commands[0]:

            self.current_class = None

            # print(self.current_class)

        else:
        
            self.current_class = text

            # print(self.current_class)
        
        self.add_extra_settings(text)

    def temp_motion1(self, text):

        if text == CustomPPT.motions[0]:

            self.current_motion1 = None

        else:

            self.current_motion1 = text

    def temp_motion2(self, text):

        if text == CustomPPT.motions[0]:

            self.current_motion2 = None

        else:

            self.current_motion2 = text

    def add_extra_settings(self, command):
        
        if self.extra_settings:
            for extra_setting in self.extra_settings:
                extra_setting.hide()
            self.extra_settings = []

        if command == 'AddMedia':
            
            # exe path
            self.exe_btn = QPushButton('exe',self)
            self.exe_btn.setGeometry(118, 330, 63, 25)
            self.exe_btn.clicked.connect(self.exe_fileopen)

            self.exe_view = QLabel('none', self)
            self.exe_view.setGeometry(190, 330, 100, 25)
            self.exe_view.setStyleSheet("color: #FF5733; border-style: solid; border-width: 2px; border-color: #FFC300; border-radius: 10px; ")

            self.media_btn = QPushButton('media',self)
            self.media_btn.setGeometry(118, 370, 63, 25)
            self.media_btn.clicked.connect(self.media_fileopen)

            self.media_view = QLabel('none',self)
            self.media_view.setGeometry(190, 370, 100, 25)
            self.media_view.setStyleSheet("color: #FF5733; border-style: solid; border-width: 2px; border-color: #FFC300; border-radius: 10px; ")

            self.extra_settings.append(self.exe_btn)
            self.extra_settings.append(self.exe_view)
            self.extra_settings.append(self.media_btn)
            self.extra_settings.append(self.media_view)

            for extra_setting in self.extra_settings:
                extra_setting.show()

        elif command == 'AddLink':
            self.open_type_link = QPushButton('link', self)
            self.open_type_link.setGeometry(118, 330, 63, 25)
            self.extra_settings.append(self.open_type_link)

            self.open_type_link.clicked.connect(self.input_link)
            self.link_view = QLabel('none', self)
            self.link_view.setGeometry(190, 330, 100, 25)
            self.link_view.setStyleSheet("color: #FF5733; border-style: solid; border-width: 2px; border-color: #FFC300; border-radius: 10px; ")
            self.extra_settings.append(self.link_view)

            for extra_setting in self.extra_settings:
                extra_setting.show()
            

    def exe_fileopen(self):

        filename =QFileDialog.getOpenFileName(self, 'Open File')
        if filename[0].split('.')[-1] != 'exe':
            self.alert_box()
            self.exe_view.setText('none')
            self.current_exepath = None
        else:
            #print(filename[0])
            self.exe_view.setText(filename[0])
            self.current_exepath = filename[0]
    
    def media_fileopen(self):

        valid = ['mp4', 'wmv', 'avi', 'mkv', 'jpg', 'jpeg', 'png', 'gif', 'bmp']
        filename =QFileDialog.getOpenFileName(self, 'Open File')
        if filename[0].split('.')[-1] not in valid:
            self.alert_box()
            self.media_view.setText('none')
            self.current_mediapath = None
        else:
            self.media_view.setText(filename[0])
            self.current_mediapath = filename[0]

    def alert_box(self):
        QMessageBox.warning(self, "Warnig", "Invalid path", QMessageBox.Cancel)

    def push_alert_box(self):
        QMessageBox.warning(self, "Warning", "Select at least one command and one motion", QMessageBox.Cancel)

    def input_link(self):
        text, ok = QInputDialog.getText(self, 'Input link', 'Enter your link:')

        if ok:
            self.link_view.setText(str(text))
            self.current_link = str(text)

    def reset_currents(self):

        self.current_class = None

        self.current_motion1 = None

        self.current_motion2 = None

        self.current_exepath = None

        self.current_mediapath = None

        self.current_link = None

    def save_confirm(self):
        QMessageBox.information(self, "Info", "Complete settings", QMessageBox.Cancel)

if __name__ == '__main__':

    app = QApplication(sys.argv)

    cp = CustomPPT()

    sys.exit(app.exec_())