import os
import sys
from PyQt5.QtWidgets import QWidget, QApplication, QVBoxLayout, QPushButton
from PyQt5.QtWidgets import QLabel, QFileDialog, QListView, QMessageBox
from PyQt5.QtGui import QFont, QFontDatabase, QIcon
from PyQt5.QtCore import QSize

class SetEvironment(QWidget):
    def __init__(self):
        super().__init__()

        self.exe_path = None
        self.ppt_path = None

        self.PATH_BASE = os.path.dirname(__file__)
        self.icon_image = 'folder.png'

        self.initUI()

    def initUI(self):
        self.lbl = QLabel('Set your Evironment', self)
        self.lbl.setGeometry(70, 40, 300, 60)
        self.lbl.setFont(QFont("Arial Black", 14))
        
        # exe path
        self.exe_lbl = QLabel('Coose your exe path', self)
        self.exe_lbl.setGeometry(90, 200, 160, 30)
        self.exe_lbl.setFont(QFont("Arial", 10))

        # exe 파일지정 GUI
        self.exe_btn = QPushButton('', self)
        self.exe_btn.setGeometry(90,240,30,30)
        self.exe_btn.clicked.connect(self.exe_fileopen)
        self.exe_btn.setIcon(QIcon(os.path.join(self.PATH_BASE, self.icon_image)))
        self.exe_btn.setIconSize(QSize(28,28))
        
        # exe 지정된 파일 보여주기
        self.exe_view = QLabel('None', self)
        self.exe_view.setGeometry(130, 240, 400, 30)
        self.exe_view.setStyleSheet("color: #FF5733; border-style: solid; border-width: 2px; border-color: #FFC300; border-radius: 10px; ")

        # ppt path
        self.ppt_lbl = QLabel('Coose your ppt path', self)
        self.ppt_lbl.setGeometry(90, 350, 160, 30)
        self.ppt_lbl.setFont(QFont("Arial", 10))

        # ppt 파일지정 GUI
        self.ppt_btn = QPushButton('', self)
        self.ppt_btn.setGeometry(90, 390, 30, 30)
        self.ppt_btn.clicked.connect(self.ppt_fileopen)
        self.ppt_btn.setIcon(QIcon(os.path.join(self.PATH_BASE, self.icon_image)))
        self.ppt_btn.setIconSize(QSize(28, 28))    

        # ppt 지정된 파일 보여주기
        self.ppt_view = QLabel('None', self)
        self.ppt_view.setGeometry(130, 390, 400, 30)
        self.ppt_view.setStyleSheet("color: #FF5733; border-style: solid; border-width: 2px; border-color: #FFC300; border-radius: 10px; ")

        # save button
        self.save_btn = QPushButton('Save', self)
        self.save_btn.setFont(QFont("Arial", 12))
        self.save_btn.setGeometry(690, 520, 60, 30)
        self.save_btn.clicked.connect(self.environment_save)

        self.setWindowTitle("Set Evironment")
        self.setGeometry(200, 200, 800, 600)
        self.show()

    def exe_fileopen(self):
        filename =QFileDialog.getOpenFileName(self, 'Open File')
        if filename[0].split('.')[-1] != 'exe':
            self.alert_box()
            self.exe_view.setText('None')
            self.exe_path = None
        else:
            #print(filename[0])
            self.exe_view.setText(filename[0])
            self.exe_path = filename[0]

    def ppt_fileopen(self):
        valid = ['pptx', 'ppt', 'pdf', 'pptm']
        filename =QFileDialog.getOpenFileName(self, 'Open File')
        if filename[0].split('.')[-1] not in valid:
            self.alert_box()
            self.ppt_view.setText('None')
            self.ppt_path = None
        else:
            self.ppt_view.setText(filename[0])
            self.ppt_path = filename[0]
    
    def alert_box(self):
        QMessageBox.warning(self, "Warnig", "Invalid path", QMessageBox.Cancel)

    def save_confirm(self):
        QMessageBox.information(self, "Info", "Complete settings", QMessageBox.Cancel)

    def environment_save(self):
        if self.exe_path == None or self.ppt_path == None:
            self.alert_box()
        else:
            save_name = 'ppt_environment_setting.txt'
            with open(os.path.join(self.PATH_BASE,save_name), "w") as f:
                f.write(f'se|{self.exe_path}|{self.ppt_path}')
            self.save_confirm()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    se = SetEvironment()
    sys.exit(app.exec_())    