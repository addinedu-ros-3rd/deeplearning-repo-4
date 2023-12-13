import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic

from_class = uic.loadUiType("haejo.ui")[0]

class WindowClass(QMainWindow, from_class) :
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Hello, Qt!")
        self.white_button.clicked.connect(self.change_to_white)
        self.dark_button.clicked.connect(self.change_to_black)
        self.db_tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        self.zeroto255 = [self.fx_button_play, self.groupBox, self.fx_button_phone, self.fx_button_door,self.fx_button_light, 
                          self.fx_button_snack, self.fx_button_desk, self.groupBox_2, self.db_comboBox, self.groupBox_3, 
                          self.db_date_from, self.db_date_to, self.db_label_for,self.db_button_search,self.db_tableWidget, self.title_label]

    
    def change_colors(self, color_rgb):
        for target in self.zeroto255:
            target.setStyleSheet(f"color: {color_rgb};")
    
    def change_to_white(self):
        self.change_colors("rgb(0, 0, 0)")
        self.setStyleSheet("background-color: rgb(245, 245, 245);")
        self.label_2.setStyleSheet("background-color: rgb(222, 221, 218);")
        self.video.setStyleSheet("background-color: rgb(255, 255, 255); ")
        self.label.setStyleSheet("background-color: rgb(222, 221, 218);  ")
        
    def change_to_black(self):
        self.change_colors("rgb(255, 255, 255)") 
        self.setStyleSheet("background-color: rgb(34, 33, 39);")
        self.label_2.setStyleSheet("background-color: rgb(50, 45, 58);")
        self.video.setStyleSheet("background-color: rgb(0, 0, 0); ")
        self.label.setStyleSheet("background-color: rgb(50, 45, 58);  ")
    


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindows = WindowClass()
    myWindows.show()
    sys.exit(app.exec_())
