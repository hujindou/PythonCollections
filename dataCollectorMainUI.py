import sys
import os

from PyQt5.QtWidgets import (QApplication, QWidget, QGridLayout, QPushButton, QLabel, QRadioButton, QButtonGroup)


class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'RealSense Data Collection'
        self.left = 10
        self.top = 10
        self.width = 1280
        self.height = 800
        self.initUI()

    def center(self):
        frameGm = self.frameGeometry()
        screen = QApplication.desktop().screenNumber(QApplication.desktop().cursor().pos())
        centerPoint = QApplication.desktop().screenGeometry(screen).center()
        frameGm.moveCenter(centerPoint)
        self.move(frameGm.topLeft())

    def yesButtonClick(self):
        option1 = None
        option2 = None
        if self.rbHavingFace.isChecked():
            option1 = 1
        if self.rbNotHavingFace.isChecked():
            option1 = 2
        if self.opencvDnn.isChecked():
            option2 = 1
        if self.dlibHog.isChecked():
            option2 = 2
        print(option1,option2)
        if option1 == 1 and option2 == 1:
            os.system("python3 depthDataCollection.py 11 /home/hjd/depthData/")
            #dc.collectDataUsingOpenCvDNN("/home/hjd/depthData/")
            pass
        elif option1 == 2 and option2 == 1:
            os.system("python3 depthDataCollection.py 21 /home/hjd/depthDataFail/")
            #dc.collectDataUsingOpenCvDNN("/home/hjd/depthDataFail/")
            pass
        elif option1 == 1 and option2 == 2:
            os.system("python3 depthDataCollection.py 12 /home/hjd/depthData/")
            #dc.collectDataUsingDlibHog("/home/hjd/depthData/")
            pass
        elif option1 == 2 and option2 == 2:
            os.system("python3 depthDataCollection.py 22 /home/hjd/depthDataFail/")
            #dc.collectDataUsingDlibHog("/home/hjd/depthDataFail/")
            pass
        pass

    def noButtonClick(self):
        pass

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        grid = QGridLayout()
        self.setLayout(grid)

        title = QLabel('Data Collection Type : ')
        grid.addWidget(title, 0, 0)

        self.rbHavingFace = QRadioButton("Collect Data with Face Depth Info")
        grid.addWidget(self.rbHavingFace, 0, 1)
        self.rbNotHavingFace = QRadioButton("Collect Data without Face Depth Info")
        grid.addWidget(self.rbNotHavingFace, 0, 2)
        rbGroup1 = QButtonGroup(self)
        rbGroup1.addButton(self.rbHavingFace)
        rbGroup1.addButton(self.rbNotHavingFace)

        title = QLabel('Face Rect Calculation Method : ')
        grid.addWidget(title, 1, 0)
        self.opencvDnn = QRadioButton("OpenCv Dnn")
        grid.addWidget(self.opencvDnn, 1, 1)
        self.dlibHog = QRadioButton("Dlib HOG")
        grid.addWidget(self.dlibHog, 1, 2)
        rbGroup2 = QButtonGroup(self)
        rbGroup2.addButton(self.opencvDnn)
        rbGroup2.addButton(self.dlibHog)

        buttonOk = QPushButton('YES')
        grid.addWidget(buttonOk, 2, 0)
        buttonCancel = QPushButton('NO')
        grid.addWidget(buttonCancel, 2, 2)
        buttonOk.clicked.connect(self.yesButtonClick)
        buttonCancel.clicked.connect(self.noButtonClick)

        self.center()
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    ex = App()
    sys.exit(app.exec_())  