from PyQt5 import uic
from urllib.parse import urlparse
import sqlite3
import os
from datetime import date, datetime, timedelta
from PyQt5.QtWidgets import (QApplication, QComboBox, QDialog, QWidget, QListWidgetItem, QMessageBox,
                             QDialogButtonBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
                             QLabel, QLineEdit, QMenu, QMenuBar, QPushButton, QSpinBox, QTextEdit, QSizePolicy,
                             QListWidget, QVBoxLayout)

import PyQt5.QtCore

MIN_FILE_SIZE = 100 * 1024 * 1024

DBNAME = "fileSum.db"
if os.path.isfile(DBNAME):
    os.remove(DBNAME)

conn = sqlite3.connect(DBNAME)

cursor = conn.cursor()

cursor.execute('''CREATE TABLE fileinfoindex
                         (filefullname TEXT PRIMARY KEY,
                          filename TEXT,
                          filesize INT,
                          uniquefilename TEXT
                          )''')

PATHLIST = [
    # "/home/hjd/mnt/home/hjd/Videos",
    "/run/media/hjd/tempuse",
    "/home/hjd/Videos",
    "/home/hjd/mnt",
    "/mnt/4ef7523c-5f60-46a7-b091-0f382601d6c6"
]

colorDic = {1: "background-color:#FA8072;", 2: "background-color:#008000;"}


class FileItem(QWidget):
    def openFile(self):
        os.system("vlc \"" + self.filenameFull + "\"")
        pass

    def deleteFile(self):
        os.remove(self.filenameFull)
        pass

    def __init__(self, filename, filenameFull, colorIndex=-1, parent=None):
        super(FileItem, self).__init__(parent)
        self.filename = filename
        self.filenameFull = filenameFull

        self.filenameLabel = QLabel(filename)
        self.filenameFullLabel = QLabel(filenameFull)

        self.buttonPlay = QPushButton('Play')
        self.buttonDel = QPushButton('Del')

        self.buttonDel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.buttonPlay.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self.buttonDel.setStyleSheet("background-color:#ff0000;")

        self.buttonPlay.clicked.connect(self.openFile)

        self.buttonDel.clicked.connect(self.deleteFile)

        if colorIndex == 1 or colorIndex == 2:
            self.filenameLabel.setStyleSheet(colorDic[colorIndex]);
            self.filenameFullLabel.setStyleSheet(colorDic[colorIndex]);

        mainLayout = QGridLayout()

        mainLayout.setColumnStretch(0, 10)
        mainLayout.setColumnStretch(1, 10)
        mainLayout.setColumnStretch(2, 300)

        mainLayout.addWidget(self.buttonPlay, 0, 0, 2, 1)
        mainLayout.addWidget(self.buttonDel, 0, 1, 2, 1)

        mainLayout.addWidget(self.filenameLabel, 0, 2, 1, 1)
        mainLayout.addWidget(self.filenameFullLabel, 1, 2, 1, 1)

        # self.setStyleSheet("background-color:#FFFF00;");

        self.setLayout(mainLayout)
        pass

    pass


class Dialog(QDialog):
    NumGridRows = 3
    NumButtons = 4

    WindoWidth = 1600
    WindowHeight = 1000

    def __init__(self):
        super(Dialog, self).__init__()

        # self.createMenu()
        self.createHorizontalGroupBox()
        # self.createGridGroupBox()
        # self.createFormGroupBox()

        bigEditor = QTextEdit()
        bigEditor.setPlainText("This widget takes up all the remaining space "
                               "in the top-level layout.")

        self.fileListView = QListWidget()
        item1 = FileItem("testFilename1", "testFileNameFull1")
        item2 = FileItem("adsfasdf", "adsfasdfasdfasdf")
        item3 = FileItem("asdfasdfsdf", "asdfasdfasdfasdfsdf")

        viewItem1 = QListWidgetItem()
        viewItem2 = QListWidgetItem()
        viewItem3 = QListWidgetItem()

        viewItem1.setSizeHint(item1.sizeHint())
        viewItem2.setSizeHint(item2.sizeHint())
        viewItem3.setSizeHint(item3.sizeHint())

        self.fileListView.addItem(viewItem1)
        self.fileListView.addItem(viewItem2)
        self.fileListView.addItem(viewItem3)

        self.fileListView.setItemWidget(viewItem1, item1)
        self.fileListView.setItemWidget(viewItem2, item2)
        self.fileListView.setItemWidget(viewItem3, item3)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

        mainLayout = QVBoxLayout()
        # mainLayout.setMenuBar(self.menuBar)
        mainLayout.addWidget(self.horizontalGroupBox)
        # mainLayout.addWidget(self.gridGroupBox)
        # mainLayout.addWidget(self.formGroupBox)
        mainLayout.addWidget(self.fileListView)
        mainLayout.addWidget(buttonBox)

        self.fileListView.clear()

        self.setLayout(mainLayout)

        self.setWindowTitle("Basic Layouts")

        self.resize(Dialog.WindoWidth, Dialog.WindowHeight)

    def createMenu(self):
        self.menuBar = QMenuBar()

        self.fileMenu = QMenu("&File", self)
        self.exitAction = self.fileMenu.addAction("E&xit")
        self.menuBar.addMenu(self.fileMenu)

        self.exitAction.triggered.connect(self.accept)

    def getAllFiles(self, directoryName):

        result = []
        if not os.path.isdir(directoryName):
            return result

        for currentdir, dirs, files in os.walk(directoryName):
            for file in files:
                tmpfullname = os.path.join(currentdir, file)
                tmpsize = os.path.getsize(tmpfullname)
                if tmpsize >= MIN_FILE_SIZE:
                    result.append(
                        (tmpfullname, file, tmpsize, file.replace("4k","")
                         .replace("4K", "")
                         .replace("1080", "")
                         .replace("2160", "")
                         )
                    )

        return result

    def writeFileInfosIntoDatabase(self, filesinfos):
        insertsql = """  INSERT INTO fileinfoindex
                                    (filefullname,
                                     filename,
                                     filesize,uniquefilename)
                                    VALUES
                                    (?,?,?,?) """
        cursor.executemany(insertsql, filesinfos)
        conn.commit()
        pass

    def clearDatabase(self):
        updateSql = """ DELETE FROM fileinfoindex """
        cursor.execute(updateSql)
        conn.commit()
        pass

    def getFilesWithEqualSize(self):
        selectsql = """ SELECT filefullname,  filename, filesize
                               FROM fileinfoindex
                               WHERE
                                    filesize IN (SELECT 
                                                    filesize
                                                FROM
                                                    fileinfoindex
                                                GROUP BY filesize
                                                HAVING COUNT(*) > 1) ORDER BY filesize DESC """
        tmpJoblist = []
        for (filefullname, filename, filesize) in cursor.execute(selectsql):
            tmpJoblist.append(
                {"filefullname": filefullname, "filesize": filesize, "filename": filename})
            pass
        return tmpJoblist

    def getAllFilesSortWithSize(self):
        selectsql = """ SELECT filefullname,  filename, filesize
                               FROM fileinfoindex ORDER BY filesize DESC
                    """
        # selectsql = """ SELECT filefullname,  filename, filesize
        #                                FROM fileinfoindex
        #                                WHERE
        #                                     uniquefilename IN (SELECT
        #                                                     uniquefilename
        #                                                 FROM
        #                                                     fileinfoindex
        #                                                 GROUP BY uniquefilename
        #                                                 HAVING COUNT(*) > 1) ORDER BY uniquefilename DESC """
        tmpJoblist = []
        for (filefullname, filename, filesize) in cursor.execute(selectsql):
            tmpJoblist.append(
                {"filefullname": filefullname, "filesize": filesize, "filename": filename})
            pass
        return tmpJoblist

    def reload(self):
        quit_msg = "Are you sure reload Database information ?"
        reply = QMessageBox.question(self, 'Message',
                                     quit_msg, QMessageBox.Yes, QMessageBox.No)

        if reply == QMessageBox.Yes:

            result = self.getFilesWithEqualSize()

            self.fileListView.clear()

            tmpidx = 0
            currentColor = 1
            for itm in result:
                if tmpidx == 0:
                    currentColor = 1
                    pass
                else:
                    if itm["filesize"] != result[tmpidx - 1]["filesize"]:
                        if currentColor == 1:
                            currentColor = 2
                        else:
                            currentColor = 1
                        pass
                    pass

                item1 = FileItem(str(itm["filesize"]) + "  " + itm["filename"], itm["filefullname"], currentColor)

                viewItem1 = QListWidgetItem()

                viewItem1.setSizeHint(item1.sizeHint())

                self.fileListView.addItem(viewItem1)

                self.fileListView.setItemWidget(viewItem1, item1)

                tmpidx += 1

                pass
            pass
        else:
            pass
        pass

    def listFileBySize(self):
        result = self.getAllFilesSortWithSize()
        self.fileListView.clear()
        for itm in result:
            item1 = FileItem(str(itm["filesize"]) + "  " + itm["filename"], itm["filefullname"])
            viewItem1 = QListWidgetItem()
            viewItem1.setSizeHint(item1.sizeHint())
            self.fileListView.addItem(viewItem1)
            self.fileListView.setItemWidget(viewItem1, item1)
        pass

    pass

    def rescan(self):
        quit_msg = "Rescan will clear all info, and it takes quite a lot time to create new info"
        reply = QMessageBox.question(self, 'Message',
                                     quit_msg, QMessageBox.Yes, QMessageBox.No)

        if reply == QMessageBox.Yes:

            self.clearDatabase()

            for rootdir in PATHLIST:
                tmpitems = self.getAllFiles(rootdir)
                self.writeFileInfosIntoDatabase(tmpitems)
                pass
            pass
        else:
            pass

    def createHorizontalGroupBox(self):
        self.horizontalGroupBox = QGroupBox("Function Buttons")
        layout = QHBoxLayout()
        button1 = QPushButton("ReScan Files and Update Database")
        button1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        button1.clicked.connect(self.rescan)

        button2 = QPushButton("ShowDBInfo")
        button2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        button2.clicked.connect(self.reload)

        button3 = QPushButton("List File By Size")
        button3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        button3.clicked.connect(self.listFileBySize)

        button4 = QPushButton("Settings")
        button4.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        layout.addWidget(button1)
        layout.addWidget(button2)
        layout.addWidget(button3)
        layout.addWidget(button4)
        self.horizontalGroupBox.setLayout(layout)

    def createGridGroupBox(self):
        self.gridGroupBox = QGroupBox("Grid layout")
        layout = QGridLayout()

        for i in range(Dialog.NumGridRows):
            label = QLabel("Line %d:" % (i + 1))
            lineEdit = QLineEdit()
            layout.addWidget(label, i + 1, 0)
            layout.addWidget(lineEdit, i + 1, 1)

        self.smallEditor = QTextEdit()
        self.smallEditor.setPlainText("This widget takes up about two thirds "
                                      "of the grid layout.")

        layout.addWidget(self.smallEditor, 0, 2, 4, 1)

        layout.setColumnStretch(1, 10)
        layout.setColumnStretch(2, 20)
        self.gridGroupBox.setLayout(layout)

    def createFormGroupBox(self):
        self.formGroupBox = QGroupBox("Form layout")
        layout = QFormLayout()
        layout.addRow(QLabel("Line 1:"), QLineEdit())
        layout.addRow(QLabel("Line 2, long text:"), QComboBox())
        layout.addRow(QLabel("Line 3:"), QSpinBox())
        self.formGroupBox.setLayout(layout)

    def resizeEvent(self, QResizeEvent):
        self.setWindowTitle(
            "Basic Layouts " + str(self.frameGeometry().width()) + " x " + str(self.frameGeometry().height()))
        # print(self.frameGeometry().width(), " ", self.frameGeometry().height())


if __name__ == '__main__':
    import sys

    # sys.exit(0)

    app = QApplication(sys.argv)
    dialog = Dialog()
    sys.exit(dialog.exec_())
