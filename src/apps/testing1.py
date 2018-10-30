import sys
from PyQt5.QtWidgets import QVBoxLayout, QMainWindow, QAction, qApp, QApplication, QWidget, QMessageBox, QDesktopWidget, QCheckBox, QPushButton, QLabel
from PyQt5.QtGui import QPixmap


class Example(QMainWindow):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        exitAct = QAction('&Exit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit application')
        exitAct.triggered.connect(qApp.quit)

        self.statusBar()

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAct)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        lay = QVBoxLayout(self.central_widget)

        label = QLabel(self)
        pixmap = QPixmap('logo.png')
        label.setPixmap(pixmap)

        cb1 = QCheckBox('Check Box 1', self)
        cb1.toggle()

        cb2 = QCheckBox('Check Box 2', self)
        cb2.toggle()

        cb3 = QCheckBox('Check Box 3', self)
        cb3.toggle()

        cb4 = QCheckBox('Check Box 4', self)
        cb4.toggle()

        btn = QPushButton('Install', self)

        lay.addWidget(cb1)
        lay.addWidget(cb2)
        lay.addWidget(cb3)
        lay.addWidget(cb4)
        lay.addWidget(btn)
        lay.addWidget(label)

        self.resize(300, 300)
        self.setWindowTitle('Setup')
        self.center()
        self.show()

    def center(self):
        '''
        Figures out the screen resolution of our monitor and centeres the window
        '''
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def closeEvent(self, event):
        '''
        Shows a message box to confirm the wondow closing
        '''

        reply = QMessageBox.question(self, 'Message',
                                     "Are you sure to quit?", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())