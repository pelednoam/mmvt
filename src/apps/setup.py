import sys
from PyQt5.QtWidgets import QWidget, QMessageBox, QApplication, QDesktopWidget, QCheckBox, QPushButton, QLabel
from PyQt5.QtGui import QPixmap


class Example(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        '''
        The creation of the GUI
        '''

        label = QLabel(self)
        pixmap = QPixmap('logo.png')
        label.setPixmap(pixmap)
        label.move(140, 10)

        cb = QCheckBox('Check Box 1', self)
        cb.move(20, 30)
        cb.toggle()

        cb = QCheckBox('Check Box 2', self)
        cb.move(20, 50)
        cb.toggle()


        cb = QCheckBox('Check Box 3', self)
        cb.move(20, 70)
        cb.toggle()

        cb = QCheckBox('Check Box 4', self)
        cb.move(20, 90)
        cb.toggle()

        btn = QPushButton('Install', self)
        btn.move(20, 140)


        self.resize(300, 190)
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