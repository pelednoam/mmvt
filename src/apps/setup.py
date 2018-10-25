import sys
from PyQt5.QtWidgets import QWidget, QMessageBox, QApplication, QDesktopWidget, QCheckBox


class Example(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        '''
        The creation of the GUI
        '''

        cb = QCheckBox('Check Box', self)
        cb.move(20, 20)
        cb.toggle()


        self.resize(250, 150)
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