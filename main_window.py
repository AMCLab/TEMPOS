import sys
from PyQt6.QtWidgets import QMainWindow, QApplication, QMainWindow
from PyQt6.QtGui import QIcon, QPixmap

from sidebar_ui import Ui_MainWindow
import sqlite_utils
from pages.page1 import Page1
from pages.create_settings import CreateSettings
from pages.generate_data import GenerateData
from pages.segment_images import SegmentImages
from pages.data_store import DataStore
from pages.ctf import CTFPage
from pages.export_datasette import ExportDatasette
from pages.train_model import TrainModel
from pages.evaluate_model import EvaluateModel
from pages.post_processing import PostProcessing
import os


basedir = os.path.dirname(__file__)


# try:
#     from ctypes import *  # Only exists on Windows.
#     myappid = 'mycompany.myproduct.subproduct.version'
#     windll.Shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
# except ImportError:
#     pass
        
class MainWindow(QMainWindow):

    def __init__(self):
        
        super(MainWindow, self).__init__()

        # Set window title
        self.setWindowTitle("TEMPOS")
        # Set window icon
        self.setWindowIcon(QIcon(os.path.join(basedir, "icons", "TEMPOS_small.ico")))

        self.data_store = DataStore()

        self.initUI()

    def initUI(self):

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.icon_widget.hide()
        self.ui.stackedWidget.setCurrentIndex(0)
        self.ui.home_Button.setChecked(True)

        self.page1 = Page1()
        self.ui.stackedWidget.addWidget(self.page1)

        self.create_settings = CreateSettings(self.data_store)
        self.ui.stackedWidget.addWidget(self.create_settings)

        self.generate_data = GenerateData(self.data_store)
        self.ui.stackedWidget.addWidget(self.generate_data)

        self.ctf_page = CTFPage(self.data_store)
        self.ui.stackedWidget.addWidget(self.ctf_page)

        self.train_model = TrainModel()
        self.ui.stackedWidget.addWidget(self.train_model)

        self.evaluate_model = EvaluateModel()
        self.ui.stackedWidget.addWidget(self.evaluate_model)

        self.segment_images = SegmentImages()
        self.ui.stackedWidget.addWidget(self.segment_images)

        self.post_processing = PostProcessing()
        self.ui.stackedWidget.addWidget(self.post_processing)    

        self.export_datasette = ExportDatasette()
        self.ui.stackedWidget.addWidget(self.export_datasette)       

         


    ## functions for changing menu page
    def on_home_icon_Button_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(0)

    def on_home_Button_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(0)
    
    def on_createSettings_icon_Button_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(1)

    def on_createSettings_Button_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(1)

    def on_generateData_icon_Button_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(2)

    def on_generateData_Button_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(2)

    def on_ctf_icon_Button_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(3)

    def on_ctf_Button_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(3)

    def on_train_icon_Button_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(4)

    def on_train_Button_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(4)

    def on_evaluate_icon_Button_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(5)

    def on_evaluate_Button_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(5)

    def on_segmentImages_icon_Button_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(6)

    def on_segmentImages_Button_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(6)

    def on_postProcessing_icon_Button_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(7)

    def on_postProcessing_Button_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(7)

    def on_exportDatasette_icon_Button_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(8)

    def on_exportDatasette_Button_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(8)

    def on_contact_icon_Button_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(9)

    def on_contact_Button_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(9)


if __name__ == '__main__':

    import multiprocessing
    
    multiprocessing.freeze_support()
    
    app = QApplication(sys.argv)


    window = MainWindow()

    app.setWindowIcon(QIcon(os.path.join(basedir, "icons", "TEMPOS.ico")))

    window.setWindowTitle("TEMPOS")

    window.show()
    # window.update()

    sys.exit(app.exec())
