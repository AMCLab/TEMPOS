from PyQt6 import QtCore, QtWidgets
from PyQt6.QtWidgets import QApplication, QWidget, QLabel
from pages.page1_text import about
import markdown

class Page1(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        


        self.horizontalLayout = QtWidgets.QHBoxLayout(self)
        self.horizontalLayout.setObjectName("horizontalLayout")

        # Lateral Bar for design
        self.lateral_bar = QtWidgets.QWidget(self)
        # self.lateral_bar.setMinimumSize(QtCore.QSize(30, 0))
        self.lateral_bar.setObjectName("lateral_bar")
        self.horizontalLayout.addWidget(self.lateral_bar)

        self.main_widget = QtWidgets.QWidget(self)
        self.main_widget.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.main_widget.setObjectName("main_widget")

        self.gridLayout = QtWidgets.QGridLayout(self.main_widget)
        self.gridLayout.setObjectName("gridLayout")

        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")

        # Create About Title
        title_label = QLabel("<font color = '#335674'>TEM Synthetic Data Generator App</font color = '#335674'>", parent=self.main_widget)
        title_label.setStyleSheet("font-size: 28px;")
        self.verticalLayout.addWidget(title_label)

        # Create About Text
        text_box = QtWidgets.QTextBrowser(parent=self.main_widget)
        text_box.setObjectName("text_box")
        text_box.setReadOnly(True)
        text_box.setOpenExternalLinks(True)  # Enable opening links in a browser
        text_box.setFrameStyle(QtWidgets.QFrame.Shape.NoFrame) 
        
        # Import text from page1_text
        html = markdown.markdown(about)

        # Add CSS style to HTML
        css = "<style>body { color: #335674; }</style>"
        html_with_style = f"<html><head>{css}</head><body>{html}</body></html>"
        
        # Set HTML to text box
        text_box.setHtml(html_with_style)


        self.verticalLayout.addWidget(text_box)

        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)
        self.horizontalLayout.addWidget(self.main_widget)
   


if __name__ == '__main__':
    app = QApplication([])
    current_page = Page1()
    current_page.show()
    app.exec()