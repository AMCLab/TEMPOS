from PyQt6 import QtWidgets, QtCore

class ClusterTypeInputDialog(QtWidgets.QDialog):

    def __init__(self, cluster_type, rows_min, rows_max, cols_min, cols_max, width_min, width_max, cluster_random_num):

        super().__init__()

        self.cluster_type = cluster_type
        self.rows_min = rows_min
        self.rows_max = rows_max
        self.cols_min = cols_min
        self.cols_max = cols_max
        self.width_min = width_min
        self.width_max = width_max
        self.cluster_random_num = cluster_random_num

        self.setupUi()
        self.handle_cluster_type()

    def setupUi(self):

        self.setWindowTitle('Input Dialog')

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)

        #layout = QtWidgets.QFormLayout()
        self.gridLayout = QtWidgets.QGridLayout(self)
        self.gridLayout.setObjectName(u"gridLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        
        # Type of Cluster
        self.labelClusterType = QtWidgets.QLabel(self)
        self.labelClusterType.setObjectName(u"labelClusterType")
        self.labelClusterType.setText("Cluster Pattern = ")

        self.horizontalLayout.addWidget(self.labelClusterType)

        self.comboBoxClusterType = QtWidgets.QComboBox(parent=self)
        self.comboBoxClusterType.addItems(['Random', 'Honeycomb'])
        self.comboBoxClusterType.setCurrentText(self.cluster_type.title())
        self.comboBoxClusterType.setObjectName("comboBoxClusterType")
        self.comboBoxClusterType.setSizePolicy(sizePolicy)

        self.comboBoxClusterType.activated.connect(self.handle_cluster_type)
        
        self.horizontalLayout.addWidget(self.comboBoxClusterType)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)

        # Create a horizontal layout for the two spin boxes for number of rows
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        
        self.labelNumRows = QtWidgets.QLabel(self)
        self.labelNumRows.setObjectName(u"labelNumRows")
        self.labelNumRows.setText("Number of rows = ")
        self.labelNumRows.setVisible(False)
        self.horizontalLayout_2.addWidget(self.labelNumRows)

        # self.horizontalSpacer = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        # self.horizontalLayout_2.addItem(self.horizontalSpacer)

        self.SpinBoxNumRowsMin = QtWidgets.QSpinBox(self)
        self.SpinBoxNumRowsMin.setObjectName(u"SpinBoxNumRowsMin")
        self.SpinBoxNumRowsMin.setValue(self.rows_min)
        self.SpinBoxNumRowsMin.setMinimum(0)
        self.SpinBoxNumRowsMin.setMaximum(10000)
        self.SpinBoxNumRowsMin.setSizePolicy(sizePolicy)
        self.SpinBoxNumRowsMin.setVisible(False)

        self.horizontalLayout_2.addWidget(self.SpinBoxNumRowsMin)

        self.labelTo = QtWidgets.QLabel(self)
        self.labelTo.setObjectName(u"labelTo")
        self.labelTo.setText("to")
        self.labelTo.setVisible(False)
        self.horizontalLayout_2.addWidget(self.labelTo)

        self.SpinBoxNumRowsMax = QtWidgets.QSpinBox(self)
        self.SpinBoxNumRowsMax.setObjectName(u"SpinBoxNumRowsMax")
        self.SpinBoxNumRowsMax.setValue(self.rows_max)
        self.SpinBoxNumRowsMax.setMinimum(0)
        self.SpinBoxNumRowsMax.setMaximum(10000)
        self.SpinBoxNumRowsMax.setSizePolicy(sizePolicy)
        self.SpinBoxNumRowsMax.setVisible(False)

        self.horizontalLayout_2.addWidget(self.SpinBoxNumRowsMax)

        self.SpinBoxNumRowsMin.editingFinished.connect(lambda: self.handle_intervals(self.SpinBoxNumRowsMin, self.SpinBoxNumRowsMax))
        self.SpinBoxNumRowsMax.editingFinished.connect(lambda: self.handle_intervals(self.SpinBoxNumRowsMin, self.SpinBoxNumRowsMax))

        #add line to main layout
        self.gridLayout.addLayout(self.horizontalLayout_2, 1, 0, 1, 1)

        # Create a horizontal layout for the two spin boxes for number of cols
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        
        self.labelNumCols = QtWidgets.QLabel(self)
        self.labelNumCols.setObjectName(u"labelNumCols")
        self.labelNumCols.setText("Number of columns = ")
        self.labelNumCols.setVisible(False)
        self.horizontalLayout_3.addWidget(self.labelNumCols)

        # self.horizontalSpacer_2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        # self.horizontalLayout_3.addItem(self.horizontalSpacer_2)

        self.SpinBoxNumColsMin = QtWidgets.QSpinBox(self)
        self.SpinBoxNumColsMin.setObjectName(u"SpinBoxNumColsMin")
        self.SpinBoxNumColsMin.setValue(self.cols_min)
        self.SpinBoxNumColsMin.setRange(0, 1000)
        self.SpinBoxNumColsMin.setVisible(False)
        self.SpinBoxNumColsMin.setSizePolicy(sizePolicy)
        self.horizontalLayout_3.addWidget(self.SpinBoxNumColsMin)

        self.labelTo2 = QtWidgets.QLabel(self)
        self.labelTo2.setObjectName(u"labelTo")
        self.labelTo2.setText("to")
        self.labelTo2.setVisible(False)
        self.horizontalLayout_3.addWidget(self.labelTo2)

        self.SpinBoxNumColsMax = QtWidgets.QSpinBox(self)
        self.SpinBoxNumColsMax.setObjectName(u"SpinBoxNumColsMax")
        self.SpinBoxNumColsMax.setValue(self.cols_max)
        self.SpinBoxNumColsMax.setRange(0, 1000)
        self.SpinBoxNumColsMax.setVisible(False)
        self.SpinBoxNumColsMax.setSizePolicy(sizePolicy)
        self.horizontalLayout_3.addWidget(self.SpinBoxNumColsMax)

        self.SpinBoxNumColsMin.editingFinished.connect(lambda: self.handle_intervals(self.SpinBoxNumColsMin, self.SpinBoxNumColsMax))
        self.SpinBoxNumColsMax.editingFinished.connect(lambda: self.handle_intervals(self.SpinBoxNumColsMin, self.SpinBoxNumColsMax))

        #add line to main layout
        self.gridLayout.addLayout(self.horizontalLayout_3, 2, 0, 1, 1)

        # Create a horizontal layout for the two spin boxes for number of widht/height
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        
        self.labelNumwidth = QtWidgets.QLabel(self)
        self.labelNumwidth.setObjectName(u"labelNumwidth")
        self.labelNumwidth.setText("Width/Height spacing = ")
        self.labelNumwidth.setVisible(False)
        self.horizontalLayout_4.addWidget(self.labelNumwidth)

        # self.horizontalSpacer_3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        # self.horizontalLayout_4.addItem(self.horizontalSpacer_3)
                
        self.SpinBoxNumwidth_min = QtWidgets.QSpinBox(self)
        self.SpinBoxNumwidth_min.setObjectName(u"SpinBoxNumwidth_min")
        self.SpinBoxNumwidth_min.setValue(self.width_min)
        self.SpinBoxNumwidth_min.setRange(0, 1000)
        self.SpinBoxNumwidth_min.setVisible(False)
        self.SpinBoxNumwidth_min.setSizePolicy(sizePolicy)
        self.horizontalLayout_4.addWidget(self.SpinBoxNumwidth_min)

        self.labelTo3 = QtWidgets.QLabel(self)
        self.labelTo3.setObjectName(u"labelTo")
        self.labelTo3.setText("to")
        self.labelTo3.setVisible(False)
        self.horizontalLayout_4.addWidget(self.labelTo3)

        self.SpinBoxNumwidth_max = QtWidgets.QSpinBox(self)
        self.SpinBoxNumwidth_max.setObjectName(u"SpinBoxNumwidth_max")
        self.SpinBoxNumwidth_max.setValue(self.width_max)
        self.SpinBoxNumwidth_max.setRange(0, 1000)
        self.SpinBoxNumwidth_max.setVisible(False)
        self.SpinBoxNumwidth_max.setSizePolicy(sizePolicy)
        self.horizontalLayout_4.addWidget(self.SpinBoxNumwidth_max)

        self.SpinBoxNumwidth_min.editingFinished.connect(lambda: self.handle_intervals(self.SpinBoxNumwidth_min, self.SpinBoxNumwidth_max))
        self.SpinBoxNumwidth_max.editingFinished.connect(lambda: self.handle_intervals(self.SpinBoxNumwidth_min, self.SpinBoxNumwidth_max))

        #add line to main layout
        self.gridLayout.addLayout(self.horizontalLayout_4, 3, 0, 1, 1)

        # Create a horizontal layout for the two spin boxes for random cluster size
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        
        self.labelClusterRandomNum = QtWidgets.QLabel(self)
        self.labelClusterRandomNum.setObjectName(u"labelClusterRandomNum")
        self.labelClusterRandomNum.setText("Number of Particles in Cluster = ")
        self.labelNumwidth.setVisible(True)
        self.horizontalLayout_5.addWidget(self.labelClusterRandomNum)

        # self.horizontalSpacer_4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        # self.horizontalLayout_5.addItem(self.horizontalSpacer_4)

        self.SpinBoxClusterRandomNum = QtWidgets.QSpinBox(self)
        self.SpinBoxClusterRandomNum.setObjectName(u"SpinBoxClusterRandomNum")
        self.SpinBoxClusterRandomNum.setValue(self.cluster_random_num)
        self.SpinBoxClusterRandomNum.setRange(0, 1000)
        self.SpinBoxClusterRandomNum.setVisible(True)
        self.SpinBoxClusterRandomNum.setSizePolicy(sizePolicy)
        self.horizontalLayout_5.addWidget(self.SpinBoxClusterRandomNum)

        #add line to main layout
        self.gridLayout.addLayout(self.horizontalLayout_5, 4, 0, 1, 1)

        # Ok/Cancel/Reset dialog button
        self.buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self.resetButton = self.buttonBox.addButton('Reset', QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)
        self.resetButton.clicked.connect(self.reset_values)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.gridLayout.addWidget(self.buttonBox, 5, 0, 1, 1)

    def handle_cluster_type(self):

        if self.comboBoxClusterType.currentText() in ['Honeycomb']:
            
            self.labelNumRows.setVisible(True)
            self.SpinBoxNumRowsMin.setVisible(True)
            self.labelTo.setVisible(True)
            self.SpinBoxNumRowsMax.setVisible(True)
            self.labelNumCols.setVisible(True)
            self.SpinBoxNumColsMin.setVisible(True)
            self.labelTo2.setVisible(True)
            self.SpinBoxNumColsMax.setVisible(True)
            self.labelNumwidth.setVisible(True)
            self.SpinBoxNumwidth_min.setVisible(True)
            self.labelTo3.setVisible(True)
            self.SpinBoxNumwidth_max.setVisible(True)
            self.labelClusterRandomNum.setVisible(False)
            self.SpinBoxClusterRandomNum.setVisible(False)
        else:
            self.labelNumRows.setVisible(False)
            self.SpinBoxNumRowsMin.setVisible(False)
            self.labelTo.setVisible(False)
            self.SpinBoxNumRowsMax.setVisible(False)
            self.labelNumCols.setVisible(False)
            self.SpinBoxNumColsMin.setVisible(False)
            self.labelTo2.setVisible(False)
            self.SpinBoxNumColsMax.setVisible(False)
            self.labelNumwidth.setVisible(False)
            self.SpinBoxNumwidth_min.setVisible(False)
            self.labelTo3.setVisible(False)
            self.SpinBoxNumwidth_max.setVisible(False)
            self.labelClusterRandomNum.setVisible(True)
            self.SpinBoxClusterRandomNum.setVisible(True)

    def handle_intervals(self, min_spin_box, max_spin_box):

        min_value = min_spin_box.value()
       
        max_value = max_spin_box.value()

        if min_value >= max_value:
            
            if self.sender() == min_spin_box:
                
                max_spin_box.setValue(min_value + 1)
           
            else:
               
                min_spin_box.setValue(max_value - 1)

    def reset_values(self):
        
        self.comboBoxClusterType.setCurrentText('Random')
        self.SpinBoxNumRowsMin.setValue(2)
        self.SpinBoxNumRowsMax.setValue(7)
        self.SpinBoxNumColsMin.setValue(2)
        self.SpinBoxNumColsMax.setValue(7)
        self.SpinBoxNumwidth_min.setValue(2)
        self.SpinBoxNumwidth_max.setValue(7)
        self.SpinBoxClusterRandomNum.setValue(10)


    def getValues(self):

        cluster_type = self.comboBoxClusterType.currentText().lower()
        rows_min = self.SpinBoxNumRowsMin.value()
        rows_max = self.SpinBoxNumRowsMax.value()
        cols_min = self.SpinBoxNumColsMin.value()
        cols_max = self.SpinBoxNumColsMax.value()
        width_min = self.SpinBoxNumwidth_min.value()
        width_max = self.SpinBoxNumwidth_max.value()
        cluster_random_num = self.SpinBoxClusterRandomNum.value()

        return cluster_type, rows_min, rows_max, cols_min, cols_max, width_min, width_max, cluster_random_num
