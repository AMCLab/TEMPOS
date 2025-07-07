from PyQt6 import QtWidgets


class NoiseInputDialog(QtWidgets.QDialog):

    def __init__(self, frequency, noise_octaves, noise_gain, noise_lacunarity):

        super().__init__()

        self.frequency_value = frequency
        self.noise_octaves_value = noise_octaves
        self.noise_gain_value = noise_gain
        self.noise_lacunarity_value = noise_lacunarity

        self.setupUi()

    def setupUi(self):

        self.setWindowTitle('Input Dialog')

        layout = QtWidgets.QFormLayout()
     
        self.doubleSpinBoxfrequency = QtWidgets.QDoubleSpinBox(parent=self)
        self.doubleSpinBoxfrequency.setObjectName("doubleSpinBoxfrequency")
        self.doubleSpinBoxfrequency.setRange(0.0, 100.0)
        self.doubleSpinBoxfrequency.setValue(self.frequency_value)
        layout.addRow("Noise Frequency = ", self.doubleSpinBoxfrequency)
        
        self.spinBoxnoise_octaves = QtWidgets.QSpinBox(parent=self)
        self.spinBoxnoise_octaves.setObjectName("spinBoxnoise_octaves")
        self.spinBoxnoise_octaves.setRange(0, 100)
        self.spinBoxnoise_octaves.setValue(self.noise_octaves_value)
        layout.addRow("Octaves = ", self.spinBoxnoise_octaves)
   
        self.spinBoxnoise_gain = QtWidgets.QDoubleSpinBox(parent=self)
        self.spinBoxnoise_gain.setObjectName("spinBoxnoise_gain")
        self.spinBoxnoise_gain.setRange(0.0, 100.0)
        self.spinBoxnoise_gain.setValue(self.noise_gain_value)
        layout.addRow("Gain = ", self.spinBoxnoise_gain)

        self.spinBoxnoise_lacunarity = QtWidgets.QDoubleSpinBox(parent=self)
        self.spinBoxnoise_lacunarity.setObjectName("spinBoxnoise_lacunarity")
        self.spinBoxnoise_lacunarity.setRange(0.0, 100.0)
        self.spinBoxnoise_lacunarity.setValue(self.noise_lacunarity_value)
        layout.addRow("Lacunarity = ", self.spinBoxnoise_lacunarity)

        self.buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self.resetButton = self.buttonBox.addButton('Reset', QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)
        self.resetButton.clicked.connect(self.reset_values)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        layout.addWidget(self.buttonBox)

        self.setLayout(layout)

    def reset_values(self):
        
        self.doubleSpinBoxfrequency.setValue(0.4)

        self.spinBoxnoise_octaves.setValue(5)

        self.spinBoxnoise_gain.setValue(0.1)

        self.spinBoxnoise_lacunarity.setValue(1.0)

    def getValues(self):

        return self.doubleSpinBoxfrequency.value(), self.spinBoxnoise_octaves.value(), self.spinBoxnoise_gain.value(), self.spinBoxnoise_lacunarity.value()

class ParticleNoiseInputDialog(QtWidgets.QDialog):

    def __init__(self, particle_frequency, particle_noise_octaves, particle_noise_gain, particle_noise_lacunarity):

        super().__init__()

        self.particle_frequency_value = particle_frequency
        self.particle_noise_octaves_value = particle_noise_octaves
        self.particle_noise_gain_value = particle_noise_gain
        self.particle_noise_lacunarity_value = particle_noise_lacunarity

        self.setupUi()

    def setupUi(self):

        self.setWindowTitle('Input Dialog')

        layout = QtWidgets.QFormLayout()
     
        self.doubleSpinBoxfrequency = QtWidgets.QDoubleSpinBox(parent=self)
        self.doubleSpinBoxfrequency.setObjectName("doubleSpinBoxfrequency")
        self.doubleSpinBoxfrequency.setRange(0.0, 100.0)
        self.doubleSpinBoxfrequency.setValue(self.particle_frequency_value)
        layout.addRow("Noise Frequency = ", self.doubleSpinBoxfrequency)
        
        self.spinBoxnoise_octaves = QtWidgets.QSpinBox(parent=self)
        self.spinBoxnoise_octaves.setObjectName("spinBoxnoise_octaves")
        self.spinBoxnoise_octaves.setRange(0, 100)
        self.spinBoxnoise_octaves.setValue(self.particle_noise_octaves_value)
        layout.addRow("Octaves = ", self.spinBoxnoise_octaves)
   
        self.spinBoxnoise_gain = QtWidgets.QDoubleSpinBox(parent=self)
        self.spinBoxnoise_gain.setObjectName("spinBoxnoise_gain")
        self.spinBoxnoise_gain.setRange(0.0, 100.0)
        self.spinBoxnoise_gain.setValue(self.particle_noise_gain_value)
        layout.addRow("Gain = ", self.spinBoxnoise_gain)

        self.spinBoxnoise_lacunarity = QtWidgets.QDoubleSpinBox(parent=self)
        self.spinBoxnoise_lacunarity.setObjectName("spinBoxnoise_lacunarity")
        self.spinBoxnoise_lacunarity.setRange(0.0, 100.0)
        self.spinBoxnoise_lacunarity.setValue(self.particle_noise_lacunarity_value)
        layout.addRow("Lacunarity = ", self.spinBoxnoise_lacunarity)

        self.buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self.resetButton = self.buttonBox.addButton('Reset', QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)
        self.resetButton.clicked.connect(self.reset_values)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        layout.addWidget(self.buttonBox)

        self.setLayout(layout)

    def reset_values(self):
        
        self.doubleSpinBoxfrequency.setValue(0.05)

        self.spinBoxnoise_octaves.setValue(15)

        self.spinBoxnoise_gain.setValue(0.5)

        self.spinBoxnoise_lacunarity.setValue(2.0)

    def getValues(self):

        return self.doubleSpinBoxfrequency.value(), self.spinBoxnoise_octaves.value(), self.spinBoxnoise_gain.value(), self.spinBoxnoise_lacunarity.value()
