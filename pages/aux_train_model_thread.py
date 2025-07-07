from PyQt6 import QtCore
import os
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer

# Get one directory before "pages" as the basedir
basedir = os.path.split(os.path.dirname(__file__))[0]

class TrainModelThread(QtCore.QThread):


    def __init__(self, labelTrainingLog, dataset_train_name, dataset_test_name, spinBoxImgSize, spinBoxNumOfClasses, comboBoxDevice,
                    spinBoxBatchSize, doubleSpinBoxLR, comboBoxModelWeights, spinBoxMaxIter, spinBoxROIBS):
       
        QtCore.QThread.__init__(self)

        self.labelTrainingLog = labelTrainingLog

        self.dataset_train_name = dataset_train_name

        self.dataset_test_name = dataset_test_name
        
        self.spinBoxImgSize = spinBoxImgSize

        self.spinBoxNumOfClasses = spinBoxNumOfClasses

        self.comboBoxDevice = comboBoxDevice

        self.spinBoxBatchSize = spinBoxBatchSize

        self.doubleSpinBoxLR = doubleSpinBoxLR

        self.spinBoxMaxIter = spinBoxMaxIter

        self.spinBoxROIBS = spinBoxROIBS

        self.comboBoxModelWeights = comboBoxModelWeights

    def run(self):

        try: 

            cfg = get_cfg()
            cfg.merge_from_file(os.path.join(basedir, "detectron2Files", "mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
            #cfg.merge_from_file("detectron2Files/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
            cfg.DATASETS.TRAIN = (self.dataset_train_name,)
            if self.dataset_test_name != None:

                cfg.DATASETS.TEST = (self.dataset_test_name,)
            else:

                cfg.DATASETS.TEST = ()

            cfg.DATALOADER.NUM_WORKERS = 2

            cfg.INPUT.MAX_SIZE_TRAIN = int(self.spinBoxImgSize.value())
            cfg.INPUT.MAX_SIZE_TEST = int(self.spinBoxImgSize.value())

            cfg.MODEL.DEVICE = self.comboBoxDevice.currentText().lower()
            #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.comboBoxModelWeights)  # Let training initialize from model zoo
            cfg.MODEL.WEIGHTS = self.comboBoxModelWeights
            #cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final_highres.pth")
            cfg.SOLVER.IMS_PER_BATCH = self.spinBoxBatchSize  # This is the real "batch size" commonly known to deep learning people
            cfg.SOLVER.BASE_LR = self.doubleSpinBoxLR  # pick a good LR
            cfg.SOLVER.MAX_ITER = self.spinBoxMaxIter    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
            cfg.SOLVER.STEPS = []        # do not decay learning rate
            cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = self.spinBoxROIBS   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.spinBoxNumOfClasses.value()

            os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
            trainer = DefaultTrainer(cfg)
            trainer.resume_or_load(resume=False)
            trainer.train()

        except Exception as e: 
            print(e)
            pass