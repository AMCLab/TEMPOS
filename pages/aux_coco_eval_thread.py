from PyQt6 import QtCore
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import io
from contextlib import redirect_stdout

class COCOEvalThread(QtCore.QThread):

    def __init__(self, dataset_eval_name, cfg, predictor, labelOutPut):
       
        QtCore.QThread.__init__(self)

        self.dataset_eval_name = dataset_eval_name

        self.cfg = cfg

        self.predictor = predictor

        self.labelOutPut = labelOutPut


    def run(self):

        evaluator = COCOEvaluator(self.dataset_eval_name, output_dir="./output")

        val_loader = build_detection_test_loader(self.cfg, self.dataset_eval_name)
        
        f = io.StringIO()

        with redirect_stdout(f):
            
            print(inference_on_dataset(self.predictor.model, val_loader, evaluator))

        # Get the output as a string
        output = f.getvalue()

        try:
            self.labelOutPut.appendPlainText(output)

            self.labelOutPut.appendPlainText("Finish Evaluation")
        except: pass