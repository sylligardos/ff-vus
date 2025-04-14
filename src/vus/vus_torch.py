"""
@who: Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: PhD Candidate, 2nd year (2025)
@what: FF-VUS
"""


import torch
from utils.utils import time_it

class VUSTorch():
    def __init__(
            self, 
            slope_size=100, 
            step=1, 
            zita=(1/torch.sqrt(torch.tensor(2))), 
            conf_matrix='dynamic', 
        ):
        """
        Initialize the torch version of the VUS metric.

        Args:
            TODO: Write the arguments description when done
        """
        if slope_size < 0:
            raise ValueError(f"Error with the slope_size: {slope_size}. It should be a positive value.")
        if step < 0 or (slope_size % step) != 0:
            raise ValueError(f"Error with the step: {step}. It should be a positive value and a perfect divider of the slope_size.")
        if zita < 0 and zita > 1:
            raise ValueError(f"Error with the zita: {zita}. It should be a value between 0 and 1.")
        
        self.slope_size = slope_size
        self.step = step
        self.zita = zita
        
        self.slope_values = torch.arange(start=0, end=self.slope_size + 1, step=self.step)
        self.n_slopes = self.slope_values.shape[0]

        conf_matrix_args = ['dynamic', 'dynamic_plus']
        if conf_matrix in conf_matrix_args:
            self.conf_matrix_mode = conf_matrix
        else:
            raise ValueError(f"Unknown argument for conf_matrix: {conf_matrix}. 'conf_matrix' should be one of {conf_matrix_args}")
    
    def compute(self, label, score):
        """
        TODO: The main computing function of the metric
        """
        thresholds, time_thresholds = time_it(self.get_unique_thresholds)(score)
        sm, time_sm = time_it(self.get_score_mask)(score, thresholds)
        # compute anomaly indexes
        # Compute position
        
        labels, time_slopes = time_it(self.add_slopes)(label)
        existence, time_existence = time_it(self.compute_existence)(labels, sm, score, thresholds)
        (fp, fn, tp, positives, negatives, fpr), time_confusion = time_it(self.compute_confusion_matrix)(labels, sm)
        (precision, recall), time_pr_rec = time_it(self.precision_recall_curve)(tp, fp, positives, existence)
        vus_pr, time_integral = time_it(self.auc)(recall, precision)
        
        time_analysis = {
            "Thresholds time": time_thresholds,
            "Score mask time": time_sm,
            "Slopes time": time_slopes,
            "Existence time": time_existence,
            "Confusion matrix time": time_confusion,
            "Precision recall curve time": time_thresholds,
            "Integral time": time_integral,
        }

        return vus_pr, time_analysis
    
    def get_score_mask(self, score, thresholds):
        return score >= thresholds[:, None]
    
    def get_unique_thresholds(self, score):
        return torch.sort(torch.unique(score), descending=True)
    
    def get_anomalies_coordinates(self, label: torch.Tensor, include_edges: bool = True):
        """
        Return the starting and ending points of all anomalies in label.
        
        If include_edges is True, include the first and last index if anomalies exist 
        at the very beginning or end.

        Args:
            TODO: Write the arguments description when done
        """
        diff = torch.diff(label)
        start_points = torch.where(diff == 1)[0] + 1
        end_points = torch.where(diff == -1)[0]

        if include_edges:
            if label[-1] == 1:
                end_points = torch.cat((end_points, torch.tensor([len(label) - 1], device=label.device)))
            if label[0] == 1:
                start_points = torch.cat((torch.tensor([0], device=label.device), start_points))
            if start_points.shape != end_points.shape:
                raise ValueError(f"The number of start and end points of anomalies does not match: {start_points} != {end_points}")

        return start_points, end_points