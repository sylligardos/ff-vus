"""
@who: Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: PhD Candidate, 2nd year (2025)
@what: FF-VUS
"""

from utils.utils import time_it

import numpy as np
from skimage.util.shape import view_as_windows as viewW
import math
import torch
import matplotlib.pyplot as plt
import seaborn as sns


class VUS:
    def __init__(self, window_size=100, step=1, zita=(1/math.sqrt(2)), device='cpu', interpolation='stepwise', dynamic='full', existence='optimized'):
        """
        Initialize the VUS metric.
        TODO: Instead of saving the variables you can save the appropriate functions in the variables instead of

        Args:
            TODO: Write the arguments description when done
        """
        self.window_size = window_size
        self.step = step
        self.zita = zita
        self.interpolation = interpolation
        self.dynamic = dynamic
        
        if existence is None:
            self.existence = self._no_existence
        elif existence == 'trivial':
            self.existence = self._trivial_existence
        elif existence == 'optimized':
            self.existence = self._optimized_existence
        else:
            raise ValueError(f"Unknown argument for existence: {existence}")

        if device.lower() == 'cpu':
            self.device = torch.device('cpu')
        elif device.lower() in ['gpu', 'cuda']:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                raise ValueError(f"Device selected is not available: {device}")
        else:
            raise ValueError(f"Unknown argument for device: {device}")

        self.pos_slopes, self.l_values = self._compute_slopes()
        self.neg_slopes = self.pos_slopes.flip(1)
        self.n_windows = self.l_values.shape[0]


    def _strided_indexing_roll(self, a, r):
        # Concatenate with sliced to cover all rolls
        a_ext = np.concatenate((a, a[:, : - 1]), axis=1)

        # Get sliding windows; use advanced-indexing to select appropriate ones
        n = a.shape[1]
        return viewW(a_ext, (1, n))[np.arange(len(r)), (n - r) % n, 0]


    def _compute_slopes(self, return_l_values=True):
        l_values = np.arange(self.window_size + 1, step=self.step)
        steps = ((1 - self.zita) / l_values[1:])
        stops = 1 + steps[::-1] * l_values[:-1]
        
        slopes = np.linspace(self.zita, stops, self.window_size + 1).T[::-1]
        slopes[slopes > 1] = 0
        
        rolls = self.window_size - np.arange(1, self.window_size + 1)
        shifted_slopes = self._strided_indexing_roll(slopes, rolls)

        if return_l_values:
            return torch.tensor(shifted_slopes, device=self.device), torch.tensor(l_values, device=self.device) 
        else:
            return torch.tensor(shifted_slopes, device=self.device)
    
    @time_it
    def compute(self, label, score):
        """
        TODO: Decide between compute single or multiple scores
        
        """
        sloped_label = self.add_slopes(label)
        thresholds, indices = torch.sort(torch.unique(score), descending=True)
        sm = score >= thresholds[:, None]
        sm_inv = ~sm
        
        existence = self.compute_existence(sloped_label, score, sm, thresholds)
        print(existence)

        return thresholds


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

    def add_slopes(self, label: torch.Tensor):
        """
        Apply slopes to the label.
        
        Args:
            TODO: Write args description
        """
        # TODO: Normal or approximate version
        start_points, end_points = self.get_anomalies_coordinates(label, include_edges=False)
        
        result = label.repeat(self.n_windows, 1)
        for curr_point in start_points:
            slope_start = max(curr_point - self.window_size, 0)
            slope_end = curr_point + 1  # to include the point in the transformation
            adjusted_l = slope_end - slope_start
            result[1:, slope_start:slope_end] = torch.max(result[1:, slope_start:slope_end], self.pos_slopes[:, :adjusted_l])
        for curr_point in end_points:
            slope_start = curr_point
            slope_end = min(curr_point + self.window_size, len(label) - 1) + 1  # to include the point in the transformation
            adjusted_l = slope_end - slope_start
            result[1:, slope_start:slope_end] = torch.max(result[1:, slope_start:slope_end], self.neg_slopes[:, -adjusted_l:])

        return result

    def compute_existence(self, sloped_label, score, sm, thresholds):
        # TODO: Normal or approximate version
        return self.existence(sloped_label, score, sm, thresholds)

    def _no_existence(self, sloped_label, score, sm, thresholds):
        """
        Returns a matrix of ones. This has no effect on the result
        """
        existence = torch.ones((self.n_windows, len(thresholds)), device=self.device)
        return existence
    
    def _trivial_existence(self, sloped_label, score, sm, thresholds):
        l, T = sloped_label.shape
        t = thresholds.shape[0]
        existence = torch.zeros((l, t), dtype=torch.float32, device=self.device)
        
        start_points, end_points = self.get_anomalies_coordinates(sloped_label[0])
        
        n_anomalies = torch.full((l,), start_points.shape[0], dtype=torch.int32, device=self.device)
        l_values = torch.arange(0, l, step=self.step, device=self.device)[:, None]
        
        all_start_points = start_points.repeat(l, 1) - l_values
        all_end_points = end_points.repeat(l, 1) + l_values

        all_start_points = all_start_points.clamp(0, T - 1)
        all_end_points = all_end_points.clamp(0, T - 1)

        for k, (start, end) in enumerate(zip(start_points[1:], end_points[:-1])):
            distance = start - end
            overlap = distance - (2 * (l - 1))
            if overlap < 0:
                index = math.ceil(overlap / 2) - 1
                n_anomalies[index:] -= 1
                all_start_points[index:, (k + 1)] = -1
                all_end_points[index:, k] = -1

        # Trivial existence computation
        for i in range(l):
            for j in range(t):
                valid_start_points = all_start_points[i, all_start_points[i] >= 0]
                valid_end_points = all_end_points[i, all_end_points[i] >= 0]
                for k in range(n_anomalies[i]):
                    start = valid_start_points[k]
                    end = valid_end_points[k]
                    if torch.any(sm[j, start: end + 1]):
                        existence[i, j] += 1

        return existence / n_anomalies[:, None]

    def _optimized_existence(self, sloped_label, score, sm, thresholds):
        l, T = sloped_label.shape
        t = thresholds.shape[0]
        existence = torch.zeros((l, t), dtype=torch.float32, device=self.device)
        thresholds_dict = {value: index for index, value in enumerate(thresholds)}

        start_points, end_points = self.get_anomalies_coordinates(sloped_label[0])

        n_anomalies = torch.full((l,), start_points.shape[0], dtype=torch.int32, device=self.device)
        l_values = torch.arange(0, l, step=self.step, device=self.device)[:, None]
        
        all_start_points = start_points.repeat(l, 1) - l_values
        all_end_points = end_points.repeat(l, 1) + l_values
        
        all_start_points = all_start_points.clamp(0, T - 1)
        all_end_points = all_end_points.clamp(0, T - 1)
        
        anomalies_overlaps = torch.zeros(start_points.shape, dtype=torch.int32, device=self.device)
        for k, (start, end) in enumerate(zip(start_points[1:], end_points[:-1])):
            distance = start - end
            overlap = 2 * (l - 1) - distance
            if overlap > 0:
                index = (l - 1) - int(overlap / 2)
                n_anomalies[index:] -= 1
                anomalies_overlaps[k] = index

        prev_existence = torch.zeros((l, t), dtype=torch.int32, device=self.device)
        for k in range(start_points.shape[0]):
            i = l - 1
            overlap_index = anomalies_overlaps[k - 1]       # This works because the last one never has an overlap ;)
            curr_existence = torch.zeros((l, t), dtype=torch.int32, device=self.device)
            tmp_existence = torch.zeros((l, t), dtype=torch.int32, device=self.device)
            
            while i >= 0:
                start = all_start_points[i, k]
                og_start = start_points[k]
                og_end = end_points[k]
                end = all_end_points[i, k]
                
                argmax_t = [score[start:og_start + 1].flip(0).argmax(),
                            score[og_start:og_end + 1].argmax(),
                            score[og_end:end + 1].argmax()]
                max_t = [score[start + ((og_start - start) - argmax_t[0])],
                        score[og_start + argmax_t[1]],
                        score[og_end + argmax_t[2]]]
                
                if max_t[1] >= max_t[0] and max_t[1] >= max_t[2]:  # Case middle
                    threshold_index = thresholds_dict[max_t[1].item()]
                    curr_existence[:i+1, threshold_index:] += 1
                    break
                elif max_t[0] > max_t[2]:  # Case left
                    threshold_index = thresholds_dict[max_t[0].item()]
                    curr_existence[argmax_t[0]:i+1, threshold_index:] += 1
                    i = argmax_t[0] - 1
                elif max_t[2] >= max_t[0]:  # Case right
                    threshold_index = thresholds_dict[max_t[2].item()]
                    curr_existence[argmax_t[2]:i+1, threshold_index:] += 1
                    i = argmax_t[2] - 1
            
            if overlap_index > 0:
                tmp_existence[overlap_index:] = torch.logical_or(curr_existence[overlap_index:], prev_existence[overlap_index:])
                curr_existence[overlap_index:] = torch.logical_and(curr_existence[overlap_index:], ~prev_existence[overlap_index:])
            
            existence += curr_existence
            prev_existence = torch.logical_or(tmp_existence, curr_existence)
        
        return existence / n_anomalies[:, None]


    def confusion_matrix():
        # TODO: Normal or approximate version
        # TODO: Trivial, dynamic_v1 (remove 0s), dynamic_v2 (remove 1s)
        pass

    def recall():
        # TODO: Normal or approximate version
        pass

    def precision():
        # TODO: Normal or approximate version
        pass

    def interpolate():
        # TODO: Decide between stepwise or linear
        pass

