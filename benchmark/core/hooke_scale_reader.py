"""Read the painted pointer against real ticks, using RGB pixels only."""
import cv2
import numpy as np


class VisualScaleReader:
    def __init__(self):
        self.calibration = None
        self.last_debug = {}

    def calibrate(self, image):
        rgb = np.asarray(image)[..., :3].astype(float)
        mask = ((rgb.max(2)-rgb.min(2) < 38) & (rgb.min(2) > 85)).astype('uint8')
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        candidates = [i for i in range(1, count) if stats[i, 3] > .25*image.shape[0]
                      and 10 < stats[i, 2] < .2*image.shape[1]
                      and stats[i, 3] > 3*stats[i, 2]]
        if not candidates:
            return False
        index = max(candidates, key=lambda i: stats[i, 4])
        x, y, w, h, _ = stats[index]
        ys = np.arange(y+3, y+int(.62*h))
        edges = np.array([np.where(labels[row] == index)[0][-1] for row in ys])
        # The ruler edge is interrupted by black ticks. Fit through its
        # visible enamel edge, excluding tick cutouts and lower occlusion.
        edge_fit = np.polyfit(ys, edges, 1)
        for _ in range(3):
            keep = edges >= np.polyval(edge_fit, ys)-.3
            if keep.sum() < 10:
                return False
            edge_fit = np.polyfit(ys[keep], edges[keep], 1)
        scores = []
        for row in ys:
            edge = int(round(np.polyval(edge_fit, row)))
            enamel = np.median(rgb[row, edge-18:edge-10].mean(1))
            scores.append(float(enamel - rgb[row, edge-8:edge-1].mean()))
        scores = np.asarray(scores)
        peaks = [i for i in range(1, len(scores)-1) if scores[i] > 25
                 and scores[i] >= scores[i-1] and scores[i] > scores[i+1]]
        if len(peaks) < 12 or not 3 <= ys[peaks[0]]-y <= .08*h:
            return False
        ticks = []
        for peak in peaks:
            # Subpixel peak position avoids rounding each 2 cm interval to
            # a whole pixel (the ruler occupies only about 200 pixels).
            weights = np.maximum(0, scores[peak-1:peak+2])
            center = float(np.dot(ys[peak-1:peak+2], weights)/weights.sum())
            if len(ticks) > 1:
                spacing = ticks[-1]-ticks[-2]
                if not .7*spacing < center-ticks[-1] < 1.3*spacing:
                    break
            ticks.append(center)
        if len(ticks) < 12:
            return False
        # A tilted ruler has a projective, not linear, pixel-to-length map.
        values = 60 - 2*np.arange(len(ticks), dtype=float)
        ticks = np.asarray(ticks)
        matrix = np.column_stack([values, np.ones_like(values), -values*ticks])
        a, b, c = np.linalg.lstsq(matrix, ticks, rcond=None)[0]
        residual = (a*values+b)/(c*values+1)-ticks
        if np.max(np.abs(residual)) > .75:
            return False
        self.calibration = (a, b, c, edge_fit, (x, y, w, h))
        self.last_debug = {'tick_rows_px': ticks.tolist(), 'tick_values_cm': values.tolist(),
                           'calibration_max_error_px': float(np.max(np.abs(residual)))}
        return True

    def read(self, image):
        if self.calibration is None and not self.calibrate(image):
            return None, 0.0, {'reason': 'ruler_calibration_failed'}
        a, b, c, edge_fit, (x, y, w, h) = self.calibration
        rgb = np.asarray(image)[..., :3].astype(float)
        r, g, blue = rgb.transpose(2, 0, 1)
        # Measure whole red components before applying the ruler region.
        # Cropping the color mask first can turn the edge of a large red
        # weight into a tiny fragment that looks like a second pointer.
        red = ((r > 65) & (r > 1.65*g) & (r > 1.6*blue)).astype('uint8')
        count, _, stats, centroids = cv2.connectedComponentsWithStats(red, 8)
        # At this camera resolution the painted 26 mm tip spans about 9–11
        # pixels. Tiny red glints reflected in the brass bridge are smaller.
        candidates = [i for i in range(1, count) if 6 <= stats[i, 2] <= 18
                      and stats[i, 3] <= 5 and stats[i, 4] >= 8
                      and y+4 <= centroids[i, 1] <= y+h-4
                      and np.polyval(edge_fit, centroids[i, 1])-12 <= centroids[i, 0]
                      <= np.polyval(edge_fit, centroids[i, 1])]
        if len(candidates) != 1:
            return None, 0.0, {'reason': 'pointer_missing_or_ambiguous', 'candidates': len(candidates)}
        pointer = float(centroids[candidates[0], 1])
        reading = (b-pointer)/(c*pointer-a)
        if not 0 <= reading <= 60:
            return None, 0.0, {'reason': 'pointer_outside_scale'}
        return float(reading), .9, {**self.last_debug, 'pointer_y_px': pointer}
