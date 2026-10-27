"""Read-only close-ups of an explicitly chosen, already observed SAM mask.

No segmentation, target selection, depth estimation or robot action happens
here. Pixel coordinates returned to motion tools always refer to the original
camera image, never to a magnified display.
"""
import cv2
import numpy as np


def mask_closeup(rgb, mask, point_xy, padding_px=32, scale=4):
    for name, value, low, high in (("padding_px", padding_px, 8, 128),
                                   ("scale", scale, 1, 8)):
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in [{low},{high}]")
    rgb, mask = np.asarray(rgb), np.asarray(mask, dtype=bool)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or mask.shape != rgb.shape[:2]:
        raise ValueError("mask and RGB dimensions differ")
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("cannot inspect an empty mask")
    h, w = mask.shape
    bounds = [int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1]
    x0, y0 = max(0, bounds[0]-padding_px), max(0, bounds[1]-padding_px)
    x1, y1 = min(w, bounds[2]+padding_px), min(h, bounds[3]+padding_px)
    raw = rgb[y0:y1, x0:x1].copy()
    overlay = raw.copy()
    local_mask = mask[y0:y1, x0:x1]
    overlay[local_mask] = (0.5*overlay[local_mask] + [0, 127.5, 40]).astype(np.uint8)
    cv2.drawMarker(overlay, (int(point_xy[0])-x0, int(point_xy[1])-y0),
                   (255, 180, 0), cv2.MARKER_CROSS, 12, 2)
    # Bound each display while retaining context around large candidates.
    factor = min(scale, 1024/max(raw.shape[:2]))
    out_w, out_h = max(1, round(raw.shape[1]*factor)), max(1, round(raw.shape[0]*factor))
    images = [cv2.resize(im, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
              for im in (raw, overlay)]
    metadata = {
        "mask_bounds_xyxy_exclusive": bounds,
        "crop_bounds_xyxy_exclusive": [x0, y0, x1, y1],
        "source_size_wh": [w, h], "display_size_wh": [out_w, out_h],
        "display_scale_xy": [out_w/(x1-x0), out_h/(y1-y0)],
        "image_order": ["unmarked_rgb_closeup", "selected_mask_and_original_click_closeup"],
        "coordinate_note": "Motion and SAM inputs use original RGB pixels. Convert a display pixel centre with source_xy = crop_origin_xy + (display_xy + 0.5) / display_scale_xy - 0.5. Magnification adds no new observation detail.",
    }
    return metadata, images
