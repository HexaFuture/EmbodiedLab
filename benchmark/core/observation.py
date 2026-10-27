"""Public camera serialization, independent of every participant's vision stack.

Only explicit camera fields cross this boundary. No scene graph, object labels,
estimated poses, masks, physical audit or parameter truth is returned.
"""
import base64
import hashlib
import math


def matrix(value, rows, columns):
    if not isinstance(value, (list, tuple)) or len(value) != rows:
        raise ValueError('invalid camera calibration shape')
    result = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != columns:
            raise ValueError('invalid camera calibration shape')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in row):
            raise ValueError('camera calibration must be finite')
        result.append(list(row))
    return result


def public_frame(frame, *, episode_id, state_id, index):
    name = frame['camera']
    size = frame['resolution']
    rgb = frame['rgb_png']
    depth = frame['depth_u16le_mm']
    if not isinstance(name, str) or not 1 <= len(name) <= 128:
        raise ValueError('invalid camera name')
    if not isinstance(size, (list, tuple)) or len(size) != 2 or any(type(v) is not int or not 1 <= v <= 4096 for v in size):
        raise ValueError('invalid image resolution')
    if not isinstance(rgb, bytes) or not rgb.startswith(b'\x89PNG\r\n\x1a\n'):
        raise ValueError('RGB camera must produce a PNG')
    if not isinstance(depth, bytes) or len(depth) != size[0]*size[1]*2:
        raise ValueError('depth camera must match RGB resolution')
    digest = hashlib.sha256(rgb).hexdigest()
    return {
        'camera': name, 'resolution': list(size),
        'rgb_ref': f'{state_id}:camera:{index}:{digest[:16]}',
        'episode_id': episode_id, 'state_id': state_id,
        'rgb': {'mime_type': 'image/png', 'encoding': 'base64',
                'data': base64.b64encode(rgb).decode('ascii'), 'sha256': digest},
        'depth': {'encoding': 'base64', 'dtype': 'uint16-le', 'unit_m': .001,
                  'invalid_value': 0, 'quantity': 'distance_to_image_plane',
                  'data': base64.b64encode(depth).decode('ascii'),
                  'sha256': hashlib.sha256(depth).hexdigest()},
        'intrinsics': matrix(frame['intrinsics'], 3, 3),
        'world_from_camera_ros': matrix(frame['world_from_camera_ros'], 4, 4),
    }
