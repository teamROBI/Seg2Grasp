"""Azure Kinect DK capture for Seg2Grasp (RGB, metric depth, organized cloud).

Thin wrapper over ``pyk4a`` matching the paper's setup (Azure Kinect DK at an
elevated angle over the bin). Returns exactly what the pipeline needs:
    rgb   [H, W, 3]  BGR uint8
    depth [H, W]     metric depth (mm)
    pc    [H, W, 3]  organized point cloud (mm)

A region-of-interest crop (the bin area) can be applied so downstream modules
only see the bin. ``pyk4a`` is an optional dependency (install with the ``robot``
extra); this module imports it lazily so the rest of the package works without it.
"""
import numpy as np


class KinectCamera:
    def __init__(self, roi=None, color_resolution="720P", fps=5):
        """
        Args:
            roi (tuple, optional): (y_start, x_start, height, width) bin crop.
            color_resolution (str): Kinect color resolution key (e.g. "720P").
            fps (int): capture frame rate (5, 15, 30).
        """
        import pyk4a
        from pyk4a import Config, PyK4A
        self._pyk4a = pyk4a
        self.roi = roi
        self.k4a = PyK4A(Config(
            color_resolution=getattr(pyk4a.ColorResolution, f"RES_{color_resolution}"),
            depth_mode=pyk4a.DepthMode.WFOV_UNBINNED,
            synchronized_images_only=True,
            camera_fps=getattr(pyk4a.FPS, f"FPS_{fps}"),
        ))
        self.k4a.start()

    def _crop(self, img):
        if self.roi is None:
            return img
        y, x, h, w = self.roi
        return img[y:y + h, x:x + w]

    def capture(self):
        """Grab one synchronized frame.

        Returns:
            (rgb, depth, pc): BGR image, metric depth (mm), organized cloud (mm),
            all cropped to the ROI if one was set.
        """
        cap = self.k4a.get_capture()
        rgb = self._crop(cap.color[:, :, :3])
        depth = self._crop(cap.transformed_depth).astype(np.float32)
        pc = self._crop(cap.transformed_depth_point_cloud).astype(np.float32)
        return rgb, depth, pc

    def close(self):
        self.k4a.stop()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
