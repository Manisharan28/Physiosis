"""
Physiosis — Video input handler for webcam and video files.
"""
import cv2
import os


class VideoHandler:
    """
    Unified video input wrapper.
    Accepts webcam index (int) or file path (str).
    """

    def __init__(self, source=0):
        """
        Args:
            source: 0 for default webcam, or a file path string for video.
        """
        self.source = source
        self.is_webcam = isinstance(source, int)

        if not self.is_webcam and not os.path.isfile(source):
            raise FileNotFoundError(f"Video file not found: {source}")

        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open video source: {source}. "
                "Check your webcam connection or file path."
            )

        self._fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def resolution(self) -> tuple:
        return (self._width, self._height)

    @property
    def total_frames(self) -> int:
        return self._total_frames if not self.is_webcam else -1

    def read_frame(self):
        """
        Read the next frame.
        
        Returns:
            (success: bool, frame: np.ndarray or None)
        """
        ret, frame = self.cap.read()
        if ret and frame is not None:
            # Flip webcam horizontally for natural mirror view
            if self.is_webcam:
                frame = cv2.flip(frame, 1)
        return ret, frame

    def release(self):
        """Release the video capture resource."""
        if self.cap is not None:
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()

    def __str__(self):
        src = "Webcam" if self.is_webcam else self.source
        return f"VideoHandler({src}, {self._width}x{self._height} @ {self._fps:.1f}fps)"
