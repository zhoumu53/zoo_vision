import cv2
import decord
import numpy as np


class VideoLoader:
    """
    Unified robust video loader:
      - Try Decord (fast, GPU capable)
      - Fall back to OpenCV
      - Supports __len__, __getitem__, and iteration
    """

    def __init__(self, video_path, ctx=None, verbose=True):
        self.video_path = video_path
        self.verbose = verbose

        # Try DECORD
        try:
            if ctx is None:
                ctx = decord.cpu(0)
            self.reader = decord.VideoReader(video_path, ctx=ctx)
            self.backend = "decord"
            self._len = len(self.reader)
            if verbose:
                print(f"[VideoLoader] Decord backend for: {video_path}")
            return
        except Exception:
            if verbose:
                print(f"[VideoLoader] Decord failed, trying OpenCV: {video_path}")

        # Fallback: OPENCV
        self.cap = cv2.VideoCapture(video_path)
        if self.cap.isOpened():
            self.backend = "opencv"
            total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total > 0:
                self._len = total
            else:
                # metadata missing -> unknown length, track manually
                self._len = None
            if verbose:
                print(f"[VideoLoader] OpenCV backend for: {video_path}")
        else:
            self.backend = None
            self._len = 0
            if verbose:
                print(f"[VideoLoader] ERROR: cannot open video: {video_path}")

    def __len__(self):
        if self._len is None:
            # unknown (OpenCV metadata missing)
            # we could scan frames, but that is expensive
            return 0
        return self._len

    def __getitem__(self, idx):
        """Random access."""
        if self.backend == "decord":
            frame = self.reader[idx].asnumpy()
            return frame

        if self.backend == "opencv":
            if self._len is None:
                # Cannot random access reliably
                raise ValueError("OpenCV loader without length cannot random index!")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = self.cap.read()
            if not ok:
                raise IndexError(f"Failed to read frame {idx}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame

        raise RuntimeError(f"Video backend not available for: {self.video_path}")

    def __iter__(self):
        """Iterator over all frames (sequential)."""
        if self.backend == "decord":
            for i in range(self._len):
                try:
                    yield self.reader[i].asnumpy()
                except Exception as exc:
                    if self.verbose:
                        print(f"[VideoLoader] Decord iteration failed at frame {i} with {exc}; switching to OpenCV fallback.")
                    # switch backend to opencv and continue from current index
                    self.cap = cv2.VideoCapture(self.video_path)
                    if not self.cap.isOpened():
                        return
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    while True:
                        ok, frame = self.cap.read()
                        if not ok:
                            break
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        yield frame
                    return
            return

        if self.backend == "opencv":
            # rewind
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield frame
            return

        # Unsupported case
        return iter(())

    def read_all(self):
        """Convenience: load all frames into a list."""
        return [f for f in self]

    def ok(self):
        """Check if loader is valid."""
        return self.backend is not None
