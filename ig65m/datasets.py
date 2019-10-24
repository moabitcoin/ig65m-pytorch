import math

from torch.utils.data import IterableDataset, get_worker_info

import cv2


class FrameRange:
    def __init__(self, video, first, last):
        assert first <= last

        for i in range(first):
            ret, _ = video.read()

            if not ret:
                raise RuntimeError("seeking to frame at index {} failed".format(i))

        self.video = video
        self.it = first
        self.last = last

    def __next__(self):
        if self.it >= self.last or not self.video.isOpened():
            raise StopIteration

        ok, frame = self.video.read()

        if not ok:
            raise RuntimeError("decoding frame at index {} failed".format(self.it))

        self.it += 1

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class BatchedRange:
    def __init__(self, rng, n):
        self.rng = rng
        self.n = n

    def __next__(self):
        ret = []

        for i in range(self.n):
            ret.append(next(self.rng))

        return ret


class TransformedRange:
    def __init__(self, rng, fn):
        self.rng = rng
        self.fn = fn

    def __next__(self):
        return self.fn(next(self.rng))


class VideoDataset(IterableDataset):
    def __init__(self, path, clip, transform=None):
        super().__init__()

        if not path.is_file():
            raise RuntimeError("video at {} does not exist".format(path))

        self.path = path
        self.clip = clip
        self.transform = transform

        video = cv2.VideoCapture(str(path))
        frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.release()

        self.first = 0
        self.last = frames

    def __len__(self):
        return self.last // self.clip

    def __iter__(self):
        info = get_worker_info()

        video = cv2.VideoCapture(str(self.path))

        if info is None:
            rng = FrameRange(video, self.first, self.last)
        else:
            per = int(math.ceil((self.last - self.first) / float(info.num_workers)))
            wid = info.id

            first = self.first + wid * per
            last = min(first + per, self.last)

            rng = FrameRange(video, first, last)

        if self.transform is not None:
            fn = self.transform
        else:
            fn = lambda v: v  # noqa: E731

        return TransformedRange(BatchedRange(rng, self.clip), fn)


class WebcamDataset(IterableDataset):
    def __init__(self, clip, transform=None):
        super().__init__()

        self.clip = clip
        self.transform = transform
        self.video = cv2.VideoCapture(0)

    def __iter__(self):
        info = get_worker_info()

        if info is not None:
            raise RuntimeError("multiple workers not supported in WebcamDataset")

        # treat webcam as fixed frame range for now: 10 minutes
        rng = FrameRange(self.video, 0, 30 * 60 * 10)

        if self.transform is not None:
            fn = self.transform
        else:
            fn = lambda v: v  # noqa: E731

        return TransformedRange(BatchedRange(rng, self.clip), fn)
