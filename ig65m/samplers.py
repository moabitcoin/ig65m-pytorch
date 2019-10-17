import random


# Reservoir sampler for randomly selecting k items from a
# stream of n items where n is large or potentially unknown.

class StreamSampler:
    def __init__(self, n):
        assert n > 0

        self.n = n
        self.added = 0
        self.reservoir = []

    def add(self, v):
        size = len(self.reservoir)

        if size < self.n:
            self.reservoir.append(v)
        else:
            assert size == self.n
            assert size <= self.added

            p = self.n / self.added

            if random.random() < p:
                i = random.randint(0, size - 1)
                self.reservoir[i] = v

        self.added += 1

    def __len__(self):
        return len(self.reservoir)

    def __getitem__(self, i):
        return self.reservoir[i]

    def __iter__(self):
        return iter(self.reservoir)
