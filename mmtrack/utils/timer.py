# Copyright (c) OpenMMLab. All rights reserved.
import time


class Timer:
    """A timer class."""

    def __init__(self, start=True, print_tmpl=None):
        self.start_time = None
        self.last_time = None
        self.print_tmpl = print_tmpl
        if start:
            self.start()

    def start(self):
        if not self.is_running:
            self.start_time = time.time()
            self.last_time = self.start_time

    def since_start(self):
        """Total time since the timer started (in seconds)"""
        if not self.is_running:
            return self.last_time - self.start_time
        else:
            return time.time() - self.start_time

    def since_last_check(self):
        """Time since the last checking (in seconds)"""
        if not self.is_running:
            dur = self.last_time - self.start_time
            self.start()
            return dur
        else:
            dur = time.time() - self.last_time
            self.last_time = time.time()
            return dur

    @property
    def is_running(self):
        return self.start_time is not None and self.last_time is None

    def __str__(self):
        return f'{self.since_start():.3f}s'

    def __repr__(self):
        return self.__str__()
