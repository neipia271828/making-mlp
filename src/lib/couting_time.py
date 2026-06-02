import time

class Timer():
    def start(self):
        self.start = time.perf_counter()
    
    def end(self):
        self.end = time.perf_counter()

    def get(self):
        return self.end - self.start