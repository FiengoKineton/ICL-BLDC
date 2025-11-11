import time
class Timer:
    def __enter__(self): self.t0=time.time(); return self
    def __exit__(self, *exc): self.dt=time.time()-self.t0
