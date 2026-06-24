class EarlyStopper:

    def __init__(self, patience=1, min_delta=0.0):
        self.patience: int = patience
        self.min_delta: float = min_delta
        self.counter: int = 0
        self.min_vloss: float = float("inf")

    def early_stop(self, vloss: float) -> bool:
        if vloss < self.min_vloss:
            self.min_vloss = vloss
            self.counter = 0
        elif vloss > (self.min_vloss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
