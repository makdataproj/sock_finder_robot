class simplePID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.targetpoint = 0
        self.intergral = 0
        self.derivative = 0
        self.prevError = 0
    def compute(self, target, current):
        error = target - current
        self.intergral += error
        self.derivative = error - self.prevError
        self.targetpoint = self.kp * error + self.ki * self.intergral + self.kd * self.derivative
        self.prevError = error
        return self.targetpoint
    def reset(self):
        self.targetpoint = 0
        self.intergral = 0
        self.derivative = 0
        self.prevError = 0