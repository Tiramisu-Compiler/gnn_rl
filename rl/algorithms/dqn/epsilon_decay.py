import ray , math 

@ray.remote(num_cpus=0.01)
class EpsilonDecay:
    def __init__(self, end_value, start_value, threshold):
        self.end_value = end_value
        self.start_value = start_value
        self.threshold = threshold
        self.steps = 0

    def get_value(self):
        value =  self.end_value + (self.start_value - self.end_value) * math.exp(
            -1 * (self.steps / self.threshold)
        )
        self.steps +=1 
        return value

    def get_steps(self):
        return self.steps