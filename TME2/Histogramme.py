import numpy as np

class Histogramme:
    
    def __init__(self, data):
        self.data = data;
    
    def predict(self,grid):
        result = np.zeros(len(grid))
        for example in self.data:
            for i in range(len(grid)):
                roof = grid[i]
                if example[1] < roof[0] and example[0] < roof[1]:
                    result[i] += 1
                    break
        result /= len(self.data)
        return result