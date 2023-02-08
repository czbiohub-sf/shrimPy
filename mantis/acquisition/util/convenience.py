import numpy as np

def get_z_range(start:float, stop:float, step:float):
    return np.linspace(start, stop, np.ceil((stop+step-start)/step).astype(int), endpoint=True)
