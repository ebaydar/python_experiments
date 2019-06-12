import matplotlib
from matplotlib import pyplot as plt 
import numpy as np 
import pandas as pd 
import scratch
import scratch.linear_algebra


v=np.array([2,1])            
w=np.array([0.5,0.8])
origin=np.array([0,0])
df=pd.DataFrame(np.array([origin,w,v]))
df.plot(kind="scatter",x=0, y=1)
plt.grid()                                                                                                                                                                                                        
plt.xlabel("age")                                                                                                                                                                                                 
plt.ylabel("height")  
plt.arrow(0,0,v[0],v[1], width=0.01, head_width=0.04)
plt.arrow(0,0,w[0],w[1], width=0.01, head_width=0.04)
plt.arrow(w[0],w[1], v[0]-w[0], v[1]-w[1], width=0.01, head_width=0.04)
plt.show()

from typing import List

Vector = List[float]
height_weight_age = [70,  # inches,
                     170, # pounds,
                     40 ] # years

grades = [95,   # exam1
          80,   # exam2
          75,   # exam3
          62 ]  # exam4

def add(v: Vector, w: Vector) -> Vector:
    """Adds corresponding elements"""
    assert len(v) == len(w), "vectors must be the same length"

    return [v_i + w_i for v_i, w_i in zip(v, w)]

assert add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]