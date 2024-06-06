import numpy as np
import matplotlib.pyplot as plt

def hello_world():
    print("Hello, World!")

def sample_plot():
    x = np.linspace(0,1)
    y = x**2
    plt.plot(x,y)
