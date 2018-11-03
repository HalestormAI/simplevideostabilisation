from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np

class Plotter(object):
    
    def plot_displacements(self, displacements):
        t = list(range(displacements.shape[0]))
        x = [pt[0] for pt in displacements]
        y = [pt[1] for pt in displacements]
        self.temporal_line_plot(t, x, y)
    
    def plot_position(self, displacements):
        positions = np.cumsum(displacements,0)
        t = list(range(displacements.shape[0]))
        x = [pt[0] for pt in positions]
        y = [pt[1] for pt in positions]
        self.temporal_line_plot(t, x, y)

    def temporal_line_plot(self, t, x, y):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.cla()
        self.ax.plot(t, x, y)
        self.ax.set_xlabel('time')
        self.ax.set_ylabel('x')
        self.ax.set_zlabel('y')
        plt.show()
        
