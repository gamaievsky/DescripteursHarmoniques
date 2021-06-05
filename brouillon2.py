
import numpy as np
from numpy import linalg as LA


a = np.array([1.5,2,5,4,6])
b = np.array([3,2,1,7,5])
print(a*b)

# norm = mpl.colors.LogNorm(vmin=0.0001,vmax=100.)
# print(norm(0.))


# Dictionnaire des descripteurs harmoniques
# class MplColorHelper:
#
#   def __init__(self, cmap_name, start_val, stop_val):
#     self.cmap_name = cmap_name
#     self.cmap = plt.get_cmap(cmap_name)
#     self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
#     self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
#
#   def get_rgb(self, val):
#     return self.scalarMap.to_rgba(val)
#
# COL = MplColorHelper('jet', 10, 100)
# y = 57
# c=COL.get_rgb(y)[:-1]
# print(c)
#
#
#
# # a = 0.00046
# # print('{:.1e}'.format(a))
