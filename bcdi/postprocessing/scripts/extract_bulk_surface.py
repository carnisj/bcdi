# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import tkinter as tk
from tkinter import filedialog
from scipy.signal import convolve
import os
import logging

scan = 2191  # spec scan number
datadir = "C:/Users/carnis/Work Folders/Documents/data/CH4760_Pt/S"+str(scan)+"/pynxraw/"
support_threshold = 0.25  # threshold for support determination
save = 1  # 1 to save
debug = 1  # 1 to see all plots
############################################################
logger = logging.getLogger()
# define a colormap
cdict = {'red':  ((0.0, 1.0, 1.0),
                  (0.11, 0.0, 0.0),
                  (0.36, 0.0, 0.0),
                  (0.62, 1.0, 1.0),
                  (0.87, 1.0, 1.0),
                  (1.0, 0.0, 0.0)),
         'green': ((0.0, 1.0, 1.0),
                   (0.11, 0.0, 0.0),
                   (0.36, 1.0, 1.0),
                   (0.62, 1.0, 1.0),
                   (0.87, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
         'blue': ((0.0, 1.0, 1.0),
                  (0.11, 1.0, 1.0),
                  (0.36, 1.0, 1.0),
                  (0.62, 0.0, 0.0),
                  (0.87, 0.0, 0.0),
                  (1.0, 0.0, 0.0))}
my_cmap = LinearSegmentedColormap('my_colormap', cdict, 256)


def calc_coordination(myamp, mythreshold, debugging=0):
    nbz, nby, nbx = myamp.shape
    mysupport = np.zeros((nbz, nby, nbx))
    mysupport[myamp > mythreshold * abs(myamp).max()] = 1

    mykernel = np.ones((3, 3, 3))
    mycoord = np.rint(convolve(mysupport, mykernel, mode='same'))
    mycoord = mycoord.astype(int)

    if debugging == 1:
        plt.figure(figsize=(18, 15))
        plt.subplot(2, 2, 1)
        plt.imshow(mycoord[:, :, nbx // 2])
        plt.colorbar()
        plt.axis('scaled')
        plt.title("Coordination matrix in middle slice in YZ")
        plt.subplot(2, 2, 2)
        plt.imshow(mycoord[:, nby // 2, :])
        plt.colorbar()
        plt.title("Coordination matrix in middle slice in XZ")
        plt.axis('scaled')
        plt.subplot(2, 2, 3)
        plt.imshow(mycoord[nbz // 2, :, :])
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.title("Coordination matrix in middle slice in XY")
        plt.axis('scaled')
        plt.pause(0.1)
    return mycoord


###################################################################
plt.ion()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir=datadir, filetypes=[("NPZ", "*.npz")])
npzfile = np.load(file_path)
amp = npzfile['amp']
strain = npzfile['strain']
support = np.zeros(amp.shape)
support[amp > support_threshold*amp.max()] = 1
nz, ny, nx = amp.shape
print("Initial data size: (", nz, ',', ny, ',', nx, ')')
coordination_matrix = calc_coordination(amp, support_threshold, debugging=0)
surface = np.copy(support)
surface[coordination_matrix > 22] = 0  # remove the bulk 22
# surface[np.nonzero(surface)] = 1
bulk = support - surface
bulk[np.nonzero(bulk)] = 1
if debug == 1:
    plt.figure(figsize=(18, 15))
    plt.subplot(3, 3, 1)
    plt.imshow(coordination_matrix[:, :, nx // 2], cmap=my_cmap)
    plt.colorbar()
    plt.axis('scaled')
    plt.title("Coordination matrix in middle slice in YZ")
    plt.subplot(3, 3, 2)
    plt.imshow(coordination_matrix[:, ny // 2, :], cmap=my_cmap)
    plt.colorbar()
    plt.title("Coordination matrix in middle slice in XZ")
    plt.axis('scaled')
    plt.text(100, -75, "threshold="+str(support_threshold))
    plt.subplot(3, 3, 3)
    plt.imshow(coordination_matrix[nz // 2, :, :], cmap=my_cmap)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title("Coordination matrix in middle slice in XY")
    plt.axis('scaled')

    plt.subplot(3, 3, 4)
    plt.imshow(surface[:, :, nx // 2])
    plt.colorbar()
    plt.axis('scaled')
    plt.title("Surface matrix in middle slice in YZ")
    plt.subplot(3, 3, 5)
    plt.imshow(surface[:, ny // 2, :])
    plt.colorbar()
    plt.title("Surface matrix in middle slice in XZ")
    plt.axis('scaled')
    plt.subplot(3, 3, 6)
    plt.imshow(surface[nz // 2, :, :])
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title("Surface matrix in middle slice in XY")
    plt.axis('scaled')

    plt.subplot(3, 3, 7)
    plt.imshow(bulk[:, :, nx // 2])
    plt.colorbar()
    plt.axis('scaled')
    plt.title("Bulk matrix in middle slice in YZ")
    plt.subplot(3, 3, 8)
    plt.imshow(bulk[:, ny // 2, :])
    plt.colorbar()
    plt.title("Bulk matrix in middle slice in XZ")
    plt.axis('scaled')
    plt.subplot(3, 3, 9)
    plt.imshow(bulk[nz // 2, :, :])
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title("Bulk matrix in middle slice in XY")
    plt.axis('scaled')
    plt.pause(0.1)

if save == 1:
    file_surface = open(os.path.join(datadir, "S" + str(scan) +
                                     "_threshold" + str(support_threshold) + "_surface.dat"), "w")
    file_bulk = open(os.path.join(datadir, "S" + str(scan) +
                                  "_threshold" + str(support_threshold) + "_bulk.dat"), "w")
    file_total = open(os.path.join(datadir, "S" + str(scan) +
                                   "_threshold" + str(support_threshold) + "_bulk+surface.dat"), "w")
    # write surface points position / strain to file
    surface_indices = np.nonzero(surface)
    nb_surface = len(surface_indices[0])
    ind_z = surface_indices[0]
    ind_y = surface_indices[1]
    ind_x = surface_indices[2]
    for point in range(nb_surface):
        file_surface.write(
            '{0: <10}'.format(str('{:.7f}'.format(strain[ind_z[point], ind_y[point], ind_x[point]]))) + '\n')

    # write bulk points position / strain to file
    bulk_indices = np.nonzero(bulk)
    nb_bulk = len(bulk_indices[0])
    ind_z = bulk_indices[0]
    ind_y = bulk_indices[1]
    ind_x = bulk_indices[2]
    for point in range(nb_bulk):
        file_bulk.write(
            '{0: <10}'.format(str('{:.7f}'.format(strain[ind_z[point], ind_y[point], ind_x[point]]))) + '\n')
    file_surface.close()
    file_bulk.close()

    # write all points position / strain to file
    total_indices = np.nonzero(support)
    nb_total = len(total_indices[0])
    ind_z = total_indices[0]
    ind_y = total_indices[1]
    ind_x = total_indices[2]
    for point in range(nb_total):
        file_total.write(
            '{0: <10}'.format(str('{:.7f}'.format(strain[ind_z[point], ind_y[point], ind_x[point]]))) + '\n')

    file_surface.close()
    file_bulk.close()
    file_total.close()

# plot histogram for surface
nb_surface = len(np.nonzero(surface)[0])
print("Surface points = ", str(nb_surface))
hist, bin_edges = np.histogram(strain[np.nonzero(surface)], bins=50)
hist = hist / nb_surface  # normalize the histogram to the number of points
hist[hist == 0] = np.nan
x_axis = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2
fig = plt.figure()
# plt.bar(x_axis, np.log10(hist), width=(bin_edges[1] - bin_edges[0]))
plt.plot(x_axis, np.log10(hist))
plt.xlim(-0.007, 0.007)
# plt.xlim(min(x_axis), max(x_axis))
plt.ylim(-6, 0)
plt.title('Histogram of the strain for surface points, ' + str(nb_surface)
          + ' points\n' + "threshold="+str(support_threshold))
plt.pause(0.1)

# plot histogram of bulk points
nb_bulk = len(np.nonzero(bulk)[0])
print("Bulk points = ", str(nb_bulk))
hist, bin_edges = np.histogram(strain[np.nonzero(bulk)], bins=50)
hist = hist / nb_bulk  # normalize the histogram to the number of points
hist[hist == 0] = np.nan
x_axis = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2
fig = plt.figure()
# plt.bar(x_axis, np.log10(hist), width=(bin_edges[1] - bin_edges[0]))
plt.plot(x_axis, np.log10(hist))
plt.xlim(-0.007, 0.007)
# plt.xlim(min(x_axis), max(x_axis))
plt.ylim(-6, 0)
plt.title('Histogram of the strain for bulk points, ' + str(nb_bulk)
          + ' points\n' + "threshold="+str(support_threshold))
plt.pause(0.1)

nb_total = len(np.nonzero(support)[0])
print("Total points = ", str(nb_total), ", surface+bulk = ", str(nb_surface+nb_bulk))
plt.ioff()
plt.show()