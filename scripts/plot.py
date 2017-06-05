#!/usr/bin/python
# -*- coding: utf8 -*


import matplotlib as mpl
import matplotlib.pylab as plt

from matplotlib import gridspec

import numpy as np

import cv2

import os

class ParticleFilterPlot(object):


    def __init__(self, fig=None):
        if fig is not None:
            self._fig = fig
        else:
            self._fig = None

        self._plotted = False
        self._wheel_ax = None
        self.frame = 0

    def colorwheel(self, gridspec):

        if self._wheel_ax is None:

            self._wheel_ax = self._fig.add_subplot(gridspec, projection='polar')


            xval = np.arange(0, 2*np.pi, 0.01)
            yval = np.ones_like(xval)

            colormap = plt.get_cmap('hsv')
            norm = mpl.colors.Normalize(0.0, 2*np.pi)

            self._wheel_ax.axis('off')

            self._wheel_ax.cla()
            self._wheel_ax.scatter(xval, yval, c=xval, s=1300, cmap=colormap, norm=norm, linewidths=0, marker='.')
            self._wheel_ax.set_yticks([])
            self._wheel_ax.set_yticklabels([])
            self._wheel_ax.set_xticks([])
            self._wheel_ax.set_xticklabels([])
            self._wheel_ax.spines['polar'].set_visible(False)


    def draw_color_wheel(self, img, angle):
        file = os.path.join(os.path.dirname(__file__), "color_wheel.png")
        wheel = cv2.imread(file, cv2.IMREAD_UNCHANGED)

        wheel = cv2.cvtColor(wheel, cv2.COLOR_BGRA2RGBA).astype(np.float32) / 255.0
        wheel = np.ascontiguousarray(np.swapaxes(wheel, 0, 1))

        im_rows, im_cols = img.shape[0], img.shape[1]
        rows,cols = wheel.shape[0], wheel.shape[1]

        im_scale = 1.0 / 6

        scale = min(im_scale * im_rows / rows, im_scale * im_cols / cols)

        M = cv2.getRotationMatrix2D((cols/2,rows/2), (angle + np.pi * 0.25) / (2 * np.pi) * 360, scale)

        wheel = cv2.warpAffine(wheel, M, (cols, rows), wheel, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT, -1.0)

        new_rows, new_cols = int(scale * rows), int(scale * cols)

        wheel = wheel[rows/2-new_rows/2:rows/2-new_rows/2+new_rows, cols/2-new_cols/2:cols/2-new_cols/2+new_cols]
        rows, cols = new_rows, new_cols


        pos_x = 0
        pos_y = 0

        alpha = wheel[:,:,3].reshape((wheel.shape[0], wheel.shape[1], 1))

        img[pos_y : pos_y+rows, pos_x : pos_x+cols, :] = wheel[:,:,0:3] * alpha + img[pos_y : pos_y+rows, pos_x : pos_x+cols, :] * (1.0 - alpha)



    def update_plot(self, mass_static, mass_dynamic, orientation, angle, plot_orientation='horizontal'):

        if self._fig is None:
            dpi = 90.0

            figsize = (1800 / dpi, 960 / dpi) if plot_orientation == 'horizontal' else (960 / dpi, 1200 / dpi)

            self._fig = plt.figure(dpi=dpi, figsize=figsize) # figure needs to be created here, apparently ros handlers don't run in the main thread

        orientation_hue = np.remainder((orientation), 2 * np.pi) / (2.0 * np.pi) * 255

        orientation_saturation = mass_dynamic

        min_point = 0.5

        orientation_value = mass_dynamic * (1.0 - min_point) + min_point

        # orientation_saturation = np.ones_like(orientation_hue)
        #
        # orientation_value = np.ones_like(orientation_hue)

        orientation_hsv = np.stack((orientation_hue, orientation_saturation, orientation_value), axis=2).astype(np.float32)

        orientation_rgb = cv2.cvtColor(orientation_hsv, cv2.COLOR_HSV2RGB)

        # orientation_rgb *= mass_dynamic.reshape(mass_dynamic.shape + (1,))

        # min_point = 0.5
        #
        # orientation_rgb = orientation_rgb * (1.0 - min_point) + min_point

        self.draw_color_wheel(orientation_rgb, angle)


        orientation_rgb = np.flipud(orientation_rgb)

        mass_static_flipped = np.flipud(mass_static)

        mass_dynamic_flipped = np.flipud(mass_dynamic)

        hist_color = "#1f77b4"
        hist_linecolor = hist_color

        hist_linewidth = 0.01

        if not self._plotted:

            height_ratios = (1.0, 0.5) if plot_orientation == 'horizontal' else (1.0, 1.0, 1.0)
            width_ratios = (1.0, 1.0, 1.0) if plot_orientation == 'horizontal' else (1, 0.1)

            rows = 2 if plot_orientation == 'horizontal' else 3
            cols = 3 if plot_orientation == 'horizontal' else 2

            gs = gridspec.GridSpec(rows, cols, height_ratios=height_ratios, width_ratios=width_ratios)


            hist_orientation = 'vertical' if plot_orientation == 'horizontal' else 'horizontal'

            g_seq = [0,3,1,4,2,5] if plot_orientation == 'horizontal' else [0,1,2,3,4,5]

            if not hasattr(self, "_static_mass_plot"):
                self._static_mass_plot = self._fig.add_subplot(gs[g_seq[0]])

            self._static_mass_plot.cla()
            self._static_mass_plot.set_title("Static mass")
            self._static_mass_image = self._static_mass_plot.imshow(mass_static_flipped, cmap='Greys', interpolation='nearest', vmin=0.0, vmax=1.0)
            self._static_mass_plot.get_xaxis().set_visible(False)
            self._static_mass_plot.get_yaxis().set_visible(False)

            if not hasattr(self, "_static_mass_hist"):
                self._static_mass_hist = self._fig.add_subplot(gs[g_seq[1]])
            self._static_mass_hist.cla()
            self._static_mass_hist.hist(mass_static[mass_static > 0.0].flatten(), bins=100, range=(0.0, 1.0), linewidth=hist_linewidth, color=hist_color, orientation=hist_orientation)

            self._static_mass_hist.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off')


            if not hasattr(self, "_dynamic_mass_plot"):
                self._dynamic_mass_plot = self._fig.add_subplot(gs[g_seq[2]])
            self._dynamic_mass_plot.cla()
            self._dynamic_mass_plot.set_title("Dynamic mass")
            self._dynamic_mass_image = self._dynamic_mass_plot.imshow(mass_dynamic_flipped, cmap='Greys', interpolation='nearest', vmin=0.0, vmax=1.0)
            self._dynamic_mass_plot.get_xaxis().set_visible(False)
            self._dynamic_mass_plot.get_yaxis().set_visible(False)


            if not hasattr(self, "_dynamic_mass_hist"):
                self._dynamic_mass_hist = self._fig.add_subplot(gs[g_seq[3]])
            self._dynamic_mass_hist.cla()
            self._dynamic_mass_hist.hist(mass_dynamic[mass_dynamic > 0.0].flatten(), bins=100, range=(0.0, 1.0), linewidth=hist_linewidth, color=hist_color, orientation=hist_orientation)


            self._dynamic_mass_hist.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off')

            if not hasattr(self, "_dynamic_orientation_plot"):
                self._dynamic_orientation_plot = self._fig.add_subplot(gs[g_seq[4]])
            self._dynamic_orientation_plot.cla()
            self._dynamic_orientation_plot.set_title("Orientation")
            self._dynamic_orientation_image = self._dynamic_orientation_plot.imshow(orientation_rgb, interpolation='nearest')
            self._dynamic_orientation_plot.get_xaxis().set_visible(False)
            self._dynamic_orientation_plot.get_yaxis().set_visible(False)

            if not hasattr(self, "_orientation_hist"):
                self._orientation_hist = self._fig.add_subplot(gs[g_seq[5]])
            self._orientation_hist.cla()
            self._orientation_hist.hist((orientation / (np.pi * 2) * 360).flatten(), bins=360/4, range=(0.0, 360), weights=mass_dynamic.flatten(), linewidth=hist_linewidth, color=hist_color, orientation=hist_orientation)

            self._orientation_hist.axis('tight')


            self._orientation_hist.set_yticks(np.arange(0, 360+1, 90))
            self._orientation_hist.set_yticklabels([str(a) + r'$^{\circ}$' for a in np.arange(0, 360+1, 90)])



            self._orientation_hist.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off')

            self._fig.tight_layout()

            self._fig.canvas.draw()
            plt.show()


            save_plots = False

            if save_plots:

                dir = "/home/viktor/Sync/School/DP/eval_sim_images"
                if not os.path.exists(dir):
                    os.makedirs(dir)




                extent = self._static_mass_plot.get_window_extent().transformed(self._fig.dpi_scale_trans.inverted())

                file = os.path.join(dir, "static_" + str(self.frame) + ".png")

                self._fig.savefig(file, dpi=300, bbox_inches=extent)

                extent = self._dynamic_mass_plot.get_window_extent().transformed(self._fig.dpi_scale_trans.inverted())

                file = os.path.join(dir, "dynamic_" + str(self.frame) + ".png")

                self._fig.savefig(file, dpi=300, bbox_inches=extent)

                extent = self._dynamic_orientation_plot.get_window_extent().transformed(self._fig.dpi_scale_trans.inverted())

                file = os.path.join(dir, "orientation_" + str(self.frame) + ".png")

                self._fig.savefig(file, dpi=300, bbox_inches=extent)

            self.frame += 1