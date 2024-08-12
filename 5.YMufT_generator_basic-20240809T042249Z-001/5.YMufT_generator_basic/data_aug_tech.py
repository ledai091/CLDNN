import numpy as np
import scipy
from scipy import ndimage

class DataAug():
    def transform_matrix_offset_center(self, matrix, x, y):
        o_x = float(x) / 2 + 0.5
        o_y = float(y) / 2 + 0.5
        offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
        reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
        transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
        return transform_matrix

    def apply_affine_transform(self, x, theta=0, zx=1, zy=1,
                               row_axis=0, col_axis=1, channel_axis=2,
                               fill_mode='nearest', cval=0., order=1):

        if scipy is None:
            raise ImportError('Image transformations require SciPy. '
                              'Install SciPy.')
        transform_matrix = None
        if theta != 0:
            theta = np.deg2rad(theta)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            if transform_matrix is None:
                transform_matrix = zoom_matrix
            else:
                transform_matrix = np.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            h, w = x.shape[row_axis], x.shape[col_axis]
            transform_matrix = self.transform_matrix_offset_center(
                transform_matrix, h, w)
            x = np.rollaxis(x, channel_axis, 0)
            final_affine_matrix = transform_matrix[:2, :2]
            final_offset = transform_matrix[:2, 2]

            channel_images = [ndimage.interpolation.affine_transform(
                x_channel,
                final_affine_matrix,
                final_offset,
                order=order,
                mode=fill_mode,
                cval=cval) for x_channel in x]
            x = np.stack(channel_images, axis=0)
            x = np.rollaxis(x, 0, channel_axis + 1)
        return x

    def random_rotation(self, x, rg, channel_axis=0, fill_mode='nearest', cval=0., interpolation_order=1):

        theta = np.random.uniform(-rg, rg)
        x = self.apply_affine_transform(x, theta=theta, channel_axis=2,
                                        fill_mode=fill_mode, cval=cval,
                                        order=interpolation_order)
        return x

    def random_zoom(self, x, zoom_range, channel_axis=0,
                    fill_mode='nearest', cval=0., interpolation_order=1):

        if len(zoom_range) != 2:
            raise ValueError('`zoom_range` should be a tuple or list of two'
                             ' floats. Received: %s' % (zoom_range,))

        if zoom_range[0] == 1 and zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
        x = self.apply_affine_transform(x, zx=zx, zy=zy, channel_axis=2,
                                   fill_mode=fill_mode, cval=cval,
                                   order=interpolation_order)
        return x