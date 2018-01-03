import numpy as np
import numpy.linalg as npl
import os.path as op
import traceback

try:
    import bpy
    import mmvt_utils as mu
    from coloring_panel import calc_colors
    IN_BLENDER = True
except:
    from src.mmvt_addon import mmvt_utils as mu
    from src.mmvt_addon.mmvt_utils import calc_colors_from_cm as calc_colors
    IN_BLENDER = False


def init(modality, modality_data=None, colormap=None, subject='', mmvt_dir=''):
    if subject == '':
        subject = mu.get_user()
    if mmvt_dir == '':
        mmvt_dir = mu.file_fol()
    if modality_data is None:
        if modality == 'ct':
            fname = op.join(mmvt_dir, subject, 'ct', 'ct_data.npz'.format(modality))
        else:
            fname = op.join(mmvt_dir, subject, 'freeview', '{}_data.npz'.format(modality))
        if op.isfile(fname):
            modality_data = mu.Bag(np.load(fname))
        else:
            print('To see the slices the following command is being called:'.format(modality))
            print('python -m src.preproc.anatomy -s {} -f save_images_data_and_header'.format(mu.get_user()))
            cmd = '{} -m src.preproc.anatomy -s {} -f save_subject_orig_trans,save_images_data_and_header --ignore_missing 1'.format(
                bpy.context.scene.python_cmd, mu.get_user())
            mu.run_command_in_new_thread(cmd, False)
            return None
    if colormap is None:
        colormap_fname = op.join(mmvt_dir, 'color_maps', 'gray.npy')
        colormap = np.load(colormap_fname)
    affine = np.array(modality_data.affine, float)
    data = modality_data.data
    clim = modality_data.precentiles
    colors_ratio = modality_data.colors_ratio
    codes = axcodes2ornt(aff2axcodes(affine))
    order = np.argsort([c[0] for c in codes])
    print(modality, order)
    flips = np.array([c[1] < 0 for c in codes])[order]
    flips[0] = not flips[0]
    sizes = [data.shape[order] for order in order]
    scalers = voxel_sizes(affine)
    r = [scalers[order[2]] / scalers[order[1]],
         scalers[order[2]] / scalers[order[0]],
         scalers[order[1]] / scalers[order[0]]]
    extras = [0] * 3
    self = mu.Bag(dict(
        data=data, affine=affine, order=order, sizes=sizes, flips=flips, clim=clim, r=r, colors_ratio=colors_ratio,
        colormap=colormap, coordinates=[], modality=modality, extras=extras))
    return self


def create_slices(xyz, state=None, modalities='mri', modality_data=None, colormap=None, plot_cross=True):
    self = mu.Bag({})
    if isinstance(modalities, str):
        modalities = [modalities]
    # modalities = set(modalities)
    # if 'mri' not in modalities:
    #     modalities.append('mri')
    for modality in modalities:
        if state is None or modality not in state:
            self[modality] = init(modality, modality_data, colormap)
        else:
            self[modality] = state[modality]
    mri = state['mri']
    if mri is None:
        return None

    x, y, z = xyz[:3]
    for modality in modalities:
        self[modality].coordinates = np.rint(np.array([x, y, z])[self[modality].order]).astype(int)
    cross_vert, cross_horiz = calc_cross(self[modality].coordinates, self[modality].sizes, self[modality].flips)
    images = {}
    xaxs, yaxs = [1, 0, 0], [2, 2, 1]
    max_xaxs_size = max([self[modality].sizes[xax] for xax, modality in zip(xaxs, modalities)])
    max_yaxs_size = max([self[modality].sizes[yax] for yax, modality in zip(yaxs, modalities)])
    max_sizes = (max_xaxs_size, max_yaxs_size)
    self[modality].cross = [None] * 3
    for ii, xax, yax, prespective, label in zip(
            [0, 1, 2], xaxs, yaxs, ['sagital', 'coronal', 'axial'], ('SAIP', 'SLIR', 'ALPR')):
        pixels = {}
        cross = (int(cross_horiz[ii][0, 1]), int(cross_vert[ii][0, 0]))
        self[modality].cross[ii] = cross
        for modality in modalities:
            s = self[modality]
            d = get_image_data(s.data, s.order, s.flips, ii, s.coordinates)
            # todo: Should do that step in the state init
            if modality == 'ct':
                d[np.where(d == 0)] = -200
            sizes = (s.sizes[xax], s.sizes[yax])
            self[modality].extras[ii] = (int((max_sizes[0] - sizes[0])/2), int((max_sizes[1] - sizes[1])/2))
            pixels[modality] = calc_slice_pixels(d, sizes, max_sizes, s.clim, s.colors_ratio, s.colormap)
        # image = create_image(d, sizes, max_sizes, s.clim, s.colors_ratio, prespective, s.colormap,
        #                      int(cross_horiz[ii][0, 1]), int(cross_vert[ii][0, 0]),
        #                      state[modality].extras[ii])
        if 'mri' in modalities and 'ct' in modalities:
            ct_ratio = bpy.context.scene.slices_modality_mix
            pixels = (1 - ct_ratio) * pixels['mri'] + ct_ratio * pixels['ct']
        else:
            pixels = pixels[modality]
        if plot_cross:
            pixels = add_cross_to_pixels(pixels, max_sizes, cross, state[modality].extras[ii])
        if IN_BLENDER:
            image = create_image(pixels, max_sizes, prespective)
            if image is not None:
                images[prespective] = image
        else:
            images[prespective] = pixels
    # print(np.dot(state[modality].affine, [x, y, z, 1])[:3])
    return images


def calc_cross(coordinates, sizes, flips):
    verts, horizs = [None] * 3, [None] * 3
    for ii, xax, yax in zip([0, 1, 2], [1, 0, 0], [2, 2, 1]):
        verts[ii] = np.array([[0] * 2, [-0.5, sizes[yax] - 0.5]]).T
        horizs[ii] = np.array([[-0.5, sizes[xax] - 0.5], [0] * 2]).T
    for ii, xax, yax in zip([0, 1, 2], [1, 0, 0], [2, 2, 1]):
        loc = coordinates[ii]
        if flips[ii]:
            loc = sizes[ii] - loc
        loc = [loc] * 2
        if ii == 0:
            verts[2][:, 0] = loc
            verts[1][:, 0] = loc
        elif ii == 1:
            horizs[2][:, 1] = loc
            verts[0][:, 0] = loc
        else:  # ii == 2
            horizs[1][:, 1] = loc
            horizs[0][:, 1] = loc
    return verts, horizs


def get_image_data(image_data, order, flips, ii, pos):
    try:
        data = np.rollaxis(image_data, axis=order[ii])[pos[ii]]  # [data_idx] # [pos[ii]]
    except:
        print('get_image_data: No data for {}'.format(pos))
        return np.zeros((256, 256))
    xax = [1, 0, 0][ii]
    yax = [2, 2, 1][ii]
    if order[xax] < order[yax]:
        data = data.T
    if flips[xax]:
        data = data[:, ::-1]
    if flips[yax]:
        data = data[::-1]
    return data


def calc_slice_pixels(data, sizes, max_sizes, clim, colors_ratio, colormap):
    colors = calc_colors(data, clim[0], colors_ratio, colormap)
    pixels = np.ones((colors.shape[0], colors.shape[1], 4))
    pixels[:, :, :3] = colors
    # todo: check all the other cases
    extra = [int((max_sizes[0] - sizes[0]) / 2), int((max_sizes[1] - sizes[1]) / 2)]
    if max_sizes[0] > sizes[0] and max_sizes[1] == sizes[1]:
        dark = np.zeros((colors.shape[0], extra[0], 4))
        dark[:, :, 3] = 1
        pixels = np.concatenate((dark, pixels, dark), axis=1)
    return pixels


def add_cross_to_pixels(pixels, max_sizes, cross, extra):
    if 0 <= cross[1] < max_sizes[1]:  # data.shape[1]:
        for x in range(max_sizes[0]):  # data.shape[0]):
            pixels[x, cross[1] + extra[0]] = [0, 1, 0, 1]
    if 0 <= cross[0] < max_sizes[0]:  # data.shape[0]:
        for y in range(max_sizes[1]):  # data.shape[1]):
            pixels[cross[0], y] = [0, 1, 0, 1]
    return pixels


# def create_image(data, sizes, max_sizes, clim, colors_ratio, prespective, colormap, horz_cross, vert_corss, extra):
def create_image(pixels, max_sizes, prespective):
    image_name = '{}.png'.format(prespective)
    if image_name not in bpy.data.images:
        image = bpy.data.images.new(image_name, width=max_sizes[0], height=max_sizes[1])
    else:
        image = bpy.data.images[image_name]
    # print([im.name for im in bpy.data.images])
    try:
        # pixels = calc_slice_pixels(data, sizes, max_sizes, clim, colors_ratio, colormap)
        # print(prespective, horz_cross, vert_corss)

        # if 0 <= cross[1] < max_sizes[1]: # data.shape[1]:
        #     for x in range(max_sizes[0]): #data.shape[0]):
        #         pixels[x, cross[1] + extra[0]] = [0, 1, 0, 1]
        # if 0 <= cross[0] < max_sizes[0]: #data.shape[0]:
        #     for y in range(max_sizes[1]): #data.shape[1]):
        #         pixels[cross[0], y] = [0, 1, 0, 1]

        # pixels[:, :, 3] = 0.5
        image.pixels = pixels.ravel()
        return image
    except:
        print(traceback.format_exc())
        return None


def on_click(ii, xy, state, modality='mri'):
    # if 'mri' in state:
    #     org_modality = modality
    #     modality = 'mri'
    s = state[modality]
    x = xy[0] - state[modality].extras[ii][0]
    y = xy[1] - state[modality].extras[ii][1]
    xax, yax = [[1, 2], [0, 2], [0, 1]][ii]
    if modality == 'mri':
        trans = [[0, 1, 2], [2, 0, 1], [1, 2, 0]][ii]
    elif modality == 'ct':
        trans = [[2, 1, 0], [1, 0, 2], [0, 2, 1]][ii]
    else:
        print('The trans should be first calculated for {}!'.format(modality))
        trans = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
    x = s.sizes[xax] - x if s.flips[xax] else x
    y = s.sizes[yax] - y if s.flips[yax] else y
    idxs = [None, None, None]
    idxs[xax] = y
    idxs[yax] = x
    idxs[ii] = s.coordinates[ii]
    print(ii, xax, yax, x, y, s.sizes, idxs, state[modality].extras)
    idxs = [idxs[ind] for ind in trans]
    # print(idxs)
    # print('Create new slices after click {} changed {},{}'.format(idxs, xax, yax))
    create_slices(idxs, state, modality) # np.dot(state.affine, idxs)[:3]
    return idxs


def plot_slices(xyz, state, modality='mri', interactive=True, pixels_around_voxel=20, fig_fname=''):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    def update(val):
        pixels_around_voxel = _pixels_around_voxel.val
        for img_num, ax in enumerate(axs.ravel()):
            x_vox_slice = get_image_data(x_vox, s.order, s.flips, img_num, s.coordinates)
            y, x = np.argwhere(x_vox_slice)[0]
            ax.set_xlim([x - pixels_around_voxel, x + pixels_around_voxel])
            ax.set_ylim([y - pixels_around_voxel, y + pixels_around_voxel])
        fig.canvas.draw_idle()

    fig, axs = plt.subplots(1, 3)#"#, True, True)
    plt.tight_layout()
    if fig_fname == '':
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

    if interactive and fig_fname == '':
        axcolor = 'lightgoldenrodyellow'
        axamp = plt.axes([0.25, 0.15, 0.5, 0.03], facecolor=axcolor)
        _pixels_around_voxel = Slider(axamp, 'Zoom', 1, 256/2, valinit=pixels_around_voxel)
        _pixels_around_voxel.on_changed(update)

    s = state[modality]
    fig.suptitle('Voxel {} ({:.2f})'.format(tuple(xyz), s.data[tuple(xyz)]))
    x_vox = np.zeros_like(s.data)
    x_vox[tuple(xyz)] = 255
    images = create_slices(xyz, state, modalities=modality, plot_cross=False)
    for img_num, ((pers, data), ax) in enumerate(zip(images.items(), axs.ravel())):
        x_vox_slice = get_image_data(x_vox, s.order, s.flips, img_num, s.coordinates)
        y, x = np.argwhere(x_vox_slice)[0]

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.imshow(data, origin='lower')
        ax.imshow(x_vox_slice, cmap=plt.cm.Reds, alpha=0.5)
        ax.axhline(y)
        ax.axvline(x)
        ax.set_xlim([x - pixels_around_voxel, x + pixels_around_voxel])
        ax.set_ylim([y - pixels_around_voxel, y + pixels_around_voxel])
        ax.set_title(pers)

    if fig_fname == '':
        plt.show()
    else:
        fig = plt.gcf()
        fig.set_size_inches((20, 8.5), forward=False)
        print('Pic was saved in {}'.format(fig_fname))
        plt.savefig(fig_fname, dpi=500)


# Most of this code is taken from nibabel
def axcodes2ornt(axcodes, labels=None):
    """ Convert axis codes `axcodes` to an orientation

    Parameters
    ----------
    axcodes : (N,) tuple
        axis codes - see ornt2axcodes docstring
    labels : optional, None or sequence of (2,) sequences
        (2,) sequences are labels for (beginning, end) of output axis.  That
        is, if the first element in `axcodes` is ``front``, and the second
        (2,) sequence in `labels` is ('back', 'front') then the first
        row of `ornt` will be ``[1, 1]``. If None, equivalent to
        ``(('L','R'),('P','A'),('I','S'))`` - that is - RAS axes.

    Returns
    -------
    ornt : (N,2) array-like
        orientation array - see io_orientation docstring

    Examples
    --------
    >>> axcodes2ornt(('F', 'L', 'U'), (('L','R'),('B','F'),('D','U')))
    array([[ 1.,  1.],
           [ 0., -1.],
           [ 2.,  1.]])
    """
    if labels is None:
        labels = list(zip('LPI', 'RAS'))
    n_axes = len(axcodes)
    ornt = np.ones((n_axes, 2), dtype=np.int8) * np.nan
    for code_idx, code in enumerate(axcodes):
        for label_idx, codes in enumerate(labels):
            if code is None:
                continue
            if code in codes:
                if code == codes[0]:
                    ornt[code_idx, :] = [label_idx, -1]
                else:
                    ornt[code_idx, :] = [label_idx, 1]
                break
    return ornt


def aff2axcodes(aff, labels=None, tol=None):
    """ axis direction codes for affine `aff`

    Parameters
    ----------
    aff : (N,M) array-like
        affine transformation matrix
    labels : optional, None or sequence of (2,) sequences
        Labels for negative and positive ends of output axes of `aff`.  See
        docstring for ``ornt2axcodes`` for more detail
    tol : None or float
        Tolerance for SVD of affine - see ``io_orientation`` for more detail.

    Returns
    -------
    axcodes : (N,) tuple
        labels for positive end of voxel axes.  Dropped axes get a label of
        None.

    Examples
    --------
    >>> aff = [[0,1,0,10],[-1,0,0,20],[0,0,1,30],[0,0,0,1]]
    >>> aff2axcodes(aff, (('L','R'),('B','F'),('D','U')))
    ('B', 'R', 'U')
    """
    ornt = io_orientation(aff, tol)
    return ornt2axcodes(ornt, labels)


def ornt2axcodes(ornt, labels=None):
    """ Convert orientation `ornt` to labels for axis directions

    Parameters
    ----------
    ornt : (N,2) array-like
        orientation array - see io_orientation docstring
    labels : optional, None or sequence of (2,) sequences
        (2,) sequences are labels for (beginning, end) of output axis.  That
        is, if the first row in `ornt` is ``[1, 1]``, and the second (2,)
        sequence in `labels` is ('back', 'front') then the first returned axis
        code will be ``'front'``.  If the first row in `ornt` had been
        ``[1, -1]`` then the first returned value would have been ``'back'``.
        If None, equivalent to ``(('L','R'),('P','A'),('I','S'))`` - that is -
        RAS axes.

    Returns
    -------
    axcodes : (N,) tuple
        labels for positive end of voxel axes.  Dropped axes get a label of
        None.

    Examples
    --------
    >>> ornt2axcodes([[1, 1],[0,-1],[2,1]], (('L','R'),('B','F'),('D','U')))
    ('F', 'L', 'U')
    """
    if labels is None:
        labels = list(zip('LPI', 'RAS'))
    axcodes = []
    for axno, direction in np.asarray(ornt):
        if np.isnan(axno):
            axcodes.append(None)
            continue
        axint = int(np.round(axno))
        if axint != axno:
            raise ValueError('Non integer axis number %f' % axno)
        elif direction == 1:
            axcode = labels[axint][1]
        elif direction == -1:
            axcode = labels[axint][0]
        else:
            raise ValueError('Direction should be -1 or 1')
        axcodes.append(axcode)
    return tuple(axcodes)


def io_orientation(affine, tol=None):
    ''' Orientation of input axes in terms of output axes for `affine`

    Valid for an affine transformation from ``p`` dimensions to ``q``
    dimensions (``affine.shape == (q + 1, p + 1)``).

    The calculated orientations can be used to transform associated
    arrays to best match the output orientations. If ``p`` > ``q``, then
    some of the output axes should be considered dropped in this
    orientation.

    Parameters
    ----------
    affine : (q+1, p+1) ndarray-like
       Transformation affine from ``p`` inputs to ``q`` outputs.  Usually this
       will be a shape (4,4) matrix, transforming 3 inputs to 3 outputs, but
       the code also handles the more general case
    tol : {None, float}, optional
       threshold below which SVD values of the affine are considered zero. If
       `tol` is None, and ``S`` is an array with singular values for `affine`,
       and ``eps`` is the epsilon value for datatype of ``S``, then `tol` set
       to ``S.max() * max((q, p)) * eps``

    Returns
    -------
    orientations : (p, 2) ndarray
       one row per input axis, where the first value in each row is the closest
       corresponding output axis. The second value in each row is 1 if the
       input axis is in the same direction as the corresponding output axis and
       -1 if it is in the opposite direction.  If a row is [np.nan, np.nan],
       which can happen when p > q, then this row should be considered dropped.
    '''
    affine = np.asarray(affine)
    q, p = affine.shape[0] - 1, affine.shape[1] - 1
    # extract the underlying rotation, zoom, shear matrix
    RZS = affine[:q, :p]
    zooms = np.sqrt(np.sum(RZS * RZS, axis=0))
    # Zooms can be zero, in which case all elements in the column are zero, and
    # we can leave them as they are
    zooms[zooms == 0] = 1
    RS = RZS / zooms
    # Transform below is polar decomposition, returning the closest
    # shearless matrix R to RS
    P, S, Qs = npl.svd(RS)
    # Threshold the singular values to determine the rank.
    if tol is None:
        tol = S.max() * max(RS.shape) * np.finfo(S.dtype).eps
    keep = (S > tol)
    R = np.dot(P[:, keep], Qs[keep])
    # the matrix R is such that np.dot(R,R.T) is projection onto the
    # columns of P[:,keep] and np.dot(R.T,R) is projection onto the rows
    # of Qs[keep].  R (== np.dot(R, np.eye(p))) gives rotation of the
    # unit input vectors to output coordinates.  Therefore, the row
    # index of abs max R[:,N], is the output axis changing most as input
    # axis N changes.  In case there are ties, we choose the axes
    # iteratively, removing used axes from consideration as we go
    ornt = np.ones((p, 2), dtype=np.int8) * np.nan
    for in_ax in range(p):
        col = R[:, in_ax]
        if not np.allclose(col, 0):
            out_ax = np.argmax(np.abs(col))
            ornt[in_ax, 0] = out_ax
            assert col[out_ax] != 0
            if col[out_ax] < 0:
                ornt[in_ax, 1] = -1
            else:
                ornt[in_ax, 1] = 1
            # remove the identified axis from further consideration, by
            # zeroing out the corresponding row in R
            R[out_ax, :] = 0
    return ornt


def voxel_sizes(affine):
    r""" Return voxel size for each input axis given `affine`

    The `affine` is the mapping between array (voxel) coordinates and mm
    (world) coordinates.

    The voxel size for the first voxel (array) axis is the distance moved in
    world coordinates when moving one unit along the first voxel (array) axis.
    This is the distance between the world coordinate of voxel (0, 0, 0) and
    the world coordinate of voxel (1, 0, 0).  The world coordinate vector of
    voxel coordinate vector (0, 0, 0) is given by ``v0 = affine.dot((0, 0, 0,
    1)[:3]``.  The world coordinate vector of voxel vector (1, 0, 0) is
    ``v1_ax1 = affine.dot((1, 0, 0, 1))[:3]``.  The final 1 in the voxel
    vectors and the ``[:3]`` at the end are because the affine works on
    homogenous coodinates.  The translations part of the affine is ``trans =
    affine[:3, 3]``, and the rotations, zooms and shearing part of the affine
    is ``rzs = affine[:3, :3]``. Because of the final 1 in the input voxel
    vector, ``v0 == rzs.dot((0, 0, 0)) + trans``, and ``v1_ax1 == rzs.dot((1,
    0, 0)) + trans``, and the difference vector is ``rzs.dot((0, 0, 0)) -
    rzs.dot((1, 0, 0)) == rzs.dot((1, 0, 0)) == rzs[:, 0]``.  The distance
    vectors in world coordinates between (0, 0, 0) and (1, 0, 0), (0, 1, 0),
    (0, 0, 1) are given by ``rzs.dot(np.eye(3)) = rzs``.  The voxel sizes are
    the Euclidean lengths of the distance vectors.  So, the voxel sizes are
    the Euclidean lengths of the columns of the affine (excluding the last row
    and column of the affine).

    Parameters
    ----------
    affine : 2D array-like
        Affine transformation array.  Usually shape (4, 4), but can be any 2D
        array.

    Returns
    -------
    vox_sizes : 1D array
        Voxel sizes for each input axis of affine.  Usually 1D array length 3,
        but in general has length (N-1) where input `affine` is shape (M, N).
    """
    top_left = affine[:-1, :-1]
    return np.sqrt(np.sum(top_left ** 2, axis=0))
