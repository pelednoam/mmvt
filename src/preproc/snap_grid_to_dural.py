import numpy as np
import os
import os.path as op
import nibabel as nib
from src.utils import utils


def snap_electrodes_to_surface(subject, elecs_pos, grid_name, subjects_dir,
                               max_steps=40000, giveup_steps=10000,
                               init_temp=1e-3, temperature_exponent=1,
                               deformation_constant=1.):
    '''
    Transforms electrodes from surface space to positions on the surface
    using a simulated annealing "snapping" algorithm which minimizes an
    objective energy function as in Dykstra et al. 2012

    Parameters
    ----------
    electrodes : List(Electrode)
        List of electrodes with the surf_coords attribute filled. Caller is
        responsible for filtering these into grids if desired.
    subjects_dir : Str | None
        The freesurfer subjects_dir. If this is None, it is assumed to be the
        $SUBJECTS_DIR environment variable. Needed to access the dural
        surface.
    subject : Str | None
        The freesurfer subject. If this is None, it is assumed to be the
        $SUBJECT environment variable. Needed to access the dural surface.
    max_steps : Int
        The maximum number of steps for the Simulated Annealing algorithm.
        Adding more steps usually causes the algorithm to take longer. The
        default value is 40000. max_steps can be smaller than giveup_steps,
        in which case giveup_steps is ignored
    giveup_steps : Int
        The number of steps after which, with no change of objective function,
        the algorithm gives up. A higher value may cause the algorithm to
        take longer. The default value is 10000.
    init_temp : Float
        The initial annealing temperature. Default value 1e-3
    temperature_exponent : Float
        The exponentially determined temperature when making random changes.
        The value is Texp0 = 1 - Texp/H where H is max_steps
    deformation_constant : Float
        A constant to weight the deformation term of the energy cost. When 1,
        the deformation and displacement are weighted equally. When less than
        1, there is assumed to be considerable deformation and the spring
        condition is weighted more highly than the deformation condition.

    There is no return value. The 'snap_coords' attribute will be used to
    store the snapped locations of the electrodes
    '''
    from scipy.spatial.distance import cdist

    create_dural_surface(subject, subjects_dir)

    n = elecs_pos.shape[0]
    e_init = np.array(elecs_pos)
    snapped_electrodes = np.zeros(elecs_pos.shape)

    # first set the alpha parameter exactly as described in Dykstra 2012.
    # this parameter controls which electrodes have virtual springs connected.
    # this may not matter but doing it is fast and safe
    alpha = np.zeros((n, n))
    init_dist = cdist(e_init, e_init)

    neighbors = []

    k_nei = np.min([n, 6])
    for i in range(n):
        neighbor_vec = init_dist[:, i]
        # take 5 highest neighbors
        h5, = np.where(np.logical_and(neighbor_vec < np.sort(neighbor_vec)[k_nei - 1],
                                      neighbor_vec != 0))

        neighbors.append(h5)

    neighbors = np.squeeze(neighbors)

    # get distances from each neighbor pairing
    neighbor_dists = []
    for i in range(n):
        neighbor_dists.append(init_dist[i, neighbors[i]])

    neighbor_dists = np.hstack(neighbor_dists)

    # collect distance into histogram of resolution 0.2
    max = np.max(np.around(neighbor_dists))
    min = np.min(np.around(neighbor_dists))

    hist, _ = np.histogram(neighbor_dists, bins=int((max - min) / 2), range=(min, max))

    fundist = np.argmax(hist) * 2 + min + 1

    # apply fundist to alpha matrix
    alpha_tweak = 1.75

    for i in range(n):
        neighbor_vec = init_dist[:, i]
        neighbor_vec[i] = np.inf

        neighbors = np.where(neighbor_vec < fundist * alpha_tweak)

        if len(neighbors) > 5:
            neighbors = np.where(neighbor_vec < np.sort(neighbor_vec)[5])

        if len(neighbors) == 0:
            closest = np.argmin(neighbors)
            neighbors = np.where(neighbor_vec < closest * alpha_tweak)

        alpha[i, neighbors] = 1

        for j in range(i):
            if alpha[j, i] == 1:
                alpha[i, j] = 1
            if alpha[i, j] == 1:
                alpha[j, i] = 1

    # alpha is set, now do the annealing
    def energycost(e_new, e_old, alpha):
        n = len(alpha)

        dist_new = cdist(e_new, e_new)
        dist_old = cdist(e_old, e_old)

        H = 0

        for i in range(n):
            H += deformation_constant * float(cdist([e_new[i]], [e_old[i]]))

            for j in range(i):
                H += alpha[i, j] * (dist_new[i, j] - dist_old[i, j]) ** 2

        return H

    # load the dural surface locations
    lh_dura, _ = nib.freesurfer.read_geometry(
        op.join(subjects_dir, subject, 'surf', 'lh.dural'))

    rh_dura, _ = nib.freesurfer.read_geometry(
        op.join(subjects_dir, subject, 'surf', 'rh.dural'))

    # lh_dura[:, 0] -= np.max(lh_dura[:, 0])
    # rh_dura[:, 0] -= np.min(rh_dura[:, 0])

    # align the surfaces correctly
    # in the tkRAS space
    # orig = op.join( subjects_dir, subject, 'mri', 'orig.mgz' )
    # ras2vox = np.linalg.inv(geo.get_vox2rasxfm( orig ))
    # tkr = geo.get_vox2rasxfm(orig, 'vox2ras-tkr')
    # lh_dura = np.array( geo.apply_affine( geo.apply_affine(lh_dura, ras2vox),
    #    tkr))
    # rh_dura = np.array( geo.apply_affine( geo.apply_affine(rh_dura, ras2vox),
    #    tkr))
    # lh_dura = geo.apply_affine( geo.apply_affine(lh_dura, ras2vox), tkr)
    # rh_dura = geo.apply_affine( geo.apply_affine(rh_dura, ras2vox), tkr)

    dura = np.vstack((lh_dura, rh_dura))

    max_deformation = 3
    deformation_choice = 50

    # adjust annealing parameters
    # H determines maximal number of steps
    H = max_steps
    # Texp determines the steepness of temperateure gradient
    Texp = 1 - temperature_exponent / H
    # T0 sets the initial temperature and scales the energy term
    T0 = init_temp
    # Hbrk sets a break point for the annealing
    Hbrk = giveup_steps

    h = 0;
    hcnt = 0
    lowcost = mincost = 1e6

    # start e-init as greedy snap to surface
    e_snapgreedy = dura[np.argmin(cdist(dura, e_init), axis=0)]

    e = np.array(e_snapgreedy).copy()
    emin = np.array(e_snapgreedy).copy()

    # the annealing schedule continues until the maximum number of moves
    while h < H:
        h += 1;
        hcnt += 1
        # terminate if no moves have been made for a long time
        if hcnt > Hbrk:
            break

        # current temperature
        T = T0 * (Texp ** h)

        # select a random electrode
        e1 = np.random.randint(n)
        # transpose it with a *nearby* point on the surface

        # find distances from this point to all points on the surface
        dists = np.squeeze(cdist(dura, [e[e1]]))
        # take a distance within the minimum 5X

        # mindist = np.min(dists)
        mindist = np.sort(dists)[deformation_choice]
        candidate_verts, = np.where(dists < mindist * max_deformation)
        choice_vert = candidate_verts[np.random.randint(len(candidate_verts))]

        e_tmp = e.copy()
        # print choice_vert
        # print np.shape(candidate_verts)
        e_tmp[e1] = dura[choice_vert]

        cost = energycost(e_tmp, e_init, alpha)

        if cost < lowcost or np.random.random() < np.exp(-(cost - lowcost) / T):
            e = e_tmp
            lowcost = cost

            if cost < mincost:
                emin = e
                mincost = cost
                print('step %i ... current lowest cost = %f' % (h, mincost))
                hcnt = 0

            if mincost == 0:
                break
        if h % 200 == 0:
            print('%s %s: step %i ... final lowest cost = %f' % (subject, grid_name, h, mincost))

    # return the emin coordinates
    for ind, loc in enumerate(emin):
        snapped_electrodes[ind] = loc

    # return the nearest vertex on the pial surface
    lh_pia, _ = nib.freesurfer.read_geometry(
        op.join(subjects_dir, subject, 'surf', 'lh.pial'))

    rh_pia, _ = nib.freesurfer.read_geometry(
        op.join(subjects_dir, subject, 'surf', 'rh.pial'))

    # expand the pial surfaces slightly to better visualize the electrodes
    # lh_pia =geo.expand_triangular_mesh(lh_pia, com_bias=(-2, 0, 0), offset=18)
    # rh_pia = geo.expand_triangular_mesh(rh_pia, com_bias=(2, 0, 0), offset=18)


    # adjust x-axis offsets as pysurfer illogically does as hard-coded step
    # lh_pia[:, 0] -= np.max(lh_pia[:, 0])
    # rh_pia[:, 0] -= np.min(rh_pia[:, 0])


    pia = np.vstack((lh_pia, rh_pia))

    e_pia = np.argmin(cdist(pia, emin), axis=0)

    snapped_electrodes_pial = np.zeros(snapped_electrodes.shape)
    for ind, soln in enumerate(e_pia):
        # elec.vertno = soln if soln < len(lh_pia) else soln - len(lh_pia)
        # elec.hemi = 'lh' if soln < len(lh_pia) else 'rh'
        snapped_electrodes_pial[ind] = pia[soln]

    output_fname = op.join(subjects_dir, subject, 'electrodes', '{}_snap_electrodes'.format(grid_name))
    np.savez(output_fname, snapped_electrodes=snapped_electrodes, snapped_electrodes_pial=snapped_electrodes_pial)
    return snapped_electrodes, snapped_electrodes_pial


def create_dural_surface(subject, subjects_dir):
    '''
    Creates the dural surface in the specified subjects_dir. This is done
    using a standalone script derived from the Freesurfer tools which actually
    use the dural surface.

    The caller is responsible for providing a correct subjects_dir, i.e., one
    which is writable. The higher-order logic should detect an unwritable
    directory, and provide a user-sanctioned space to write the new fake
    subjects_dir to.

    Parameters
    ----------
    subjects_dir : Str | None
        The freesurfer subjects_dir. If this is None, it is assumed to be the
        $SUBJECTS_DIR environment variable. If this folder is not writable,
        the program will crash.
    subject : Str | None
        The freesurfer subject. If this is None, it is assumed to be the
        $SUBJECT environment variable.
    '''
    print('create dural surface step')

    scripts_dir = op.join(utils.get_parent_fol(op.dirname(__file__), 2), 'utils')
    os.environ['SCRIPTS_DIR'] = scripts_dir
    print(scripts_dir)

    if (op.exists(op.join(subjects_dir,subject,'surf','lh.dural'))
            and op.exists(op.join(subjects_dir, subject,'surf',
            'rh.dural'))):
        return

    import subprocess

    curdir = os.getcwd()
    os.chdir(op.join(subjects_dir, subject, 'surf'))

    for hemi in ('lh','rh'):
        make_dural_surface_cmd = [op.join(scripts_dir, 
            'make_dural_surface.csh'),'-i','%s.pial'%hemi]
        print(make_dural_surface_cmd)

        # This worked for me only after:
        # 1) Installing python2 (from anaconda), because of mcubes
        # 2) Installing mcubes usign pytohn2 (https://github.com/pmneila/PyMCubes)
        # 3) Open the termianl in the subjects_dir/subject/surf and run those commands:
        #   a) ...code_path/electrodes_rois/src/make_dural_surface.csh -i rh.pial
        #   b) ...code_path/electrodes_rois/src/make_dural_surface.csh -i lh.pial

        p = subprocess.call(make_dural_surface_cmd)

    os.chdir(curdir)