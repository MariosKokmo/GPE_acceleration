import numpy as np

def _create_vortex_list(imprint_position_x, imprint_position_y, imprinting_charge, imprint_times):
    """
    Takes as input 3 arrays. Each array is for one simulation and
    can contain multiple sub-arrays. Each sub-array contains the vortex-related
    parameters for a differetn imprint time.
    
    Returns:
    --------
        np.array, 3D numpy array. 3 x #of vortices x #of imprints
    Example:
    --------
    imprinting_charge = [ [1,2], [3,4], [5,6] ]
    There are 3 imprint times. In the first imprint we imprint 2 vortices
    one with charge 1 and the other with charge 2. In the second imprint the
    two vortices that are imprinted are of charge 3 and 4 respectively etc.
    """
    assert len(imprint_position_x) == len(imprint_position_y)
    assert len(imprint_position_x) == len(imprinting_charge)
    vortex_list_all_iterations = []
    for idx, _ in enumerate(imprint_times):
        vort_x = imprint_position_x[idx]
        vort_y = imprint_position_y[idx]
        vort_charge = imprinting_charge[idx]
        vort = np.vstack((vort_x, vort_y, vort_charge))
        vort = np.expand_dims(vort, axis = 0)
        vortex_list_all_iterations.append(vort)
    return np.concatenate(vortex_list_all_iterations, axis=0)

def _calculate_all_phases(imprinting_vortices):
        """
        Calculates a special phase. This could be a huge anti-charge etc.

        Input:
            imprinting_vortices: np.array, shape=(no. imprints, 3, no.vortices)
            each element is a collection
            of vortex positions along with their charges

        """
        for _, imprint in enumerate(imprinting_vortices):
            x = tuple(imprint[0])
            y = tuple(imprint[1])
            charge = tuple(imprint[2])
            key = (x,y,charge)
            print(str(key))

imprint_position_x = np.array([[0],[0],[0]])
imprint_position_y = np.array([[0],[0],[0]])
imprinting_charge = np.array([[1],[1],[1]])
imprint_times = np.array([7,10,15])

vortex_array = _create_vortex_list(imprint_position_x, imprint_position_y, imprinting_charge, imprint_times)
print(vortex_array)

_calculate_all_phases(vortex_array)