import numpy as np

# Set of default parameters for the 2D model
default_params = {\
'X_discretization':30.,
'X_extent':36., # mm
'Z_discretization':30.,
'Z_extent':36., # mm
'exc_connect_extent':5., # mm
'inh_connect_extent':1., # mm
'conduction_velocity_mm_s': 300. # mm/s
}


def pixels_per_mm(MODEL):
    params = get_model_params(MODEL)
    return params['X_discretization']/params['X_extent']
    
def mm_per_pixel(MODEL):
    params = get_model_params(MODEL)
    return params['X_extent']/params['X_discretization']

def from_mm_to_discretized_model(params):
    """
    Convert all quantities of the model from mm to pixels
    """
    params['mm_per_pixel'] = params['X_extent']/params['X_discretization']
    params['exc_decay_connect'] = params['exc_connect_extent']/params['mm_per_pixel']
    params['inh_decay_connect'] = params['inh_connect_extent']/params['mm_per_pixel']
    # In practice connectivity extends up to 3 std dev.
    params['exc_connected_neighbors'] = int(3.*params['exc_decay_connect']/params['mm_per_pixel'])
    params['inh_connected_neighbors'] = int(3.*params['inh_decay_connect']/params['mm_per_pixel'])
    params['conduction_velocity'] = params['conduction_velocity_mm_s']/params['mm_per_pixel']

def get_model_params(MODEL, custom={}):
    """
    we start with the passive parameters, by default
    they will be overwritten by some specific models
    (e.g. Adexp-Rs, Wang-Buszaki) and not by others (IAF, EIF)
    """
    params = default_params
    
    # Overiding default params by custom params
    for key, val in custom.items():
        params[key] = val
    
    """
    Default parameters are ok for the modelisation standard integrate and fire models
    We basically just use the default parameters for the 2D model
    Note that I have never tested the 'HD' versions - if you want to test them out don't forget
    to build the connectivity matrix in their 'HD' version as well
    """
    if MODEL=='RING-2D':
        # params by default 
        params['name'] = MODEL
    elif MODEL=='RING-2D-HD':
        # params by default 
        params['name'] = MODEL
        params['X_discretization']=100.
        params['Z_discretization']=100.
    elif MODEL=='SHEET-2D':
        # params by default 
        params['name'] = MODEL
    elif MODEL=='SHEET-2D-HD':
        # params by default 
        params['name'] = MODEL
        params['X_discretization']=100.
        params['Z_discretization']=100.
    else:
        params = None
        print('==========> ||ring model not recognized|| <================')

    # Discretization
    from_mm_to_discretized_model(params)
    return params


def gaussian_connectivity(x, x0, dx, normalization):
    """
    Determines the weight of the connectivity between each neighbouring network
    Normalization factor can be modified in the 'NORMALIZATION' section of the connectivity
    matrix
    """
    return normalization*1./(np.sqrt(2.*np.pi)*(dx+1e-12))*np.exp(-(x-x0)**2/2./(1e-12+dx)**2)


def pseq_ring_params(RING, custom={}):
    """ 
    - Retrieves model parameters 
    - Builds neighbour's matrix
    - Builds X and Z vectors for the position of the networks
    """
    # Model parameters
    params = get_model_params(RING, custom=custom)
    exc_connected_neighbors = params['exc_connected_neighbors']
    exc_decay_connect = params['exc_decay_connect']
    inh_connected_neighbors = params['inh_connected_neighbors']
    inh_decay_connect = params['inh_decay_connect']
    conduction_velocity = params['conduction_velocity']

    # Construction of neighbour's 2D matrixes
    # Excitatory population
    radius_exc = exc_connected_neighbors
    y,x = np.ogrid[-radius_exc: radius_exc+1, -radius_exc: radius_exc+1]
    Xn_exc = np.sqrt(x**2+y**2)
    for row in range(len(Xn_exc[0])):
        for col in range(len(Xn_exc[1])):
            if Xn_exc[row,col]>radius_exc:
                Xn_exc[row,col]=-99  
            else:
                if col <= len(Xn_exc[1])//2-1:
                    Xn_exc[row,col] = - Xn_exc[row,col] # - sign on the left
                else:
                    Xn_exc[row,col] = Xn_exc[row,col]  # + sign on the right
    # Inhibitory population
    radius_inh = inh_connected_neighbors
    y,x = np.ogrid[-radius_inh: radius_inh+1, -radius_inh: radius_inh+1]
    Xn_inh = np.sqrt(x**2+y**2)
    for row in range(len(Xn_inh[0])):
        for col in range(len(Xn_inh[1])):
            if Xn_inh[row,col]>radius_inh:
                Xn_inh[row,col]=-99     
            else:
                if col <= len(Xn_inh[1])//2-1:
                    Xn_inh[row,col] = - Xn_inh[row,col]
                else:
                    Xn_inh[row,col] = Xn_inh[row,col]

    # Construction of X and Z vectors
    X = np.linspace(0, params['X_extent'], int(params['X_discretization']), endpoint=True)
    Z = np.linspace(0, params['Z_extent'], int(params['Z_discretization']), endpoint=True)
    return X, Z, Xn_exc, Xn_inh, exc_connected_neighbors, exc_decay_connect,\
        inh_connected_neighbors, inh_decay_connect, conduction_velocity