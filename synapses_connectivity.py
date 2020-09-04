import numpy as np

def get_connectivity_and_synapses_matrix(NAME, SI_units=True):
    """
    Synapses and connectivity default parameters
    Returns M the matrix containing all these parameters
    """

    # Creates empty array of objects
    M = np.empty((2, 2), dtype=object)

    if NAME=='CONFIG1':
        exc_pop = {'p_conn':0.05/4, 'Q':1., 'Tsyn':5., 'Erev':0.}
        inh_pop = {'p_conn':0.05/4, 'Q':5., 'Tsyn':5., 'Erev':-80.}
        M[:,0] = [exc_pop.copy(), inh_pop.copy()] # post-synaptic : exc
        M[:,1] = [exc_pop.copy(), inh_pop.copy()] # post-synaptic : inh
        M[0,0]['name'], M[1,0]['name'] = 'ee', 'ie'
        M[0,1]['name'], M[1,1]['name'] = 'ei', 'ii'
        # In the first element we put the network number and connectivity information
        M[0,0]['Ntot'], M[0,0]['gei'] = 10000, 0.2
        M[0,0]['ext_drive'] = 2. # we also store here the choosen excitatory drive
        M[0,0]['afferent_exc_fraction'] = 1. # we also store here the choosen excitatory drive
     
    else:
        print('====================================================')
        print('------------ NETWORK NOT RECOGNIZED !! ---------------')
        print('====================================================')

    if SI_units:
        for m in M.flatten():
            m['Q'] *= 1e-9
            m['Erev'] *= 1e-3
            m['Tsyn'] *= 1e-3
    else:
        print('synaptic network parameters --NOT-- in SI units')

    return M