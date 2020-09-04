from synapses_connectivity import get_connectivity_and_synapses_matrix
from toolbox import get_fluct_regime_vars, pseq_params
from mean_field import find_fixed_point_first_order
from mean_field import reformat_syn_parameters
from mean_field import load_transfer_functions
from cell_library import get_neuron_params
from model_params import *
from model_stim import *
import numpy as np

# Some Cython stuff
cimport numpy
cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)


def Euler_method_for_ring_model(str NRN1, str NRN2, str NTWK, str RING, str STIM, float BIN=5e-3, \
                float sim_time=1.0, dict custom_ring_params={}, dict custom_stim_params={}):
    """
    Given two afferent rate input excitatory and inhibitory respectively
    this function computes the prediction of a first order rate model
    (e.g. Wilson and Cowan in the 70s, or 1st order of El Boustani and
    Destexhe 2009) by implementing a simple Euler method
    IN A 2D GRID WITH LATERAL CONNECTIVITY
    the number of laterally connected units is 'connected_neighbors'
    there is an exponential decay of the strength 'decay_connect'
    ----------------------------------------------------------------
    the core of the formalism is the transfer function, see Zerlaut et al. 2015
    it can also be Kuhn et al. 2004 or Amit & Brunel 1997
    -----------------------------------------------------------------
    nu_0 is the starting value value of the recurrent network activity
    it should be the fixed point of the network dynamics
    -----------------------------------------------------------------
    t is the discretization used to solve the euler method
    BIN is the initial sampling bin that should correspond to the
    markovian time scale where the formalism holds (~5ms)
    
    conduction_velocity=0e-3, in ms per pixel
    """
    
    '''
    VARIABLES DECLARATION
    Needed for Cython optimization
    '''
    cdef dict simulation_parameters, params, random_conn_params 
    cdef object[:, :] M
    cdef list M_conn_exc, M_conn_inh
    cdef numpy.ndarray[numpy.float_t, ndim=3] Fe_aff, Fi_aff, Fe, Fi, muVn
    cdef numpy.ndarray[numpy.float_t, ndim=1] t, X, Z
    cdef float ext_drive, fe0, fi0, muV0, conduction_velocity
    cdef int nb_exc_neighb, nb_inh_neighb
    cdef int i_t, i_z, i_x, i_exc, i_inh, it_delayed
    cdef float fe, fi, fe_pure_exc, dt


    simulation_parameters = {
        'NRN1':NRN1,
        'NRN2':NRN2,
        'NTWK':NTWK,
        'RING':RING,
        'STIM':STIM,
        'BIN':BIN,
        'sim_time':sim_time,
        'custom_ring_params':custom_ring_params,
        'custom_stim_params':custom_stim_params
        }

    print('Loading parameters [...]')
    M = get_connectivity_and_synapses_matrix(NTWK, SI_units=True)
    params = get_neuron_params(NRN2, SI_units=True)
    reformat_syn_parameters(params, M)
    ext_drive = M[0,0]['ext_drive']

    print('Computing fixed point [...]')
    fe0, fi0 = find_fixed_point_first_order(NRN1, NRN2, NTWK, exc_aff=ext_drive)
    muV0, _, _, _ = get_fluct_regime_vars(fe0 + ext_drive, fi0, *pseq_params(params))
    
    print('Loading transfer functions [...]')
    TF1, TF2 = load_transfer_functions(NRN1, NRN2, NTWK)

    print('Initializing geometry [...]')
    X, Z, _, _, _, _, _, _, conduction_velocity = pseq_ring_params(RING, custom=custom_ring_params)

    print('Loading connectivity matrix [...]')
    M_conn_exc, M_conn_inh, nb_exc_neighb, nb_inh_neighb, random_conn_params = \
        np.load('data/conn_matrix_torus_random.npy', allow_pickle=True)
    
    print('Initializing stimulation [...]')
    t, Fe_aff = get_stimulation(X, Z, STIM, sim_time, custom=custom_stim_params)
    Fi_aff = 0*Fe_aff # no afferent inhibition yet
    
    print('Initializing model variables [...]')
    Fe, Fi, muVn = 0*Fe_aff + fe0, 0*Fe_aff + fi0, 0*Fe_aff + muV0

    print('Starting simulation:')
    dt = (t[1]-t[0])/10
    

    '''
    Euler method for the activity rate
    '''
    # Time loop
    for i_t in range(len(t)-1): 
        # Simple time print 
        if i_t % 100 == 0:
            print('---- Computing time t =', i_t/2, 'ms out of', int(sim_time*1000), 'ms')
        # To shorten simulation time
        # if i_t == 101:
        #     print('----- End of temporal loop')
        #     break

        # Loop over every mean field network
        for i_z in range(len(Z)): 
            for i_x in range(len(X)): 

                fe = ext_drive
                fi = 0
                fe_pure_exc = Fe_aff[i_t, i_x, i_z]
                
                # Excitatory neighbours
                for i_exc in range(len(M_conn_exc[i_z][i_x])):
                    # Delay in propagation due to limited axon conduction
                    if i_t > int(abs(M_conn_exc[i_z][i_x][i_exc]['dist'])/conduction_velocity/dt):
                        it_delayed = i_t-int(abs(M_conn_exc[i_z][i_x][i_exc]['dist'])/conduction_velocity/dt)
                    else:
                        it_delayed = 0
                    # Using the connectivity matrix to find the weight and the position of this neighbour
                    fe += M_conn_exc[i_z][i_x][i_exc]['weight'] * \
                        Fe[it_delayed, M_conn_exc[i_z][i_x][i_exc]['pos_x'], M_conn_exc[i_z][i_x][i_exc]['pos_z']]

                # Inhibitory neighbours
                for i_inh in range(len(M_conn_inh[i_z][i_x])):
                    # Delay in propagation due to limited axon conduction
                    if i_t>int(abs(M_conn_inh[i_z][i_x][i_inh]['dist'])/conduction_velocity/dt):
                        it_delayed = i_t-int(abs(M_conn_inh[i_z][i_x][i_inh]['dist'])/conduction_velocity/dt)
                    else:
                        it_delayed = 0
                    # Using the connectivity matrix to find the weight and the position of this neighbour
                    fi += M_conn_inh[i_z][i_x][i_inh]['weight'] * \
                        Fi[it_delayed, M_conn_inh[i_z][i_x][i_inh]['pos_x'], M_conn_inh[i_z][i_x][i_inh]['pos_z']]

                # Model output
                muVn[i_t+1, i_x, i_z], _, _, _ = get_fluct_regime_vars(fe, fi, *pseq_params(params))
                Fe[i_t+1, i_x, i_z] = Fe[i_t, i_x, i_z] + dt/BIN*(TF1(fe + fe_pure_exc,fi) - Fe[i_t, i_x, i_z])
                Fi[i_t+1, i_x, i_z] = Fi[i_t, i_x, i_z] + dt/BIN*(TF2(fe,fi) - Fi[i_t, i_x, i_z])

    print('Simulation finished')

    return simulation_parameters, random_conn_params, t, X, Z, Fe_aff, Fe, Fi, np.abs((muVn-muV0)/muV0)
