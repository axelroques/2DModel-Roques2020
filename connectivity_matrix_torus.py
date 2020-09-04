from model_params import *

"""
Builds the connectivity matrix for the torus model
Generates a numpy file called conn_matrix_torus.npy containing:
    - the constructed matrix (for both excitatory and inhibitory neighbouring networks)
    - information on the number of excitatory and inhibitory networks
"""

def make_conn_matrix_torus(MODEL='RING-2D', custom_ring_params={}):
    '''
    LOADING PARAMETERS
    '''
    X, Z, Xn_exc, Xn_inh, exc_connected_neighbors, exc_decay_connect, inh_connected_neighbors,\
            inh_decay_connect, conduction_velocity = pseq_ring_params(MODEL, custom=custom_ring_params)
    
    # Takes care of the normalization of neighbouring connection
    Xn_inh_copy = Xn_inh.copy()
    Xn_exc_copy = Xn_exc.copy()
    Xn_inh_copy[Xn_inh_copy == -99] = 0
    Xn_exc_copy[Xn_exc_copy == -99] = 0
    total_number_neighbour_inh = np.count_nonzero(Xn_inh_copy)+1 
    total_number_neighbour_exc = np.count_nonzero(Xn_exc_copy)+1
    # +1 because otherwise the central pixel isn't counted

    '''
    NORMALIZATION
    '''
    connectivity_normalization_inh = (len(Xn_inh_copy[0])-1)/total_number_neighbour_inh
    connectivity_normalization_exc = (len(Xn_exc_copy[0])-1)/total_number_neighbour_exc
    # Comment to modify normalization factors:
    #   - Uncommented = 'augmented normalization'
    #   - Commented = 'base normalization'
    connectivity_normalization_inh = 0.8
    connectivity_normalization_exc = 0.5
    print('Normalization:')
    print('Exc =', connectivity_normalization_exc, '; Inh =', connectivity_normalization_inh)
    print('Exc neighbours:', total_number_neighbour_exc, 'Inh neighbours:', total_number_neighbour_inh)

    '''
    INITIALISATION
    Generates an array of lists of dictionaries:
        - Each element of the array is a vector that accounts for all the neighbours of this pixel
        - Each element of this vector is a dictionnary that has 4 arguments: 
        dist, weight, pos_x, and pos_z 
        --> M_conn_exc[1][9][5]['dist'] gives the distance of the 5th neighbour from 
        the pixel at z=1 and x=9
    '''
    M_conn_exc = [[[dict(dist=None,weight=None,pos_x=None,pos_z=None) \
                        for nb in range(total_number_neighbour_exc)] \
                        for z in range(len(Z))] \
                        for x in range(len(X))]

    M_conn_inh = [[[dict(dist=None,weight=None,pos_x=None,pos_z=None) \
                        for nb in range(total_number_neighbour_inh)] \
                        for z in range(len(Z))] \
                        for x in range(len(X))]

    '''
    CONSTRUCTION
    We construct the connectivity matrix for the torus model, i.e. with circular boundary
    conditions
    '''

    # Pixel loop
    for i_z in range(len(Z)): 
        for i_x in range(len(X)): 
            # Loop over neighbouring excitatory pixels
            i_row = 0
            neighbour_number = 0
            for row in Xn_exc:
                i_row = i_row + 1
                for i_xn in row: 
                    '''
                    A neighbour is valid if it is close enough (i_xn != -99 by construction 
                    of neighbour's matrix)
                    '''
                    if (i_xn != -99):
                        # Weight calculation
                        exc_weight = gaussian_connectivity(i_xn, 0., exc_decay_connect,connectivity_normalization_exc)                            
                        # Upper half of the neighbour's matrix
                        if i_row <= len(Xn_exc[0])//2+1:
                            centered_row_upper = -1*((len(Xn_exc[0])//2+1-i_row)%(len(Xn_exc[0])//2+1))
                            zpos_neighbour = i_z + centered_row_upper
                            xpos_neighbour = np.sign(i_xn) * np.round(np.sqrt(i_xn**2 - centered_row_upper**2))
                            # Folded boundary conditions for x and z
                            i_xC = int((i_x+xpos_neighbour)%(len(X)))
                            i_zC = int(zpos_neighbour%(len(Z)))
                        # Lower half of the neighbour's matrix
                        if i_row > len(Xn_exc[0])//2+1:
                            centered_row_lower = (i_row + len(Xn_exc[0])//2)%len(Xn_exc[0])
                            zpos_neighbour = i_z + centered_row_lower
                            xpos_neighbour = np.sign(i_xn) * np.round(np.sqrt(i_xn**2 - centered_row_lower**2))
                            # Folded boundary conditions for x and z
                            i_xC = int((i_x+xpos_neighbour)%(len(X)))
                            i_zC = int(zpos_neighbour%(len(Z)))

                        # Updating connectivity matrix
                        M_conn_exc[i_z][i_x][neighbour_number]['dist'] = i_xn
                        M_conn_exc[i_z][i_x][neighbour_number]['weight'] = exc_weight
                        M_conn_exc[i_z][i_x][neighbour_number]['pos_x'] = i_xC
                        M_conn_exc[i_z][i_x][neighbour_number]['pos_z'] = i_zC
                        
                        neighbour_number += 1
                        
            # Loop over neighbouring inhibitory pixels    
            i_row = 0
            neighbour_number = 0
            for row in Xn_inh:
                i_row = i_row + 1
                for i_xn in row: 
                    '''
                    A neighbour is valid if it is close enough (i_xn != -99 by construction 
                    of neighbour's matrix)
                    '''
                    if (i_xn != -99):
                        # Weight calculation
                        inh_weight = gaussian_connectivity(i_xn, 0., inh_decay_connect, connectivity_normalization_inh)
                        # Upper half of the neighbour's matrix
                        if i_row <= len(Xn_inh[0])//2+1:
                            centered_row_upper = -1*((len(Xn_inh[0])//2+1-i_row)%(len(Xn_inh[0])//2+1))
                            zpos_neighbour = i_z + centered_row_upper
                            xpos_neighbour = np.sign(i_xn) * np.round(np.sqrt(i_xn**2 - centered_row_upper**2))
                            # Folded boundary conditions for x and z
                            i_xC = int((i_x+xpos_neighbour)%(len(X)))
                            i_zC = int(zpos_neighbour%(len(Z)))
                        # Lower half of the neighbour's matrix
                        if i_row > len(Xn_inh[0])//2+1:
                            centered_row_lower = (i_row + len(Xn_inh[0])//2)%len(Xn_inh[0])
                            zpos_neighbour = i_z + centered_row_lower
                            xpos_neighbour = np.sign(i_xn) * np.round(np.sqrt(i_xn**2 - centered_row_lower**2))
                            # Folded boundary conditions for x and z
                            i_xC = int((i_x+xpos_neighbour)%(len(X)))
                            i_zC = int(zpos_neighbour%(len(Z)))
                        
                        # Updating connectivity matrix
                        M_conn_inh[i_z][i_x][neighbour_number]['dist'] = i_xn
                        M_conn_inh[i_z][i_x][neighbour_number]['weight'] = inh_weight
                        M_conn_inh[i_z][i_x][neighbour_number]['pos_x'] = i_xC
                        M_conn_inh[i_z][i_x][neighbour_number]['pos_z'] = i_zC
                        
                        neighbour_number += 1          

    '''
    SAVE
    ''' 
    file = 'conn_matrix_torus.npy'
    np.save('data/' + file, [M_conn_exc,M_conn_inh,total_number_neighbour_exc,total_number_neighbour_inh])
    print('\n -------> Results saved in', str('/data/' + file))
    return