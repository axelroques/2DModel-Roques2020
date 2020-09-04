import sys
sys.path.append('../')

"""
Neuron 'microscopic' properties

Output:	dictionary containg neuron parameters
"""

def get_neuron_params(NAME, name='', SI_units=True):
    if NAME=='FS-cell':
        params = {
            'name':name, 
            'N':1,
            'Gl':10., 
            'Cm':200.,
            'Trefrac':5,
            'El':-65.,
            'Vthre':-50., 
            'Vreset':-65., 
            'delta_v':0.5,
            'ampnoise':0.,
            'a':0., 
            'b': 0., 
            'tauw':1e9
            }
    elif NAME=='RS-cell':
        params = {
            'name':name, 
            'N':1,
            'Gl':10., 
            'Cm':200.,
            'Trefrac':5,
            'El':-65., 
            'Vthre':-50., 
            'Vreset':-65., 
            'delta_v':2.,
            'ampnoise':0.,
            'a':4., 
            'b':20., 
            'tauw':500.
            }
    else:
        print('====================================================')
        print('------------ CELL NOT RECOGNIZED !! ---------------')
        print('====================================================')

    if SI_units:
        # mV to V
        params['El'], params['Vthre'], params['Vreset'], params['delta_v'] =\
            1e-3*params['El'], 1e-3*params['Vthre'], 1e-3*params['Vreset'], 1e-3*params['delta_v']
        # ms to s
        params['Trefrac'], params['tauw'] = 1e-3*params['Trefrac'], 1e-3*params['tauw']
        # nS to S
        params['a'], params['Gl'] = 1e-9*params['a'], 1e-9*params['Gl']
        # pF to F and pA to A
        params['Cm'], params['b'] = 1e-12*params['Cm'], 1e-12*params['b']
        
    else:
        print('cell parameters --NOT-- in SI units')

    return params.copy()