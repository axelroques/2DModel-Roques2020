from synapses_connectivity import get_connectivity_and_synapses_matrix
from toolbox import TF_my_template, pseq_params
from scipy.integrate import odeint
from scipy.special import erf
from scipy.stats import norm
from numpy import meshgrid
from cell_library import *
from numpy import arange
import numpy as np
import random
import math


def reformat_syn_parameters(params, M):
  """
  Transfer values for the synapses parameters contained in M (built from synapses_connectivity.py) to 
  the params dictionary
  """
  params['Qe'], params['Te'], params['Ee'] = M[0,0]['Q'], M[0,0]['Tsyn'], M[0,0]['Erev']
  params['Qi'], params['Ti'], params['Ei'] = M[1,1]['Q'], M[1,1]['Tsyn'], M[1,1]['Erev']
  params['pconnec'] = M[0,0]['p_conn']
  params['Ntot'], params['gei'] = M[0,0]['Ntot'], M[0,0]['gei']



def load_transfer_functions(NRN1, NRN2, NTWK):
  """   
  Returns transfer functions for both RS and FS cell types
  """

  M = get_connectivity_and_synapses_matrix(NTWK)
    
  # NRN1
  params1 = get_neuron_params(NRN1, SI_units=True)
  reformat_syn_parameters(params1, M)
  try:               
      P1 = np.load('data/RS-cell_CONFIG1_fit.npy')       
      params1['P'] = P1

      def TF1(fe, fi):
          return TF_my_template(fe, fi, *pseq_params(params1))      
  except IOError:
      print('=======================================================')
      print('========  fit for NRN1 not available  =================')
      print('=======================================================')

  # NRN2
  params2 = get_neuron_params(NRN2)
  reformat_syn_parameters(params2, M)
  try:        
      P2 = np.load('data/FS-cell_CONFIG1_fit.npy')      
      params2['P'] = P2

      def TF2(fe, fi):
          return TF_my_template(fe, fi, *pseq_params(params2))
  except IOError:
      print('=======================================================')
      print('=====  fit for NRN2 not available  ====================')
      print('=======================================================')
    
  return TF1, TF2



def build_up_differential_operator_first_order(TF1, TF2, T=5e-3):
  """
  Solves mean field equation for the first order system
  Only used for the initialization of the model with the find_fixed_point_first_order function
  """
  def A0(V, exc_aff=0, inh_aff=0, pure_exc_aff=0):
      return 1./T*(TF1(V[0]+exc_aff+pure_exc_aff, V[1]+inh_aff)-V[0])
  
  def A1(V, exc_aff=0, inh_aff=0, pure_exc_aff=0):
      return 1./T*(TF2(V[0]+exc_aff, V[1]+inh_aff)-V[1])
  
  def Diff_OP(V, exc_aff=0, inh_aff=0, pure_exc_aff=0):
      return np.array([A0(V, exc_aff=exc_aff,inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),\
                       A1(V, exc_aff=exc_aff, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff)])
  return Diff_OP


  
def find_fixed_point_first_order(NRN1, NRN2, NTWK,\
                                 Ne=8000, Ni=2000, exc_aff=0.,\
                                 verbose=False):
  """
  Used to initialize the model with the external drive until a stationary state
  """
  TF1, TF2 = load_transfer_functions(NRN1, NRN2, NTWK)
    
  t = np.arange(2000)*1e-4 # Time vector
    
  # Solve equation (first order)
  def dX_dt_scalar(X, t=0):
      return build_up_differential_operator_first_order(TF1, TF2, T=5e-3)(X, exc_aff=exc_aff)
  
  X0 = [1, 10] # need inhibition stronger than excitation
  X = odeint(dX_dt_scalar, X0, t) 
  if verbose:
      print('first order prediction: ', X[-1])

  return X[-1][0], X[-1][1]