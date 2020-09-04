from matplotlib.ticker import PercentFormatter
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import scipy.special as sp_spec
from celluloid import Camera
import matplotlib.cm as cm
import matplotlib as mpl
import numpy as np


def heaviside(x):
    """
    Simple Heaviside function
    """
    return 0.5*(1+np.sign(x))

def pseq_params(params):
    """
    Returns 'microscopic' parameters and results from the fit (the Ps for the threshold function)
    """
    for key, dval in zip(['Ntot', 'pconnec', 'gei'], [1, 2., 0.5]):
        if not key in params.keys():
            params[key] = dval

    if 'P' in params.keys():
        P = params['P']
    else: # no correction
        P = [-45e-3]
        for i in range(1,11):
            P.append(0)
    return params['Qe'], params['Te'], params['Ee'], params['Qi'], params['Ti'], params['Ei'], params['Gl'], params['Cm'], params['El'], params['Ntot'], params['pconnec'], params['gei'], P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10]


def TF_my_template(fe, fi, Qe, Te, Ee, Qi, Ti, Ei, Gl, Cm, El, Ntot, pconnec, gei, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10):
    """
    Numeric implementation of the transfer function
    """
    # here TOTAL (sum over synapses) excitatory and inhibitory input
    muV, sV, muGn, TvN = get_fluct_regime_vars(fe, fi, Qe, Te, Ee, Qi, Ti, Ei, Gl, Cm, El, Ntot, pconnec, gei, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10)
    Vthre = threshold_func(muV, sV, TvN, muGn, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10)
    Fout_th = erfc_func(muV, sV, TvN, Vthre, Gl, Cm)
    return Fout_th


def get_fluct_regime_vars(Fe, Fi, Qe, Te, Ee, Qi, Ti, Ei, Gl, Cm, El, Ntot, pconnec, gei, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10):
    """
    Computes values needed for the transfer function
    """
    # here TOTAL (sum over synapses) excitatory and inhibitory input
    fe = Fe*(1.-gei)*pconnec*Ntot
    fi = Fi*gei*pconnec*Ntot
    
    muGe, muGi = Qe*Te*fe, Qi*Ti*fi
    muG = Gl+muGe+muGi
    muV = (muGe*Ee+muGi*Ei+Gl*El)/muG
    muGn, Tm = muG/Gl, Cm/muG
    
    Ue, Ui = Qe/muG*(Ee-muV), Qi/muG*(Ei-muV)

    sV = np.sqrt(\
                 fe*(Ue*Te)**2/2./(Te+Tm)+\
                 fi*(Qi*Ui)**2/2./(Ti+Tm))

    fe, fi = fe+1e-9, fi+1e-9 # just to insure a non zero division, 
    Tv = ( fe*(Ue*Te)**2 + fi*(Qi*Ui)**2 ) /( fe*(Ue*Te)**2/(Te+Tm) + fi*(Qi*Ui)**2/(Ti+Tm) )
    TvN = Tv*Gl/Cm

    return muV, sV+1e-12, muGn, TvN


def erfc_func(muV, sV, TvN, Vthre, Gl, Cm):
    """
    Numeric implementation of the error function
    """
    return .5/TvN*Gl/Cm*(sp_spec.erfc((Vthre-muV)/np.sqrt(2)/sV))


def threshold_func(muV, sV, TvN, muGn, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10):
    """
    Numeric implementation of the threshold function
    Here are the previous mysterious comments from Yann Zerlaut, if they can be of any help:
        setting by default to True the square
        because when use by external modules, coeff[5:]=np.zeros(3)
        in the case of a linear threshold
    """
    
    muV0, DmuV0 = -60e-3,10e-3
    sV0, DsV0 =4e-3, 6e-3
    TvN0, DTvN0 = 0.5, 1.
    
    return P0+P1*(muV-muV0)/DmuV0+\
        P2*(sV-sV0)/DsV0+P3*(TvN-TvN0)/DTvN0+\
        1*P4*np.log(muGn)+P5*((muV-muV0)/DmuV0)**2+\
        P6*((sV-sV0)/DsV0)**2+P7*((TvN-TvN0)/DTvN0)**2+\
        P8*(muV-muV0)/DmuV0*(sV-sV0)/DsV0+\
        P9*(muV-muV0)/DmuV0*(TvN-TvN0)/DTvN0+\
        P10*(sV-sV0)/DsV0*(TvN-TvN0)/DTvN0


"""
What follows are some function that I used to plot my figures
A description for each of them can be found in the Notebook
"""
def xz_plot(Feaff, Fe, Fi, muVn, t, x, z):

    fig, axs = plt.subplots(2, 2, figsize=(6,6))

    axs[0, 0].plot(Feaff[:t,x,z])
    axs[0, 0].set_title('$\\nu_e^{aff}$ for pixel (' + str(x) + ', ' + str(z) + ')')
    axs[0, 0].set(xlabel='t (ms)', ylabel='$\\nu_e^{aff}(x, t)$')

    axs[0, 1].plot(Fe[:t,x,z], 'tab:orange')
    axs[0, 1].set_title('$\\nu_e$ for pixel (' + str(x) + ', ' + str(z) + ')')
    axs[0, 1].set(xlabel='t (ms)', ylabel='$\\nu_e(x, t)$')

    axs[1, 0].plot(Fi[:t,x,z], 'tab:green')
    axs[1, 0].set_title('$\\nu_i$ for pixel (' + str(x) + ', ' + str(z) + ')')
    axs[1, 0].set(xlabel='t (ms)', ylabel='$\\nu_i(x, t)$')

    axs[1, 1].plot(muVn[:t,x,z], 'tab:red')
    axs[1, 1].set_title('$\\mu_V^{N}$ for pixel (' + str(x) + ', ' + str(z) + ')')
    axs[1, 1].set(xlabel='t (ms)', ylabel='$\\mu_V^{N}(x, t)$')
 
    plt.setp(axs, xticks=np.linspace(0, t, 5), xticklabels=np.linspace(0, t/2, 5, dtype=int))
    fig.tight_layout()
    plt.show()
    return


def xz_combined(Feaff, Fe, Fi, muVn, t, x, z):
    fig, ax = plt.subplots()
    plt.title('$\\nu_e^{aff}$, $\\nu_e$, $\\nu_i$ and $\\mu_V^{N}$ for pixel (' + str(x) + ', ' + str(z) + ')')
    axes = [ax, ax.twinx(), ax.twinx(), ax.twinx()]

    fig.subplots_adjust(right=0.75)
    axes[-2].spines['right'].set_position(('axes', 1.3))
    axes[-2].set_frame_on(True)
    axes[-2].patch.set_visible(False)
    axes[-1].spines['right'].set_position(('axes', 1.6))
    axes[-1].set_frame_on(True)
    axes[-1].patch.set_visible(False)

    axes[0].plot(Feaff[:t,x,z], color='Blue')
    axes[0].set_ylabel('$\\nu_e^{aff}(x, t)$', color='Blue')
    axes[0].set_xlabel('t (ms)')
    axes[0].tick_params(axis='y', colors='Blue')

    axes[1].plot(Fe[:t,x,z], color='Orange')
    axes[1].set_ylabel('$\\nu_e(x, t)$', color='Orange')
    axes[1].set_xlabel('t (ms)')
    axes[1].tick_params(axis='y', colors='Orange')

    axes[2].plot(Fi[:t,x,z], color='Green')
    axes[2].set_ylabel('$\\nu_i(x, t)$', color='Green')
    axes[2].set_xlabel('t (ms)')
    axes[2].tick_params(axis='y', colors='Green')

    axes[3].plot(muVn[:t,x,z], color='Red')
    axes[3].set_ylabel('$\\mu_V^{N}(x, t)$', color='Red')
    axes[3].set_xlabel('t (ms)')
    axes[3].tick_params(axis='y', colors='Red')

    plt.setp(ax, xticks=np.linspace(0, t, 5), xticklabels=np.linspace(0, t/2, 5, dtype=int))
    fig.tight_layout()
    plt.show()
    return


def xz_movie(Feaff, Fe, Fi, muVn, X, Z, t, frames, title, save=False):

    def colorbar_format(x, pos):
        a = '{:.3f}'.format(x)
        return format(a)

    fig, axs = plt.subplots(2, 2, figsize=(8,8))
    axs[0, 0].set_title('$\\nu_e^{aff}$')
    axs[0, 0].set(xlabel='X (mm)', ylabel='Z (mm)')
    axs[0, 1].set_title('$\\nu_e$')
    axs[0, 1].set(xlabel='X (mm)', ylabel='Z (mm)')    
    axs[1, 0].set_title('$\\nu_i$')
    axs[1, 0].set(xlabel='X (mm)', ylabel='Z (mm)')    
    axs[1, 1].set_title('$\\mu_V^{N}$')
    axs[1, 1].set(xlabel='X (mm)', ylabel='Z (mm)')

    camera = Camera(fig)

    for i in range(0, t, frames):
        cbar0 = axs[0, 0].contourf(X, Z, Feaff[i,:,:].T,
                    np.linspace(Feaff.min(), Feaff.max(), 20),
                    cmap=mpl.cm.viridis)
        cbar1 = axs[0, 1].contourf(X, Z, Fe[i,:,:].T,
                    np.linspace(Fe.min(), Fe.max(), 20),
                    cmap=mpl.cm.viridis)
        cbar2 = axs[1, 0].contourf(X, Z, Fi[i,:,:].T,
                    np.linspace(Fi.min(), Fi.max(), 20),
                    cmap=mpl.cm.viridis)
        cbar3 = axs[1, 1].contourf(X, Z, muVn[i,:,:].T,
                    np.linspace(muVn.min(), muVn.max(), 20),
                    cmap=mpl.cm.viridis)
        camera.snap()

    anim = camera.animate()

    fig.colorbar(cbar0, ax=axs[0, 0], format=ticker.FuncFormatter(colorbar_format))
    fig.colorbar(cbar1, ax=axs[0, 1], format=ticker.FuncFormatter(colorbar_format))
    fig.colorbar(cbar2, ax=axs[1, 0], format=ticker.FuncFormatter(colorbar_format))
    fig.colorbar(cbar3, ax=axs[1, 1], format=ticker.FuncFormatter(colorbar_format))

    fig.tight_layout()
    plt.show()

    if save:
        anim.save('figures/' + title + '.mp4')
    return


def show_rand_conn(random_conn_params):

    fig = plt.figure(figsize=(8.5,8.5))
    ax = fig.add_subplot(111)
    colors = cm.rainbow(np.linspace(0, 1, random_conn_params['nb_random_conn']))

    for x_pix, z_pix, x_neigh, z_neigh, c in zip(\
            random_conn_params['x_pixel'], random_conn_params['z_pixel'],\
            random_conn_params['x_neigh'], random_conn_params['z_neigh'],colors):
        ax.scatter(x_pix, z_pix, color=c)
        ax.scatter(x_neigh, z_neigh, color=c)
        ax.plot([x_pix, x_neigh], [z_pix, z_neigh], color=c, linewidth=1)
        
    plt.title('Random connectivity')
    ax.set_xlabel('X (pixel)')
    ax.set_ylabel('Z (pixel)')
    plt.show()
    return