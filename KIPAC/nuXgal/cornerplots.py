import getdist
import h5py
from getdist import plots, MCSamples
import sys
import matplotlib
plt.style.use('../../scripts/jupyterhub.style')
plt.rc('text.latex', preamble=r'\usepackage{../../scripts/apjfonts}')

year = int(sys.argv[1])
f    = h5py.File('../../corner_WISE_%d.h5'%year, 'r')  

f['mcmc'].keys() 
f1 = f['mcmc']['chain'][100:,:,0].flatten() # 100 is the burn-in
f2 = f['mcmc']['chain'][100:,:,1].flatten()
f3 = f['mcmc']['chain'][100:,:,2].flatten()

if year==3:
    samp  = MCSamples(samples=np.c_[f1,f2,f3],names = ['f1','f2','f3'],labels = [r'$f_{\rm astro,1}$',r'$f_{\rm astro,2}$',r'$f_{\rm astro,3}$'],label="${\\rm Y3}\ 3\\times2\ {\\rm alt\ sample}$",settings={'mult_bias_correction_order':0,'smooth_scale_2D':0.2, 'smooth_scale_1D':0.2})

    g = plots.getSubplotPlotter(chain_dir='./', width_inch=4.5)

    g.settings.axes_fontsize   = 13
    g.settings.lab_fontsize    = 15
    g.settings.legend_fontsize = 13

    g.triangle_plot([samp], filled=[True,True], colors=['steelblue','teal'],alphas=[1.0,0.7],ls=['-','-'],contour_colors=['gray','teal'],contour_lws=[1.2,1.2]);
    plt.grid(False)
    for i in range(0,6):
        g.fig.get_axes()[i].grid(False)

    #--------set ylim -------
    g.fig.get_axes()[3].set_ylim(0,0.1)
    g.fig.get_axes()[4].set_ylim(0,0.9)
    g.fig.get_axes()[5].set_ylim(0,0.9)

    #--------set xlim -------
    g.fig.get_axes()[2].set_xlim(0,1)   #diag

    g.fig.get_axes()[0].set_xlim(0,0.038)
    g.fig.get_axes()[3].set_xlim(0,0.038)
    g.fig.get_axes()[4].set_xlim(0,0.038)

    g.fig.get_axes()[1].set_xlim(0,0.1)
    g.fig.get_axes()[5].set_xlim(0,0.1)

    #--------set xticks --------
    g.fig.get_axes()[2].tick_params(axis='x', rotation=25)

    g.fig.get_axes()[3].collections[1].set_alpha(0.7)
    g.fig.get_axes()[3].collections[2].set_linewidth(1.2)
    g.fig.get_axes()[3].set_yticks([0.01,0.04,0.07])
    #g.fig.get_axes()[3].tick_params(axis='y', rotation=25)

    g.fig.get_axes()[4].collections[1].set_alpha(0.7)
    g.fig.get_axes()[4].collections[2].set_linewidth(1.2)
    g.fig.get_axes()[4].tick_params(axis='x', rotation=25)
    g.fig.get_axes()[4].set_xticks([0.01,0.03])
    g.fig.get_axes()[3].set_xticks([0.01,0.03])
    g.fig.get_axes()[4].set_xticks([0.01,0.03])
    g.fig.get_axes()[0].set_xticks([0.01,0.03])
    g.fig.get_axes()[4].set_yticks([0.1,0.40,0.7])
    g.fig.get_axes()[5].set_yticks([0.1,0.40,0.7])
    g.fig.get_axes()[4].set_ylabel(ylabel=r'$f_{\rm astro,3}$',labelpad=10);

    #g.fig.get_axes()[4].tick_params(axis='y', rotation=25)

    g.fig.get_axes()[5].collections[1].set_alpha(0.7)
    g.fig.get_axes()[5].collections[2].set_linewidth(1.2)
    g.fig.get_axes()[5].tick_params(axis='x', rotation=25)
    g.fig.get_axes()[5].set_xticks([0.02,0.05,0.08])
    g.fig.get_axes()[1].set_xticks([0.02,0.05,0.08])

    g.fig.get_axes()[2].set_xticks([0.2,0.5,0.8])
    g.fig.get_axes()[2].set_xticks([0.2,0.5,0.8])
    g.fig.get_axes()[2].set_xlabel(xlabel=r'$f_{\rm astro,3}$',labelpad=19);

    plt.savefig('corner_WISE_3.pdf',dpi=300)

if year==10:

    g = plots.getSubplotPlotter(chain_dir='./', width_inch=4.5)

    g.settings.axes_fontsize   = 13
    g.settings.lab_fontsize    = 15
    g.settings.legend_fontsize = 13

    g.triangle_plot([samp], filled=[True,True], colors=['steelblue','teal'],alphas=[1.0,0.7],ls=['-','-'],contour_colors=['gray','teal'],contour_lws=[1.2,1.2]);
    plt.grid(False)
    for i in range(0,6):
        g.fig.get_axes()[i].grid(False)

    #--------set ylim -------
    g.fig.get_axes()[3].set_ylim(0,0.1)
    g.fig.get_axes()[4].set_ylim(0,0.9)
    g.fig.get_axes()[5].set_ylim(0,0.9)

    #--------set xlim -------
    g.fig.get_axes()[2].set_xlim(0,1)   #diag

    g.fig.get_axes()[0].set_xlim(0,0.038)
    g.fig.get_axes()[3].set_xlim(0,0.038)
    g.fig.get_axes()[4].set_xlim(0,0.038)

    g.fig.get_axes()[1].set_xlim(0,0.1)
    g.fig.get_axes()[5].set_xlim(0,0.1)

    #--------set xticks --------
    g.fig.get_axes()[2].tick_params(axis='x', rotation=25)

    g.fig.get_axes()[3].collections[1].set_alpha(0.7)
    g.fig.get_axes()[3].collections[2].set_linewidth(1.2)
    g.fig.get_axes()[3].set_yticks([0.01,0.04,0.07])
    #g.fig.get_axes()[3].tick_params(axis='y', rotation=25)

    g.fig.get_axes()[4].collections[1].set_alpha(0.7)
    g.fig.get_axes()[4].collections[2].set_linewidth(1.2)
    g.fig.get_axes()[4].tick_params(axis='x', rotation=25)
    g.fig.get_axes()[4].set_xticks([0.01,0.03])
    g.fig.get_axes()[3].set_xticks([0.01,0.03])
    g.fig.get_axes()[4].set_xticks([0.01,0.03])
    g.fig.get_axes()[0].set_xticks([0.01,0.03])
    g.fig.get_axes()[4].set_yticks([0.1,0.40,0.7])
    g.fig.get_axes()[5].set_yticks([0.1,0.40,0.7])
    g.fig.get_axes()[4].set_ylabel(ylabel=r'$f_{\rm astro,3}$',labelpad=10);

    #g.fig.get_axes()[4].tick_params(axis='y', rotation=25)

    g.fig.get_axes()[5].collections[1].set_alpha(0.7)
    g.fig.get_axes()[5].collections[2].set_linewidth(1.2)
    g.fig.get_axes()[5].tick_params(axis='x', rotation=25)
    g.fig.get_axes()[5].set_xticks([0.02,0.05,0.08])
    g.fig.get_axes()[1].set_xticks([0.02,0.05,0.08])

    g.fig.get_axes()[2].set_xticks([0.2,0.5,0.8])
    g.fig.get_axes()[2].set_xticks([0.2,0.5,0.8])
    g.fig.get_axes()[2].set_xlabel(xlabel=r'$f_{\rm astro,3}$',labelpad=19);

    #truthvalues
    g.fig.get_axes()[3].axvline(0.00221405,ls='--',dashes=(4,2))
    g.fig.get_axes()[3].axhline(0.01216614,ls='--',dashes=(4,2))
    g.fig.get_axes()[4].axvline(0.00221405,ls='--',dashes=(4,2))
    g.fig.get_axes()[4].axhline(0.70877666,ls='--',dashes=(4,2))
    g.fig.get_axes()[5].axvline(0.01216614,ls='--',dashes=(4,2))
    g.fig.get_axes()[5].axhline(0.70877666,ls='--',dashes=(4,2))

    plt.savefig('corner_WISE_10.pdf',dpi=300)