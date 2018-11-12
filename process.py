#! /usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('Agg')
import os
import scipy.optimize as opt
import matplotlib.pyplot as plt
import time
import sys
import scipy.special as spec

def model1(x, b):
    """Gaussian"""
    return np.exp(-b*x**2)

def model_quasigaussian(x, a, b):
    """Quasi gaussian model"""
    return np.exp(-a*np.abs(x)**b)

def model_deltaz(x, a):
    """Quadratic"""
    return a*x**2

def model_zmean_exp(x, a, b):
    return a*np.exp(-x*b)

def model_zmean(x, a, b):
    return a*np.exp(-b*x)

def model2d(x, a):
    (k, deltaz) = x
    return np.exp(-a * deltaz**2 * k**2)


def model3d(x, a, b):
    (k, zmean, deltaz) = x
    return np.ravel(np.exp(-a * k**2 * deltaz**2 * np.exp(-b * zmean)))

def model3d2(x, a):
    (k, zmean, deltaz) = x
    return np.ravel(np.exp(-a * k**2 *deltaz**2 / zmean**2))

def model3d3(x, a, b):
    (k, zmean, deltaz) = x
    return np.ravel(np.exp(-a * k**2 *deltaz**2 / (1 + zmean)**b))

def model3d4(x, a, b):
    (k, zmean, deltaz) = x
    return np.ravel(np.exp(-a * k**2 *deltaz**2 / (1 + b*zmean)))

def model3d5(x, a, b):
    (k, zmean, deltaz) = x
    return np.ravel(np.exp(-a * k**2 *deltaz**2 / (b + zmean)))

def model3d6(x, a, b):
    (k, zmean, deltaz) = x
    return np.ravel(np.exp(-a * k**2 *deltaz**2 / ((1 + zmean + deltaz/2.) * (1 + zmean - deltaz/2.))**b))

models = [model3d, model3d2, model3d3, model3d4, model3d5, model3d6]


# parameters
prefix = "daint_2048"

redshifts = np.array([60, 30, 10, 5,3, 2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.3, 2.2, 2.1, 2, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0])

ngrid = 2048
Lbox = 1024
knyq = np.pi * ngrid/Lbox
bins = 1024

def process_data(prefix):

    if not os.path.exists("%sprocess"%prefix):
        os.mkdir("%sprocess"%prefix)

    try:
        corrcoeffmod = pd.read_pickle("%sprocess/saved_data.pkl" % prefix)

    except FileNotFoundError:

        print("Processed file not found, processing now...")

        # reading all the data
        for n1 in range(len(redshifts)):
            for n2 in range(n1 + 1):
                temp = pd.read_csv('%s/lcdm_pk%03d_phi_%03d.dat' % (prefix, n1, n2), sep = '\s+', header=None, names = ['k', 'Pk', 'sigma k', 'sigma Pk', 'count'], index_col=None, comment='#')
                df_cross['%.1f, %.1f, pk' % (redshifts[n1], redshifts[n2])] = temp['Pk'].values
                df_cross['%.1f, %.1f, sigmapk' % (redshifts[n1], redshifts[n2])] = temp['sigma Pk'].values
            df_pk['%.1f, pk' % (redshifts[n1])] = temp['Pk'].values
            df_pk['%.1f, k' % (redshifts[n1])] = temp['k'].values
            df_pk['%.1f, sigmapk' % (redshifts[n1])] = temp['sigma Pk'].values

        avg_cross = pd.DataFrame(index=range(bins))
        avg_pk = pd.DataFrame(index=range(bins))

        # averaging of the data (including the errors)
        for n1 in range(len(redshifts)):
            for n2 in range(n1 + 1):
                avg_cross['%.1f, %.1f, pk' % (redshifts[n1], redshifts[n2])] = df_cross['%.1f, %.1f, pk' %(redshifts[n1], redshifts[n2])]
                avg_cross['%.1f, %.1f, sigmapk' % (redshifts[n1], redshifts[n2])] = df_cross['%.1f, %.1f, sigmapk' %(redshifts[n1], redshifts[n2])]
            avg_pk['%.1f, pk' % (redshifts[n1])] = df_pk['%.1f, pk' %(redshifts[n1])]
            avg_pk['%.1f, sigmapk' % (redshifts[n1])] = df_pk['%.1f, sigmapk' %(redshifts[n1])]

        # index for a single correlation coefficient must be the wavenumber k
        corrcoeff = pd.DataFrame(index=df_pk['0.0, k'])

        # setting the correlation coefficient
        for i in range(len(redshifts)):
            for j in range(i + 1):
                corrcoeff['%.1f, %.1f, c' % (float(redshifts[i]), float(redshifts[j]))] = \
                    avg_cross['%.1f, %.1f, pk' % (redshifts[i], redshifts[j])].values\
                    /\
                    np.sqrt(\
                        avg_pk['%.1f, pk' % (redshifts[i])].values * avg_pk['%.1f, pk' % (redshifts[j])].values\
                    )
                corrcoeff['%.1f, %.1f, sigmac' % (float(redshifts[i]), float(redshifts[j]))] = \
                    np.sqrt(\
                    (avg_cross['%.1f, %.1f, sigmapk' %(float(redshifts[i]), float(redshifts[j]))].values/avg_cross['%.1f, %.1f, pk'%(float(redshifts[i]), float(redshifts[j]))].values)**2 +\
                    (avg_pk['%.1f, sigmapk'%float(redshifts[i])].values/avg_pk['%.1f, pk'%float(redshifts[i])].values/2.)**2 +\
                    (avg_pk['%.1f, sigmapk'%float(redshifts[j])].values/avg_pk['%.1f, pk'%float(redshifts[j])].values/2.)**2)\
                    *\
                    np.abs(corrcoeff['%.1f, %.1f, c' % (float(redshifts[i]), float(redshifts[j]))])
                corrcoeff['%.1f, %.1f, c' % (float(redshifts[j]), float(redshifts[i]))] = corrcoeff['%.1f, %.1f, c' % (float(redshifts[i]), float(redshifts[j]))]
                corrcoeff['%.1f, %.1f, sigmac' % (float(redshifts[j]), float(redshifts[i]))] = corrcoeff['%.1f, %.1f, sigmac' % (float(redshifts[i]), float(redshifts[j]))]

        ind = len(corrcoeff.index)*len(corrcoeff.columns)
        corrcoeffmod = pd.DataFrame(index = range(ind))

        ccc = 0

        # putting everything in one giant dataframe
        for k in range(len(corrcoeff.index)):
            for i in range(len(redshifts)):
                for j in range(len(redshifts)):
                    corrcoeffmod.at[ccc, 'k'] = corrcoeff.index[k]
                    corrcoeffmod.at[ccc, 'deltaz'] = np.abs(redshifts[i] - redshifts[j])
                    corrcoeffmod.at[ccc, 'z1'] = redshifts[i]
                    corrcoeffmod.at[ccc, 'z2'] = redshifts[j]
                    corrcoeffmod.at[ccc, 'zmean'] = (redshifts[i] + redshifts[j])/2
                    corrcoeffmod.at[ccc, 'c'] = corrcoeff['%.1f, %.1f, c' % (float(redshifts[i]), float(redshifts[j]))].iat[k]
                    corrcoeffmod.at[ccc, 'sigmac'] = corrcoeff['%.1f, %.1f, sigmac' % (float(redshifts[i]), float(redshifts[j]))].iat[k]
                    ccc += 1

        corrcoeffmod.to_pickle('%sprocess/saved_data.pkl' % prefix)

    print("Processing done")

    return corrcoeffmod


# selects the redshift interval
def select_interval(corrcoeffmod, zmax=100, kmax=100, cmin = -1e10, dropna=False, deltaz_min = 0.0):
    if dropna:
        corrcoeffmod = corrcoeffmod.loc[corrcoeffmod['deltaz'] != 0].dropna()
    corrcoeffmod = corrcoeffmod.loc[(corrcoeffmod['z1'] <= zmax) & (corrcoeffmod['zmean'] > 0) & (corrcoeffmod['k'].values <= kmax) & (corrcoeffmod['z2'] <= zmax) & (corrcoeffmod['c'] >= cmin) & (corrcoeffmod['deltaz'] >=deltaz_min)]
    return corrcoeffmod


# plots the stuff from the UETC paper
def plot_uetc_paper(corrcoeffmod, prefix):
    plt.ylim(ymin = 1e-6, ymax = 1e1)
    plt.xlim(xmin = 1e-2, xmax = 1e1)
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('1 - corr. coeff.')
    for z in [0.2, 0.5, 1.5, 2.0]:
        temp = corrcoeffmod.loc[(round(corrcoeffmod['z1'], 2) == 1.0) & (round(corrcoeffmod['z2'], 2) == z)]
        plt.loglog(temp['k'].values, 1 - temp['c'].values, label="z2 = %.2f"%z)
    plt.title("z1 = 1.0")
    plt.legend()
    plt.grid()
    plt.savefig("%sprocess/uetc_comp_x_k_y_c.jpg" % (prefix), dpi = 300)
    plt.close()

    plt.xlim(xmin = 0, xmax = 3)
    plt.ylim(ymin = 1e-5, ymax = 1e2)
    plt.xlabel('z2')
    plt.ylabel('1 - corr. coeff.')
    for k in [0.5, 1.0, 5.0, 10]:
        temp = corrcoeffmod.loc[(round(corrcoeffmod['k'], 2) == k) & (round(corrcoeffmod['z1'], 2) == 1.0)]
        plt.semilogy(temp['z2'].values, 1 - temp['c'].values, label="k = %.2f" % k)
    plt.title("z1 = 1.0")
    plt.legend()
    plt.grid()
    plt.savefig("%sprocess/uetc_comp_x_z_y_c.jpg" % (prefix), dpi = 300)
    plt.close()


# fitting the data
def fit_data(corrcoeffmod, models):
    df = pd.DataFrame(index=range(len(models)), columns = ["par", "cov", "sigma", "model", "chi2"])
    for i in range(len(models)):
        params, cov = opt.curve_fit(\
            models[i],\
            (corrcoeffmod['k'].values, corrcoeffmod['zmean'].values, corrcoeffmod['deltaz'].values),\
            corrcoeffmod['c'].values,\
            sigma = corrcoeffmod['sigmac'].values,\
            maxfev = 100000\
        )
        df.at[i, "par"] = params
        df.at[i, "cov"] = cov
        df.at[i, "sigma"] = np.sqrt(np.diag(cov))
        df.at[i, "model"] = models[i].__name__
        df.at[i, "chi2"] = sum((corrcoeffmod['c'].values - models[i]((corrcoeffmod['k'].values, corrcoeffmod['zmean'].values, corrcoeffmod['deltaz'].values), *params)) / corrcoeffmod['sigmac'].values) ** 2/(len(corrcoeffmod.index) - len(params))

    return df


# making plots for fixed zmean and deltaz as well as the model
def plot_zmean_deltaz(corrcoeffmod, models, params):
    for deltaz in np.unique(corrcoeffmod['deltaz'].values):
        temp_outer = corrcoeffmod.loc[corrcoeffmod['deltaz'] == deltaz]
        for zmean in np.unique(temp_outer['zmean'].values):
            temp_inner = temp_outer.loc[temp_outer['zmean'] == zmean]
            plt.ylim(ymin = 0.0, ymax = 1.2)
            plt.xlabel('k [h/Mpc]')
            plt.ylabel('corr. coeff.')
            plt.errorbar(temp_inner['k'].values, temp_inner['c'].values, yerr = temp_inner['sigmac'].values, fmt = 'ko', markersize = 2)
            for i in range(len(models)):
                plt.plot(temp_inner['k'].values, models[i]((temp_inner['k'].values, zmean, deltaz), *(params.loc[params["model"] == models[i].__name__]["par"].values[0])), label = '%s' % models[i].__name__)
            plt.legend()
            plt.grid()
            plt.xscale('log')
            plt.show()
            #plt.savefig("%sprocess/plot_deltaz_%.1f_zmean_%.2f.pdf" % (prefix, deltaz, zmean), dpi = 300)
            #plt.close()

            plt.xlabel('k [h/Mpc]')
            plt.ylabel('residual')
            for i in range(len(models)):
                plt.plot(temp_inner['k'].values, temp_inner['c'].values - models[i]((temp_inner['k'].values, zmean, deltaz), *(params.loc[params["model"] == models[i].__name__]["par"].values[0])), 'o', markersize = 2, label = '%s' % (models[i].__name__))
            plt.legend()
            plt.grid()
            plt.xscale('log')
            plt.show()
            #plt.savefig("%sprocess/plot_deltaz_%.1f_zmean_%.2f_res.pdf" % (prefix, deltaz, zmean), dpi = 300)
            #plt.close()

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# fit correlation coefficient as a function of wavenumber k for FIXED zmean and deltaz
def plot_zmean_deltaz_fit(prefix, df, model, name="gaussian"):
    for zmean in np.unique(df.zmean.values):
        # has fixed zmean
        temp = df.loc[df.zmean == zmean]
        for deltaz in np.unique(temp.deltaz.values):
            # has fixed zmean AND deltaz
            temp_inner = temp.loc[temp.deltaz == deltaz]
            par, cov = opt.curve_fit(\
                model,\
                (temp_inner.k.values),\
                temp_inner.c.values,\
                sigma = temp_inner.sigmac.values,\
                maxfev = 100000\
            )
            plt.ylim(ymin = 0.0, ymax = 1.2)
            plt.xlabel('k [h/Mpc]')
            plt.ylabel('corr. coeff.')
            plt.errorbar(temp_inner['k'].values, temp_inner['c'].values, yerr = temp_inner['sigmac'].values, fmt = 'ko', markersize = 2)
            plt.plot(temp_inner['k'].values, model((temp_inner['k'].values), *par), label = name)
            plt.title("chi2/dof = %f" % (sum((temp_inner.c.values - model((temp_inner.k.values), *par)) / temp_inner.sigmac.values) ** 2/(len(temp_inner.index) - len(par))))
            plt.legend()
            plt.grid()
            plt.xscale('log')
            #plt.show()
            plt.savefig("%sprocess/%s_plot_deltaz_%.1f_zmean_%.2f.pdf" % (prefix, name, deltaz, zmean), dpi = 300)
            plt.close()



df = process_data(prefix).dropna()
df = select_interval(df, zmax=3.0, cmin = 0.05, dropna=True, kmax = knyq/1.1)

plot_uetc_paper(df, prefix)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(fit_data(df, models))

plot_zmean_deltaz_fit(prefix, df, model_quasigaussian, "quasigaussian")
sys.exit()
plot_zmean_deltaz(df, models, fit_data(df, models))
