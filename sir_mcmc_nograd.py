"""

https://stackoverflow.com/questions/41109292/solving-odes-in-pymc3


findings: final Ro values very dependent on when we start the simulation.
it is not yet clear what makes most sense: fixing start datea and tuning start infection rate,
or setting start infection rate and tuning the start date. the latter requires a change in the code
for the mc sampling from posteriors, because right now this assumes that the timeaxis is alsways the same.
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from scipy import stats
import datetime
import pandas as pd
import seaborn as sns
from tqdm import trange, tqdm
import pymc3 as pm
import theano
import theano.tensor as tt


reference_date = pd.to_datetime('20200101') # this reference date is only for the definition of the
# date in the model. it is NOT the start date of the model. in the model we use integers as dates, which descrive
# the day relative to the reference date (can also be negative)


fixed_params = dict(
    L=365,  # length run (days)
    tstep = 0.5, # days, must by <=1 and 1/n
    t_start_qua=0.,  # time quarantine starts (days)
    t_stop_qua=0.,  # time quarantine ends (days)
    # Initial conditions. They are more crucial than one might think, especially when considering the start date!
  #  I_init = 1e-4,
    Ro_qua=2.5,  # Ro during quarantine
    tau = 14,
   # d=0.01,
    date_start = 30,
)


percs = [95, 90, 75, 50, 25, 10, 5]
linestyle_percs = [':','--','-.','-','-.','--',':']
N_mc_post = 10000 # number of sims made from posterior distribution
target_var = 'D'  # which data to fit on. 'D' for deaths, 'I' for infected
error = 'L2'  # L1 | L2

country = 'SWE'
plot_single_sims = False


def num_to_date(num_dates):
    return reference_date + pd.to_timedelta(num_dates, unit='d')


def SIR(x, t, Ro, tau, d, t_start_qua, t_stop_qua, Ro_qua):
    """
        ODE of the model
    """
    if (t >= t_start_qua / tau) & (t <= t_stop_qua / tau):
        Ro_t = Ro_qua
    else:
        Ro_t = Ro
    S, I, R, P = x
    xdot = np.array([-Ro_t * S * I, Ro_t * S * I - I, (1. - d) * I, d * I])
    return xdot


def solve(L,tstep, I_init, Ro, tau, d, t_start_qua, t_stop_qua, Ro_qua, date_start):
    """solve the model with a fixed set of parameters"""
    t = np.arange(0, L, tstep) / tau
    x0 = [1-I_init, I_init, 0, 0]
    x = odeint(SIR, x0, t, args=(Ro, tau, d, t_start_qua, t_stop_qua, Ro_qua))
    # output
    # odeint return len(t) timesteps, but including the initial condition.
    # therefore -1 in the next line
    dates = np.arange(date_start, date_start + L, tstep)
    return x, dates



def load_data(country):
    """
    load your data here. it must be a pandas dataframe
    with daily values, and days from reference date as index (integer)
    curently : data https://opendata.ecdc.europa.eu/covid19/casedistribution/csv

    """
    ifile='data/opendata_europe.csv'
    data = pd.read_csv(ifile)
    # select country
    data = data[data['countryterritoryCode']==country]
    # format dates
    dates = [pd.to_datetime(f'{e.year}-{e.month}-{e.day}') for _,e in data.iterrows()]
    data['date'] =pd.DatetimeIndex(dates)
    # revers the order (from first to last date_
    data = data[::-1]
    data = data.reset_index()
    # retain only necessary columns
    data = data[['date','cases', 'deaths']]
    data = data.set_index('date')
    # the data is the rate (per day). conver to absolute cases
    data = data.cumsum()
    data = data.rename(columns={'deaths':'D', 'cases':'I'})
    # convert dates to integers
    data.index = (data.index - reference_date).days
    return data


def evaluate_model_log(theta, data, sigma):
    """
        this is the central function. it takes in a set of RV parameters (in theta),
        runs the model with this parametr set, compares the results to data,
        and then returns the log-likelyhood of this parameterset. sigma
        is the observation uncertainty std (which is needed in bayesian optimization)
    """
    data_obs = data
    rv_params = {prior_names[i]:theta[i] for i in range(len(prior_names))}
    res, t_res = solve(**rv_params,**fixed_params)
    data_model = pd.DataFrame(res, columns=['S', 'I', 'R', 'D'], index=t_res)
    data_model = data_model[target_var]
    # extract the right dates from the model (same dates as in obs)
    # in case of sub-daily model resolution this automatically reduces
    # it to daily resolution
    data_model_sub = data_model[data_model.index.isin(global_obs_dates)]
    # it can be that the model starts too late (or the data to early, as you see it).
    if data_model_sub.index[0] > global_obs_dates[0]:
        # in this case we have to fill up model data with zeros
        n_append = len(data_obs) - len(data_model_sub)
        data_model_sub = np.concatenate([np.zeros(n_append), data_model_sub.values])
    else:
        data_model_sub = data_model_sub.values

    return -0.5 * np.sum((data_model_sub - data_obs)**2/sigma**2)


# in order to use our above function with theano,
# we need to use the following wrapper:
# https://stackoverflow.com/questions/41109292/solving-odes-in-pymc3
class LogLike(tt.Op):

    itypes = [tt.dvector] # expects a vector of parameter values when called
    otypes = [tt.dscalar] # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, data, sigma):


        # add inputs as class attributes
        self.likelihood = loglike
        self.data = data
        self.sigma = sigma

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        theta, = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(theta, self.data, self.sigma)

        outputs[0][0] = np.array(logl) # output the log-likelihood



print('loading data')
data_obs = load_data(country)
time_obs = num_to_date(data_obs.index)
global_obs_dates = data_obs.index
data_obs.plot()
plt.savefig(f'{country}_obs.svg')

# the model is written in dimensionless form , so everything is in fractions.
# therfore, to compare it with the real data, we need to have the population size
# of the country

population_per_country = {'AUT':8.822e6,
                          'SWE':10.12e6}

N_pop = population_per_country[country]

data_obs_dimless = data_obs[['D', 'I']] / N_pop


ndraws = 100 # number of draws from the distribution
nburn = 100  # number of "burn-in points" (which we'll discard)
#nburn = 100   # number of "burn-in points" (which we'll discard)



def lognorm_coeffs(loc,scale):
    return np.log(loc**2/np.sqrt(loc**2+scale**2)), np.log(1+scale**2/loc**2)

prior_names = ['Ro', 'I_init', 'd']
# use PyMC3 to sampler from log-likelihood
with pm.Model() as pmodel:
    #sigma = pm.Normal('sigma',0.00001)
    # sigma is the "noise" in the observations that we assume
    sigma = 0.000005
    Ro = pm.TruncatedNormal('Ro',2,1, lower=0)
    # discrete uniform seems to have a bug (does not respect bounds)
    # therefore we use continous uniform and round it to full numbers
    # date_start = pm.Uniform('date_start',50,75)
    # date_start = tt.round(date_start)
    I_init = pm.Lognormal('I_init',*lognorm_coeffs(1e-5,1e-5))
    d = pm.Lognormal('d',*lognorm_coeffs(0.01,0.01))
    # convert params to tensor
    theta = tt.as_tensor_variable([Ro, I_init, d])

    # create our Op
    logl = LogLike(evaluate_model_log, data_obs_dimless['D'].values, sigma)

    # use a DensityDist (use a lamdba function to "call" the Op)
    pm.DensityDist('likelihood', lambda v: logl(v), observed={'v': theta},
                   random = Ro.random
                   )
    print(pm.find_MAP(model=pmodel))
    trace = pm.sample(ndraws, tune=nburn, discard_tuned_samples=True)

plt.figure()
pm.traceplot(trace)


# convert trace to usuable list of dicts
# trace.points contains this, but also contains other vars (like limits)
# so welect only yhe ones we need
trace_points = [{key:e[key] for key in prior_names} for e in trace.points()]

def single_run(theta):
    res, t_res = solve(**theta,**fixed_params)
    return res, t_res

res, t_res = single_run(trace_points[0])

plt.figure()
plt.plot(res)


# plot priors and posteriors
n_sample = 100000
priors_sample = pm.sample_prior_predictive(n_sample, pmodel)
plt.figure(figsize=(10,3))
n_priors = len(prior_names)

colors = sns.color_palette('colorblind', 4)
posterior_sample = {key: trace.get_values(key) for key in prior_names}
for i, key in enumerate(prior_names):
    ax = plt.subplot(1,n_priors, i+1)
    x_prior = priors_sample[key]
    x_post = posterior_sample[key]
    sns.distplot(x_prior, label='prior', color=colors[0])
    sns.distplot(x_post, label='posterior', color=colors[1])
    ax.set_xlabel(key)
    # plt.axvline(posterior_means[key], color=colors[1])
    plt.legend()
    #plt.title(f'post mean={posterior_means[key]:.3f}')
    sns.despine()
plt.suptitle(f'{country}')
plt.tight_layout()
plt.savefig(f'{country}_priors_and_posteriors.svg')


def mc_posterior(trace_points):
    """
        sample from the posteriors and make model runs with the smapled parameters
        compute percentiles of the solution
    """
    #TODO: the following does not make sense when date_start is not fixed!
    res = []
    for theta in trace_points:
        pert_res, t_res = single_run(theta)
        res.append(pert_res)

    res = np.array(res)
    # compute percentiles
    percs_res = np.percentile(res, percs, axis=0)
    return percs_res, res, t_res


percs_res, res_all, time_num = mc_posterior(trace_points)

time = num_to_date(time_num)
# plot
fig, ax = plt.subplots(1, 1, figsize=[8, 3])
lines_per_percentile = []
for ip in range(len(percs)):
    l1 = ax.plot(time, percs_res[ip,:, 0], color=colors[0], linestyle=linestyle_percs[ip])
    l2 = ax.plot(time, percs_res[ip,:, 1], color=colors[1], linestyle=linestyle_percs[ip])
    l3 = ax.plot(time, percs_res[ip,:, 2], color=colors[2], linestyle=linestyle_percs[ip])
    #l4 = ax.plot(time, percs_res[ip,:, 3], color=colors[3], linestyle=linestyle_percs[ip])
    lines_per_percentile.append([l1, l2, l3])
ax.set_xlabel('time (days)', fontsize=12)
ax.set_ylabel('fraction of population ', fontsize=12)
# it does not make too much senes to plot the death here, since the fraction is so small
#ax.legend(['Susceptible', 'Infected', 'Removed', 'Dead'], loc=(1.1, 0.7))
ax.legend(['Susceptible', 'Infected', 'Removed'], loc=(1.1, 0.7))
ax.grid(axis='y', linestyle=':', linewidth=0.5)
plt.title(f'{country}')
plt.tight_layout()
plt.savefig(f'{country}_modelrun.svg')



# plot only deaths

fig, ax = plt.subplots(1, 1, figsize=[8, 3])
plt.plot(time_obs, data_obs_dimless['D'])
for ip in range(len(percs)):
    l1 = ax.plot(time, percs_res[ip,:, 3], color=colors[3], linestyle=linestyle_percs[ip])
ax.legend(['real']+[f'model p{p}' for p in percs], loc=(1.05, 0.1))
ax.set_xlabel('time (days)', fontsize=12)
ax.set_ylabel('fraction of population', fontsize=12)
plt.title(f'{country} deaths')
plt.tight_layout()
plt.savefig(f'{country}_modelfit.svg')
ax.semilogy()
plt.savefig(f'{country}_modelfit_logy.svg')

fig, ax = plt.subplots(1, 1, figsize=[8, 3])
plt.plot(time_obs, data_obs_dimless['D']*N_pop)
for ip in range(len(percs)):
    l1 = ax.plot(time, percs_res[ip,:, 3]*N_pop, color=colors[3], linestyle=linestyle_percs[ip])
ax.legend(['real']+[f'model p{p}' for p in percs], loc=(1.05, 0.1))
ax.set_xlabel('time (days)', fontsize=12)
ax.set_ylabel('No people', fontsize=12)
plt.title(f'{country} deaths')
plt.tight_layout()
plt.savefig(f'{country}_modelfit_total.svg')
ax.semilogy()
plt.savefig(f'{country}_modelfit_total_logy.svg')