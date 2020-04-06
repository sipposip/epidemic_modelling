"""
stochastic version of the SIR model

the model is fitted to real data, using a monte carlo technique.
starting from prior distributions of model parameters, a large number of model runs is made using monte-carlo sampling.
each run is compared to the data, and weighted according to its fit to the data.
With this, posterior distributions of the model parameters are computed. This is done with forming a pool
of possible parameter values (the ones used in the monte-carlo sampling), and then randomly drawing from them,
but with non-uniform probablities. the probabiliyt of chosing the parameters P from run X is p=1/e*norm, where "e" is the
L1-error of the fit of model X, and norm a factor so that the probability summed over all runs ==1.
with the posterior distributions again monte carlo sampling is done, and man simulations are made. These simulations
are then the probabilistic prediction.


"""


import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from scipy import stats
import datetime
import pandas as pd
import seaborn as sns
from tqdm import trange, tqdm

reference_date = pd.to_datetime('20200101') # this reference date is only for the definition of the
# date in the model. it is NOT the start date of the model. in the model we use integers as dates, which descrive
# the day relative to the reference date (can also be negative)


fixed_params = dict(
    L=365,  # length run (days)
    tstep = 0.5, # days, must by <=1 and 1/n
    t_start_qua=0.,  # time quarantine starts (days)
    t_stop_qua=0.,  # time quarantine ends (days)
    # Initial conditions. They are more crucial than one might think, especially when considering the start date!
    x0=np.array([1-1e-3, 1e-3, 0., 0.]),
    Ro_qua=2.5,  # Ro during quarantine

)


priors = dict(
    # each entry is a function f(n) that returns n samples from the prior distribution
    Ro = lambda n: np.abs(stats.norm(loc=2, scale=1).rvs(n)),
    tau = stats.norm(loc=14, scale=1).rvs,
    d=lambda n: np.abs(stats.norm(loc=0.01, scale=0.01).rvs(n)),  # death rate (% infected who die)
    # the date the pandemic starts. uniform discrete distribution. unit: day relative to reference date
    date_start = lambda n: np.random.randint(30,80, size=n) # upper limit must be lower than the lower limit of obs
)

# sanity check
for key in priors:
    if key in fixed_params.keys():
        raise ValueError(f'"{key}" is both in fixed_params and priors!')


percs = [95, 90, 75, 50, 25, 10, 5]
linestyle_percs = [':','--','-.','-','-.','--',':']
N_mc_fit = 10000 #  number of sims that are tested on the data
N_mc_post = 10000 # number of sims made from posterior distribution
target_var = 'D'  # which data to fit on. 'D' for deaths, 'I' for infected

country = 'AUT'
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


def solve(L,tstep, x0, Ro, tau, d, t_start_qua, t_stop_qua, Ro_qua, date_start):
    """solve the model with a fixed set of parameters"""
    t = np.arange(0, L, tstep) / tau
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


def evaluate_model(data_model, data_obs):
    """
        compare a single model simulation with the observations.
        compute L1 error
        note:all data dimensionless
    """

    data_model  = data_model[target_var]
    data_obs  = data_obs[target_var]
    # extract the right dates from the model (same dates as in obs)
    # in case of sub-daily model resolution this automatically reduces
    # it to daily resolution
    data_model_sub = data_model[data_model.index.isin(data_obs.index)]
    # it can be that the model starts too late (or the data to early, as you see it)
    if data_model_sub.index[0] > data_obs.index[0]:
        # in this case we have to fill up model data with zeros
        n_append = len(data_obs) - len(data_model_sub)
        data_model = np.concatenate ([np.zeros(n_append), data_model_sub.values])
    assert(data_model.shape == data_obs.shape)
    # l1 error
    return np.mean(np.abs(data_model - data_obs.values))


def evaluate_single_run(params):
    res, t_res = solve(**params)
    data_model = pd.DataFrame(res, columns=['S', 'I', 'R', 'D'], index=t_res)
    fiterror = evaluate_model(data_model, data_obs_dimless)
    return fiterror, data_model


def fit(fixed_params, priors, N_mc_fit):
    """
        do a monte-carlo simulation for fitting.
        N_mc_fit runs of the model are made. each one draws parameters from the priors.
        Then the run is evaluated and stored. returns the evaluation score.
    """

    record_params = {key:[] for key in priors.keys()}
    record_params['fiterror'] = []
    #record_modelrun =  []
    for _ in trange(N_mc_fit):
        pertubed_params = fixed_params.copy()
        for param in priors.keys():
            # the prior function returns arrays (of size 1 if n=1), but here we need
            # a float, therefore we select the first (and only) element
            pertubed_params[param] = priors[param](1)[0]

        fiterror, data_model = evaluate_single_run(pertubed_params)
        for key in priors.keys():
            record_params[key].append(pertubed_params[key])
        record_params['fiterror'].append(fiterror)
        #record_modelrun.append(data_model)
    assert(all(len(e)==N_mc_fit for e in record_params.values()))
    record_params = pd.DataFrame(record_params)

    weights = 1 / record_params['fiterror']
    # norm the weights to sum 1
    weights = weights / np.sum(weights)
    record_params['weights'] = weights
    return record_params #, record_modelrun


def sample_from_posteriors(record_params, n):
    """
    get an empirical sample of size n from the posterior for all params, weighted by
    the weights computed in the fit process
    :return: if n==1: dictionary, else DataFrame
    """
    n_smaples_in_record = len(record_params)
    # get indices for sampling, based on the weights from the fit.
    idcs = np.random.choice(np.arange(n_smaples_in_record),p=record_params['weights'], size=n)
    params = record_params.iloc[idcs]
    # we dont need the fiterror and weights
    params = params.drop(['fiterror', 'weights'], axis=1)
    if n==1:
        # conver to dict, without index
        params = params.to_dict('records')[0]
    return params


def compute_posterior_means(record_params):
    return {key:np.average(record_params[key], weights=record_params['weights']) for key in record_params.keys()}


def mc_posterior(fixed_params, record_params, N_mc_post):
    """
        sample from the posteriors and make model runs with the smapled parameters
        compute percentiles of the solution
    """
    res = []
    for _ in trange(N_mc_post):
        pertubed_params = fixed_params.copy()
        params_from_posterior = sample_from_posteriors(record_params, n=1)
        for key in params_from_posterior:
            pertubed_params[key] = params_from_posterior[key]

        pert_res, t_res = solve(**pertubed_params)
        res.append(pert_res)

    res = np.array(res)
    # compute percentiles
    percs_res = np.percentile(res, percs, axis=0)
    return percs_res, res, t_res




print('loading data')
data_obs = load_data(country)
time_obs = num_to_date(data_obs.index)
data_obs.plot()
plt.savefig(f'{country}_obs.svg')

# the model is written in dimensionless form , so everything is in fractions.
# therfore, to compare it with the real data, we need to have the population size
# of the country

population_per_country = {'AUT':8.822e6,
                          'SWE':10.12e6}

N_pop = population_per_country[country]

data_obs_dimless = data_obs[['D', 'I']] / N_pop

plt.figure()
data_obs_dimless.plot()
plt.savefig(f'{country}_obs_dimless.svg')

print('fitting model (=computing posteriors)')
#record_params, record_modelruns = fit(fixed_params, priors, N_mc_fit)
record_params = fit(fixed_params, priors, N_mc_fit)
posterior_means = compute_posterior_means(record_params)
print('running models from posteriors')
percs_res, res_all, time_num = mc_posterior(fixed_params, record_params, N_mc_post)

time = num_to_date(time_num)


# plotting

colors = sns.color_palette('colorblind', 4)

# plot priors and posteriors
plt.figure(figsize=(10,3))
n_priors = len(priors)
n_sample = 100000
posterior_sample = sample_from_posteriors(record_params, n_sample)
for i, key in enumerate(priors.keys()):
    ax = plt.subplot(1,n_priors, i+1)
    x_prior = priors[key](n_sample)
    x_post = posterior_sample[key]
    sns.distplot(x_prior, label='prior', color=colors[0])
    sns.distplot(x_post, label='posterior', color=colors[1])

    plt.axvline(posterior_means[key], color=colors[1])
    plt.legend()
    plt.title(f'post mean={posterior_means[key]:.3f}')
    sns.despine()
plt.suptitle(f'{country}')
plt.tight_layout()
plt.savefig(f'{country}_priors_and_posteriors.svg')

# plot
fig, ax = plt.subplots(1, 1, figsize=[8, 3])
lines_per_percentile = []
for ip in range(len(percs)):
    l1 = ax.plot(time, percs_res[ip,:, 0], color=colors[0], linestyle=linestyle_percs[ip])
    l2 = ax.plot(time, percs_res[ip,:, 1], color=colors[1], linestyle=linestyle_percs[ip])
    l3 = ax.plot(time, percs_res[ip,:, 2], color=colors[2], linestyle=linestyle_percs[ip])
    #l4 = ax.plot(time, percs_res[ip,:, 3], color=colors[3], linestyle=linestyle_percs[ip])
    lines_per_percentile.append([l1, l2, l3])

single_lines = []
if plot_single_sims:
    for i in range(N_mc_fit):
        l1 = ax.plot(time, res_all[i,:, 0], color=colors[0], alpha=0.1)
        l2 = ax.plot(time, res_all[i,:, 1], color=colors[1], alpha=0.1)
        l3 = ax.plot(time, res_all[i,:, 2], color=colors[2], alpha=0.1)
        #l4 = ax.plot(time, res_all[i,:, 3], color=colors[3], alpha=0.1)
        single_lines.append([l1,l2,l3])
# TODO: the following does no yett work for a datetime x axis. TODO: change t_start_qua to datetimes!
#rect = ax.axvspan(fixed_params['t_start_qua'], fixed_params['t_stop_qua'], color='0.5', alpha=0.5, linewidth=0, zorder=1)
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
