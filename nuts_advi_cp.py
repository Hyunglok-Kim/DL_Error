import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import pymc3 as pm
import arviz as az
import pandas as pd
import scipy.linalg as sp
from scipy.special import boxcox, inv_boxcox
import seaborn as sns
print('Running on PyMC3 v{}'.format(pm.__version__))
import theano
from sklearn.model_selection import train_test_split
import shelve
import warnings

#matplotlib inline
sns.set()
warnings.filterwarnings('ignore')

## ==== Functions ==== ##
def nuts_advi(X, y, ofp, y_dist, opt, test_size=0.33):
    
    k = X.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    y_tensor = theano.shared(y_train.astype('float64'))
    X_tensor = theano.shared(X_train.astype('float64'))

    with pm.Model() as base_model:

        # Intercept term
        β0 = pm.Normal('β0', mu=0, sd=1e5)

        # Beta coefficients for predictor variables
        β = pm.MvNormal('β', mu=np.zeros(k), cov=np.eye(k), shape=k)

        # Calcuate mean from the normal variables, and add intercept
        mu = pm.math.dot(X_tensor,β) + β0

        # Pass the mu and beta with the observed data
        if y_dist == 'Gumbel':
            # Gumbel distribution 
            beta = pm.HalfCauchy('beta', 1e5)
            y_likelihood = pm.Gumbel('fMSE', mu=mu, beta=beta, observed=y_tensor)
        elif y_dist == 'Beta':
            # Beta distribution
            # Intercept term
            β0x = pm.Normal('β0x', mu=0, sd=1e5)

            # Beta coefficients for predictor variables
            βx = pm.MvNormal('βx', mu=np.zeros(k), cov=np.eye(k), shape=k)

            # Calcuate mean from the normal variables, and add intercept
            mux = pm.math.dot(X_tensor,βx) + β0x

            y_likelihood = pm.Beta('fMSE', alpha=np.abs(mu)+1/1e10, beta=np.abs(mux)+1/1e10, observed=y_tensor)
        #start = pm.find_MAP()

    #pm.model_to_graphviz(base_model)
    
    if opt == 'nuts':
        with base_model:
        # Variational inference with ADVI optimization
            step       = pm.NUTS(target_accept=0.95)
            trace_nuts = pm.sample(draws=4000, step=step, tune=1000, cores=4, init='adapt_diag')
            idata_nuts = az.from_pymc3(trace_nuts)

        filename = ofp + '_nuts.out'
        my_shelf = shelve.open(filename, 'n')
        my_shelf['base_model'] = base_model
        my_shelf['trace_nuts'] = trace_nuts
        my_shelf['idata_nuts'] = idata_nuts
        my_shelf['X_tensor']   = X_tensor
        my_shelf['y_tensor']   = y_tensor
        my_shelf['X_train']    = X_train
        my_shelf['y_train']    = y_train
        my_shelf['X_test']     = X_test
        my_shelf['y_test']     = y_test
        my_shelf.close()
        
    elif opt == 'advi':
        map_tensor_batch = {y_tensor: pm.Minibatch(y_train, 1000), X_tensor: pm.Minibatch(X_train, 1000)}
        
        with base_model:
             fit_advi= pm.fit(method=pm.ADVI(), n=1000000, more_replacements = map_tensor_batch)
        
        trace_advi = fit_advi.sample(10000)
       
        filename = ofp + '_advi.out'
        my_shelf = shelve.open(filename, 'n')
        my_shelf['base_model'] = base_model
        my_shelf['fit_advi']   = fit_advi
        my_shelf['trace_advi'] = trace_advi
        my_shelf['X_tensor']   = X_tensor
        my_shelf['y_tensor']   = y_tensor
        my_shelf['X_train']    = X_train
        my_shelf['y_train']    = y_train
        my_shelf['X_test']     = X_test
        my_shelf['y_test']     = y_test
        my_shelf.close()

## ==== data load ==== ##
# load the data sets
pr_file   = "/home/hyung/Lab/data/DL_Error/predictors.csv"
res_file  = "/home/hyung/Lab/data/DL_Error/responses.csv"
pr_data   = pd.read_csv(pr_file)
res_data  = pd.read_csv(res_file)

# Rainf_f/Precip/SWdown_min have some issue because they are all zeros
pr_data.drop(columns=['Rainf_min', 'Rainf_f_min', 'Rainf_f_max','Rainf_f_tavg', 'TotalPrecip_min'], inplace=True)
pr_data.drop(columns=['Evap_min', 'Evap_max', 'Evap_tavg'], inplace=True)
pr_data.drop(columns=['LWdown_f_max', 'LWdown_f_min', 'LWdown_f_tavg'], inplace=True)
pr_data.drop(columns=['Qair_f_max',	'Qair_f_min', 'Qh_max',	'Qh_min'], inplace=True)
pr_data.drop(columns=['Qle_min', 'Qle_max', 'Qle_tavg'], inplace=True)
pr_data.drop(columns=['SWdown_f_min', 'SWdown_f_max', 'SWdown_f_tavg'], inplace=True)
pr_data.drop(columns=['SMOS_RFI_min', 'SoilMoist_max', 'SoilMoist_min',	'SoilMoist_tavg'], inplace=True)
pr_data.drop(columns=['Tair_f_max', 'Tair_f_min', 'Tair_f_tavg', 'aspect'], inplace=True)
pr_data.drop(columns=['Wind_f_max',	'Wind_f_min', 'Wind_f_tavg'], inplace=True)
pr_data.drop(columns=['LAI_min', 'LAI_max','Greenness_min', 'Greenness_max', 'AvgSurfT_min', 'AvgSurfT_max'],inplace=True)
pr_data.drop(columns=['SoilTemp_min', 'SoilTemp_max','RadT_min', 'RadT_max'],inplace=True)
pr_data.drop(columns=['SMAP_vo_min', 'SMAP_vo_max','SMAP_rc_min', 'SMAP_rc_max'],inplace=True)
pr_data.drop(columns=['albedo_max', 'albedo_min','albedo_std', 'TotalPrecip_max','Rainf_max','SMOS_RFI_max'],inplace=True)

# TC estimations with std value larger than 0.2 might be unstable
std_thred     = 0.1
mask_std_A2   = res_data['AMSR2_std'] <= std_thred
mask_std_AS   = res_data['ASCAT_std'] <= std_thred
mask_std_SMOS = res_data['SMOS_std'] <= std_thred
mask_std_SMAP = res_data['SMAP_std'] <= std_thred

# 2 clean the data sets
selected_predictors = list(pr_data.columns.values)

sel_A2   = selected_predictors.copy()
sel_A2.append('AMSR2_fMSE')
sel_AS   = selected_predictors.copy()
sel_AS.append('ASCAT_fMSE')
sel_SMOS = selected_predictors.copy()
sel_SMOS.append('SMOS_fMSE')
sel_SMAP = selected_predictors.copy()
sel_SMAP.append('SMAP_fMSE')

A2_fMSE   = pr_data.join(res_data['AMSR2_fMSE'])[sel_A2]
AS_fMSE   = pr_data.join(res_data['ASCAT_fMSE'])[sel_AS]
SMOS_fMSE = pr_data.join(res_data['SMOS_fMSE'])[sel_SMOS]
SMAP_fMSE = pr_data.join(res_data['SMAP_fMSE'])[sel_SMAP]

# select fMSE <= threshold
A2_fMSE   = A2_fMSE[mask_std_A2]
AS_fMSE   = AS_fMSE[mask_std_AS]
SMOS_fMSE = SMOS_fMSE[mask_std_SMOS]
SMAP_fMSE = SMAP_fMSE[mask_std_SMAP]
                      
# drop N/A
A2_fMSE.dropna(axis=0, how='any', inplace=True)
AS_fMSE.dropna(axis=0, how='any', inplace=True)
SMOS_fMSE.dropna(axis=0, how='any', inplace=True)
SMAP_fMSE.dropna(axis=0, how='any', inplace=True)

# Numeric, categorical predictors and response, y`
# AMSR2
A2_num   = A2_fMSE.drop(columns=['ltype', 'AMSR2_fMSE'])
A2_cat   = A2_fMSE['ltype']
A2_y     = A2_fMSE['AMSR2_fMSE']
# ASCAT
AS_num   = AS_fMSE.drop(columns=['ltype', 'ASCAT_fMSE'])
AS_cat   = AS_fMSE['ltype']
AS_y     = AS_fMSE['ASCAT_fMSE']
# SMOS
SMOS_num = SMOS_fMSE.drop(columns=['ltype', 'SMOS_fMSE'])
SMOS_cat = SMOS_fMSE['ltype']
SMOS_y   = SMOS_fMSE['SMOS_fMSE']
# SMAP
SMAP_num = SMAP_fMSE.drop(columns=['ltype', 'SMAP_fMSE'])
SMAP_cat = SMAP_fMSE['ltype']
SMAP_y   = SMAP_fMSE['SMAP_fMSE']

all_predictors = SMAP_num.columns.values

# Standardize numeric/response columns, to mean 0 variance 1
# AMSR2
A2_mean       = A2_num.mean()
A2_std        = A2_num.std()
A2_num_scaled = np.array((A2_num - A2_mean) / A2_std)
A2_y_mean     = A2_y.mean()
A2_y_std      = A2_y.std()
A2_y_scaled   = np.array((A2_y - A2_y_mean) / A2_y_std)
# ASCAT
AS_mean       = AS_num.mean()
AS_std        = AS_num.std()
AS_num_scaled = np.array((AS_num - AS_mean) / AS_std)
AS_y_mean     = AS_y.mean()
AS_y_std      = AS_y.std()
AS_y_scaled   = np.array((AS_y - AS_y_mean) / AS_y_std)
# SMOS
SMOS_mean       = SMOS_num.mean()
SMOS_std        = SMOS_num.std()
SMOS_num_scaled = np.array((SMOS_num - SMOS_mean) / SMOS_std)
SMOS_y_mean     = SMOS_y.mean()
SMOS_y_std      = SMOS_y.std()
SMOS_y_scaled   = np.array((SMOS_y - SMOS_y_mean) / SMOS_y_std)
# SMAP
SMAP_mean        = SMAP_num.mean()
SMAP_std         = SMAP_num.std()
SMAP_num_scaled  = np.array((SMAP_num - SMAP_mean) / SMAP_std)
SMAP_y_mean      = SMAP_y.mean()
SMAP_y_std       = SMAP_y.std()
SMAP_y_scaled    = np.array((SMAP_y - SMAP_y_mean) / SMAP_y_std)

## ==== Take Inputs from Console ==== ##
product = str(input('Product?'))
method = str(input('Method?'))
dist = str(input('Distribution?'))

## ==== NUTS or ADVI ==== ## 
test_size = 0.33
ofp = '/home/hyung/Lab/libs/python/DL_Error_data/'+product+'_ywos_cp_'+method+'_'+dist
if product == 'SMAP':
    nuts_advi(SMAP_num_scaled, SMAP_y.values, ofp, dist, method, test_size)
elif product == 'SMOS':
    nuts_advi(SMOS_num_scaled, SMOS_y.values, ofp, dist, method, test_size)
elif product == 'ASCAT':
    nuts_advi(AS_num_scaled, AS_y.values, ofp, dist, method, test_size)
elif product == 'AMSR2':
    nuts_advi(A2_num_scaled, A2_y.values, ofp, dist, method, test_size)
