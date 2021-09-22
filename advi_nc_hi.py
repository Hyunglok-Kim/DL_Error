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

## ==== data load ==== ##
# 1 load the data sets
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

#Group the data using RFI values
nog = 10
RFI_tavg = pr_data['SMOS_RFI_tavg'].dropna(axis=0, how='any')
RFI_p = np.empty(nog, dtype=object)
for i in range(1,nog+1):
    RFI_p[i-1] = np.percentile(RFI_tavg, i*100/nog)

RFI_class = np.copy(pr_data['SMOS_RFI_tavg'])

for i in range(0,nog):
    if i > 0:
        RFI_class[(pr_data['SMOS_RFI_tavg'] > RFI_p[i-1]) & (pr_data['SMOS_RFI_tavg'] <= RFI_p[i])]=i
    elif i == 0:
        RFI_class[(pr_data['SMOS_RFI_tavg'] <= RFI_p[i])]=i
pr_data['RFI_class'] = RFI_class

# TC estimations with std value larger than 0.2 might be unstable
std_thred     = 0.1
mask_std_A2   = (res_data['AMSR2_std'] <= std_thred) & (pr_data.ltype!=21) & (pr_data.ltype!=17)
mask_std_AS   = (res_data['ASCAT_std'] <= std_thred) & (pr_data.ltype!=21) & (pr_data.ltype!=17)
mask_std_SMOS = (res_data['SMOS_std'] <= std_thred) & (pr_data.ltype!=21) & (pr_data.ltype!=17)
mask_std_SMAP = (res_data['SMAP_std'] <= std_thred) & (pr_data.ltype!=21) & (pr_data.ltype!=17)

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
A2_num   = A2_fMSE.drop(columns=['ltype', 'RFI_class', 'SMOS_RFI_tavg', 'AMSR2_fMSE'])
A2_cat   = A2_fMSE['ltype']
A2_RFI   = A2_fMSE['RFI_class']
A2_y     = A2_fMSE['AMSR2_fMSE']
# ASCAT
AS_num   = AS_fMSE.drop(columns=['ltype', 'RFI_class', 'SMOS_RFI_tavg', 'ASCAT_fMSE'])
AS_cat   = AS_fMSE['ltype']
AS_RFI   = AS_fMSE['RFI_class']
AS_y     = AS_fMSE['ASCAT_fMSE']
# SMOS
SMOS_num = SMOS_fMSE.drop(columns=['ltype', 'RFI_class', 'SMOS_RFI_tavg', 'SMOS_fMSE'])
SMOS_cat = SMOS_fMSE['ltype']
SMOS_RFI = SMOS_fMSE['RFI_class']
SMOS_y   = SMOS_fMSE['SMOS_fMSE']
# SMAP
SMAP_num = SMAP_fMSE.drop(columns=['ltype', 'RFI_class', 'SMOS_RFI_tavg', 'SMAP_fMSE'])
SMAP_cat = SMAP_fMSE['ltype']
SMAP_RFI = SMAP_fMSE['RFI_class']
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

## ==== Take Inputrs from Console ==== ##
product = str(input('Product?'))
method = str(input('Method?'))
dist = str(input('Distribution?'))

## ==== Product Select ==== ##
if product == 'SMAP':
    fMSE = SMAP_fMSE
    num_scaled = SMAP_num_scaled
    RFI = SMAP_RFI
    y_input = SMAP_y
    #y_input, ld = stats.boxcox(y_input)
elif product == 'SMOS':
    fMSE = SMOS_fMSE
    num_scaled = SMOS_num_scaled
    RFI = SMOS_RFI
    y_input = SMOS_y
elif product == 'ASCAT':
    fMSE = AS_fMSE
    num_scaled = AS_num_scaled
    RFI = AS_RFI
    y_input = AS_y
elif product == 'AMSR2':
    fMSE = A2_fMSE
    num_scaled = A2_num_scaled
    RFI = A2_RFI
    y_input = A2_y

## ==== Input Data ==== ##
X_input = np.hstack((num_scaled, RFI.values.reshape(-1,1)))
Rclasses = fMSE['RFI_class'].unique()
nRclasses = len(Rclasses)
Rclass_lookup = dict(zip(Rclasses, range(nRclasses)))
Rclass = fMSE['Rclass_code'] = fMSE.RFI_class.replace(Rclass_lookup).values.astype('int')
X_train, X_test, y_train, y_test = train_test_split(X_input, y_input, test_size=0.33, stratify=Rclass, random_state=42)
Rclass_train = X_train[:,-1].astype('int')
Rclass_test  = X_test[:,-1].astype('int')
X_train  = X_train[:,:-1]
X_test   = X_test[:,:-1]

X        = X_train
y        = y_train
Rclass   = Rclass_train
nop      = X_train.shape[1]

## ==== Hierarchical Model Non-centered ==
with pm.Model() as hi_model:

    # Priors for the model paramters
    # Gaussians for the means of the priors of the random intercpets and slopes
    mu_a = pm.Normal('mu_a', mu=0, sd=1e5)
    for i in range(1,nop+1):
        exec('mu_b'+str(i)+'=pm.Normal(\'mu_b'+str(i)+'\', mu=0, sd=1e5)')
    
    # Half-Cauchy for the standard deviation of the priros of the random intercpets and slops
    sigma_a  = pm.HalfCauchy('sigma_a', 1e5)
    for i in range(1,nop+1):
        exec('sigma_b'+str(i)+'=pm.HalfCauchy(\'sigma_b'+str(i)+'\', beta=1e5)')
  
    # Gaussian priors for random intercpets and slopes
    a_offset = pm.Normal('Intercept_offset', mu=0, sd=1, shape=nRclasses)
    a  = pm.Deterministic('Intercept', mu_a + a_offset*sigma_a)
    for i in range(1,nop+1):
        exec('b'+str(i)+'_offset = pm.Normal(\''+all_predictors[i-1]+'_offset\', mu=0, sd=1, shape=nRclasses)')
        exec('b'+str(i)+'=pm.Deterministic(all_predictors[i-1], mu_b'+str(i)+'+b'+str(i)+'_offset*sigma_b'+str(i)+')')
        
    # Model
    mu_init = 'mu = a[Rclass]'
    for i in range(1,nop+1):
        mu_init = mu_init+'+b'+str(i)+'[Rclass]*X[:,'+str(i-1)+']'
    exec(mu_init)
    
    # Data likelihood
    if dist == 'Gumbel':
        #Model errors
        beta = pm.HalfCauchy('sigma_y', 1e5)
        y_hat = pm.Gumbel('fMSE', mu = mu, beta = beta, observed = y)
        #y_hat = pm.Normal('fMSE', mu = mu, sd = beta, observed = y)
        #y_hat = pm.Rice('fMSE', nu = mu, sigma = beta, observed = y)
        #y_hat = pm.Kumaraswamy('fMSE', a = np.abs(mu), beta = np.abs(mux), observed=y)

    elif dist == 'Beta':
        mu_ax = pm.Normal('mu_ax', mu=0, sd=1e5)
        for i in range(1,nop+1):
            exec('mu_bx'+str(i)+'=pm.Normal(\'mu_bx'+str(i)+'\', mu=0, sd=1e5)')
        
        # Half-Cauchy for the standard deviation of the priros of the random intercpets and slops
        sigma_ax  = pm.HalfCauchy('sigma_xa', 1e5)
        for i in range(1,nop+1):
            exec('sigma_bx'+str(i)+'=pm.HalfCauchy(\'sigma_bx'+str(i)+'\', beta=1e5)')
  
        # Gaussian priors for random intercpets and slopes
        ax_offset = pm.Normal('Interceptx_offset', mu=0, sd=1, shape=nRclasses)
        ax  = pm.Deterministic('Interceptx', mu_ax + ax_offset*sigma_ax)
        for i in range(1,nop+1):
            exec('bx'+str(i)+'_offset = pm.Normal(\''+all_predictors[i-1]+'x_offset\', mu=0, sd=1, shape=nRclasses)')
            exec('bx'+str(i)+'=pm.Deterministic(\''+all_predictors[i-1]+'x\', mu_bx'+str(i)+'+bx'+str(i)+'_offset*sigma_bx'+str(i)+')')
            
        # Model
        mux_init = 'mux = a[Rclass]'
        for i in range(1,nop+1):
            mux_init = mux_init+'+bx'+str(i)+'[Rclass]*X[:,'+str(i-1)+']'
        exec(mux_init)
        y_hat = pm.Beta('fMSE', alpha = np.abs(mu), beta = np.abs(mux), observed=y) 

#pm.model_to_graphviz(hi_model)

## ==== Run the Model ==== ##
with hi_model:
    fit_advi= pm.fit(method = pm.ADVI(), n = 500000)
    trace_advi= fit_advi.sample(10000)

## ==== Save the Ouputs ==== ##
ofp = '/home/hyung/Lab/libs/python/DL_Error_data/'+product+'_ywos_nc_hi_'+dist
filename = ofp + '_' + method +'.out'
myshelf = shelve.open(filename, 'n')
myshelf['hi_model'] = hi_model
if method == 'advi':
    myshelf['fit_advi'] = fit_advi
    myshelf['trace_advi'] = trace_advi
elif method == 'nuts':
    myshelf['trace_nuts'] = trace_nuts
    myshelf['idata_nuts'] = idata_nuts
myshelf['X_train'] = X_train
myshelf['X_test'] = X_test
myshelf['y_train'] = y_train
myshelf['y_test'] = y_test

