# STATISTICAL ANALYSIS BY Faycal Kilali
# Date: 2023, April.
# LICENSE: CC0 1.0 Universal
# GitHub: www.github.com/faycalki
# Linear regression model function partially from Mount Allison University's MATH-1311 Course.

import re
import pandas as pd 
import numpy as np
import scipy.stats as stats
from scipy.stats import ttest_ind
from matplotlib import pyplot as plt
import seaborn as sns
from lmfit import Parameters, minimize, fit_report
import matplotlib.patches as mpatches

# Some linear fitting
def linear_model(x, params): # Linear model, with a more descriptive name.

    #Extract the current parameter value

    a0 = params['a0'].value
    a1 = params['a1'].value
    #a2 = params['a2'].value

    y = a0 + a1*x # polynomial
    # The general trend is y = f(a)/0! x + f'(a)/1! * x^2 + ... + f^(n)(x)/(n!) x^n (taylor series polynomial)
    return y # Outputs y

def least_squares(params, x, y): # Least squares model, with a more descriptive name.

    #Compute the model values:

    model_values = linear_model(x,params)

    #Compute the error (python automatically squares it)

    error = y - model_values

    #Output the errors

    return error


# Purpose: to extract the year's values from the launch_date column for the AMD datasets, that is, we won't be using this for the intel datasets
def year_extraction_for_amd(row): 
    launch_date = row['launch_date']
    match = year_pattern.search(launch_date)
    if match:
        if match.group(2):
            return match.group(2)
        elif match.group(3):
            return match.group(3)[-4:]
        elif match.group(4):
            return match.group(4)[-4:]
        elif match.group(5):
            return '20' + match.group(5)[1:]
        elif match.group(6):
            return match.group(6)[:4]
    else:
        return np.nan
    

dataset_intel_nonarc = pd.read_csv('intel_processors.csv', low_memory=False)
dataset_intel_arc =  pd.read_csv('intel_ark_processors.csv', low_memory=False) # Includes both the ARC and normal processors
df_amd =  pd.read_csv('amd_processors.csv', low_memory=False) 
dataset_cpu_benchmark = pd.read_csv('cpu_benchmarks.csv', low_memory=False)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# find the common columns between the two datasets.
common_cols = np.intersect1d(dataset_intel_nonarc.columns, dataset_intel_arc.columns)

# concatenate only the common columns
df_intel = pd.concat([dataset_intel_nonarc[common_cols], dataset_intel_arc[common_cols]], ignore_index=True)

# We only care about a select few columns
df_intel_select = df_intel[["cores", "threads", "name", "base_frequency", "turbo_frequency", "tdp", "memory_type", "max_temp", "sku", "launch_date"]]
df_amd_select = df_amd[["cores", "threads", "name", "base_frequency", "turbo_frequency", "tdp", "memory_type", "max_temp", "sku", "launch_date"]]

## Dropping the values from here
# For the correlation coefficient comparison
df_intel_clean = df_intel_select.dropna(subset=["turbo_frequency", "tdp", "max_temp", "launch_date", "threads", "base_frequency", "cores"])
df_amd_clean = df_amd_select.dropna(subset =["turbo_frequency", "tdp", "max_temp", "launch_date", "threads", "base_frequency", "cores"])

# For CPU/Threads comparison
df_intel_clean_only_cpu_and_threads= df_intel_select.dropna(subset=["cores", "threads", "launch_date"])
df_amd_clean_only_cpu_and_threads = df_amd_select.dropna(subset =["cores", "threads", "launch_date"])

# For Base. Freq and Turbo. Freq comparison
df_intel_clean_only_frequencies= df_intel_select.dropna(subset=["base_frequency", "turbo_frequency", "launch_date"])
df_amd_clean_only_frequencies = df_amd_select.dropna(subset =["base_frequency", "turbo_frequency", "launch_date"])

# For max temperature and TDP comparison
df_intel_clean_only_temp_and_tdp = df_intel_select.dropna(subset=["tdp", "launch_date", "max_temp"])
df_amd_clean_only_temp_and_tdp = df_amd_select.dropna(subset =["tdp", "launch_date", "max_temp"])

#### fill missing values with mean of non-missing values in the same column, note we only do this to numerical columns
## Because this is causing issues with turbo frequency (as it did not exist back then), and because of suggestions, instead the values will be dropped
## Rather than filled with a mean.
#int_columns = ['cores', 'threads', 'tdp', 'max_temp', 'base_frequency', 'turbo_frequency']
#for col in int_columns:
#    mean_value_intel = df_intel_select[col][df_intel_select[col].notna()].mean()
#    mean_value_amd = df_amd_select[col][df_amd_select[col].notna()].mean()
#    df_intel_select[col].fillna(mean_value_intel, inplace=True)
#    df_amd_select[col].fillna(mean_value_amd, inplace=True)
#df_intel_clean = df_intel_select
#df_amd_clean = df_amd_select
# Dropping those, maybe I can get some help with this one
#df_intel_clean = df_intel_clean.dropna(subset=["launch_date"])
#df_amd_clean = df_amd_clean.dropna(subset=["launch_date"])

# Select relevant columns
df_intel_spec = df_intel_clean[['cores', 'threads', 'base_frequency', 'turbo_frequency', "launch_date", "tdp", "max_temp"]]
df_amd_spec = df_amd_clean[['cores', 'threads', 'base_frequency', 'turbo_frequency', "launch_date", "max_temp", "tdp"]]

df_intel_spec_cpu_and_threads = df_intel_clean_only_cpu_and_threads[['cores', 'threads', 'base_frequency', 'turbo_frequency', "launch_date", "tdp", "max_temp"]]
df_amd_spec_cpu_and_threads = df_amd_clean_only_cpu_and_threads[['cores', 'threads', 'base_frequency', 'turbo_frequency', "launch_date", "max_temp", "tdp"]]

df_intel_spec_frequencies = df_intel_clean_only_frequencies[['cores', 'threads', 'base_frequency', 'turbo_frequency', "launch_date", "tdp", "max_temp"]]
df_amd_spec_frequencies = df_amd_clean_only_frequencies[['cores', 'threads', 'base_frequency', 'turbo_frequency', "launch_date", "max_temp", "tdp"]]

df_intel_spec_temp_and_tdp = df_intel_clean_only_temp_and_tdp[['cores', 'threads', 'base_frequency', 'turbo_frequency', "launch_date", "tdp", "max_temp"]]
df_amd_spec_temp_and_tdp = df_amd_clean_only_temp_and_tdp[['cores', 'threads', 'base_frequency', 'turbo_frequency', "launch_date", "max_temp", "tdp"]]

# Extract year from the launch_date column of the corresponding dataset that we'll be analyzing
df_intel_spec['year'] = pd.to_datetime(df_intel_spec['launch_date']).dt.year.astype(str)
df_intel_spec_cpu_and_threads['year'] = pd.to_datetime(df_intel_spec_cpu_and_threads['launch_date']).dt.year.astype(str)
df_intel_spec_frequencies['year'] = pd.to_datetime(df_intel_spec_frequencies['launch_date']).dt.year.astype(str)
df_intel_spec_temp_and_tdp['year'] = pd.to_datetime(df_intel_spec_temp_and_tdp['launch_date']).dt.year.astype(str)

# Initial approach of multiple regex 
year_pattern = re.compile(r'((\d{4})|(\d{2}/(\d{2}/)?\d{4})|(\d{4}-\d{2}-\d{2})|(\d{2}/\d{4}))')
df_amd_spec['year'] = df_amd_spec.apply(year_extraction_for_amd, axis=1) # Applying changes
df_amd_spec = df_amd_spec.dropna(subset=['year']) # Limitation for now
df_amd_spec['year'] = df_amd_spec['year'].astype(int)
df_amd_spec = df_amd_spec[(df_amd_spec['year'] >= 1985) & (df_amd_spec['year'] <= 2023)] # AMD only started manufacturing by itself at 1985. 2 outliers here that are further filtered due to our regex not capturing properly
#print("Count of AMD CPUs with years through the regex %d" % len(df_amd_spec['year'])) # 178 in total

df_amd_spec_cpu_and_threads['year'] = df_amd_spec_cpu_and_threads.apply(year_extraction_for_amd, axis=1)
df_amd_spec_cpu_and_threads = df_amd_spec_cpu_and_threads.dropna(subset=['year'])
df_amd_spec_cpu_and_threads['year'] = df_amd_spec_cpu_and_threads['year'].astype(int)
df_amd_spec_cpu_and_threads = df_amd_spec_cpu_and_threads[(df_amd_spec_cpu_and_threads['year'] >= 1985) & (df_amd_spec_cpu_and_threads['year'] <= 2023)] 

df_amd_spec_frequencies['year'] = df_amd_spec_frequencies.apply(year_extraction_for_amd, axis=1)
df_amd_spec_frequencies = df_amd_spec_frequencies.dropna(subset=['year'])
df_amd_spec_frequencies['year'] = df_amd_spec_frequencies['year'].astype(int)
df_amd_spec_frequencies = df_amd_spec_frequencies[(df_amd_spec_frequencies['year'] >= 1985) & (df_amd_spec_frequencies['year'] <= 2023)] 

df_amd_spec_temp_and_tdp['year'] = df_amd_spec_temp_and_tdp.apply(year_extraction_for_amd, axis=1)
df_amd_spec_temp_and_tdp = df_amd_spec_temp_and_tdp.dropna(subset=['year'])
df_amd_spec_temp_and_tdp['year'] = df_amd_spec_temp_and_tdp['year'].astype(int)
df_amd_spec_temp_and_tdp = df_amd_spec_temp_and_tdp[(df_amd_spec_temp_and_tdp['year'] >= 1985) & (df_amd_spec_temp_and_tdp['year'] <= 2023)] 


# Converting the year column to numeric type
df_intel_spec['year'] = pd.to_numeric(df_intel_spec['year'])
df_amd_spec['year'] = pd.to_numeric(df_amd_spec['year'])
df_intel_spec_cpu_and_threads['year'] = pd.to_numeric(df_intel_spec_cpu_and_threads['year'])
df_amd_spec_cpu_and_threads['year'] = pd.to_numeric(df_amd_spec_cpu_and_threads['year'])
df_intel_spec_frequencies['year'] = pd.to_numeric(df_intel_spec_frequencies['year'])
df_amd_spec_frequencies['year'] = pd.to_numeric(df_amd_spec_frequencies['year'])
df_intel_spec_temp_and_tdp['year'] = pd.to_numeric(df_intel_spec_temp_and_tdp['year'])
df_amd_spec_temp_and_tdp['year'] = pd.to_numeric(df_amd_spec_temp_and_tdp['year'])

# For the general case
general_processors = pd.concat([df_amd_spec, df_intel_spec])
general_cpu_and_threads = pd.concat([df_amd_spec_cpu_and_threads, df_intel_spec_cpu_and_threads])
general_frequencies = pd.concat([df_amd_spec_frequencies, df_intel_spec_frequencies])
general_temp_and_tdp = pd.concat([df_amd_spec_temp_and_tdp, df_intel_spec_temp_and_tdp])


# Mean per year
intel_mean_spec_per_year = df_intel_spec.groupby('year').mean().reset_index()
amd_mean_spec_per_year = df_amd_spec.groupby('year').mean().reset_index()

df_intel_spec_cpu_and_threads_mean_per_year = df_intel_spec_cpu_and_threads.groupby('year').mean().reset_index()
df_amd_spec_cpu_and_threads_mean_per_year = df_amd_spec_cpu_and_threads.groupby('year').mean().reset_index()
df_intel_spec_frequencies_mean_per_year = df_intel_spec_frequencies.groupby('year').mean().reset_index()
df_amd_spec_frequencies_mean_per_year = df_amd_spec_frequencies.groupby('year').mean().reset_index()
df_intel_spec_temp_and_tdp_mean_per_year = df_intel_spec_temp_and_tdp.groupby('year').mean().reset_index()
df_amd_spec_temp_and_tdp_mean_per_year = df_amd_spec_temp_and_tdp.groupby('year').mean().reset_index()

general_processors_mean_per_year = general_processors.groupby('year').mean().reset_index()

general_cpu_and_threads_mean_per_year = general_cpu_and_threads.groupby('year').mean().reset_index()
general_frequencies_mean_per_year = general_frequencies.groupby('year').mean().reset_index()
general_temp_and_tdp_per_year = general_temp_and_tdp.groupby('year').mean().reset_index()

# Summary statistics for all years
#intel_mean_spec_overall = df_intel_spec.mean()
#amd_mean_spec_overall = df_amd_spec.mean()
#general_processors_mean_overall = general_processors.mean()
#
#intel_median_spec_overall = df_intel_spec.median()
#amd_median_spec_overall = df_amd_spec.median()
#general_processors_median_overall = general_processors.median()
#
#intel_mode_spec_overall = df_intel_spec.mode()
#amd_mode_spec_overall = df_amd_spec.mode()
#general_processors_mode_overall = general_processors.mode()
#
#intel_max_spec_overall = df_intel_spec.max()
#amd_max_spec_overall = df_amd_spec.max()
#general_processors_max_overall = general_processors.max()
#
#intel_min_spec_overall = df_intel_spec.min()
#amd_min_spec_overall = df_amd_spec.min()
#general_processors_min_overall = general_processors.min()
#
#intel_std_spec_overall = df_intel_spec.std()
#amd_std_spec_overall = df_amd_spec.std()
#general_processors_std_overall = general_processors.std()

### General case 
# Some conversions
general_processors['base_frequency'] = general_processors['base_frequency'].astype(int)
general_processors['turbo_frequency'] = general_processors['turbo_frequency'].astype(int)
general_processors['tdp'] = general_processors['tdp'].astype(int)
general_processors['max_temp'] = general_processors['max_temp'].astype(int)
general_processors['cores'] = general_processors['cores'].astype(int)
general_processors['threads'] = general_processors['threads'].astype(int)
# Convert columns to numeric type
general_processors['cores'] = pd.to_numeric(general_processors['cores'])

# Some color blind friendly colors are
# Blue and orange
# Purple and yellow
# Green and pink
# Red and light blue
# Teal and coral
# We'll define colors for each paramater
colorblind_friendly = ['#0072B2', '#009E73', '#D55E00', '#CC79A7', '#F0E442']

colors_general_bt = {'base_frequency': 'blue', 'turbo_frequency': 'orange'}
palette_general_bt = sns.color_palette([colors_general_bt[param] for param in colors_general_bt.keys()])

colors_general_ct = {'cores': 'teal', 'threads': 'coral'}
palette_general_ct = sns.color_palette([colors_general_ct[param] for param in colors_general_ct.keys()])

## General trend of base cores and threads
figg1 = plt.figure(figsize=(10, 6))
ax = figg1.subplots()
sns.lineplot(x='year', y='cores', data=general_cpu_and_threads.reset_index(), label='Cores', color='teal')
sns.lineplot(x='year', y='threads', data=general_cpu_and_threads.reset_index(), label='Threads', color='coral')
plt.xlabel('Year')
plt.ylabel('Number of Cores/Threads')
plt.title('Comparison of Cores and Threads by Year')
plt.savefig('Kilali_Faycal_gen1.png', dpi=300) 

## Distribution of cores and threads

sns.catplot(x="year", y="value", hue="variable", kind = 'bar', data=general_cpu_and_threads.melt(id_vars='year', value_vars=['cores', 'threads']), ax=ax, palette=palette_general_ct, legend= False)
cores_patch = mpatches.Patch(color='teal', label='Cores')
threads_patch = mpatches.Patch(color='coral', label='Threads')
plt.legend(handles=[cores_patch, threads_patch])
plt.title('Mean Cores and Threads per processor by Year')
plt.xlabel('Year')
plt.ylabel('Mean Number of Cores/Threads')
plt.savefig('Kilali_Faycal_gen2.png', dpi=300) 


# General trend of base frequency and turbo frequency
figg3 = plt.figure(figsize=(10, 6))
ax = figg3.subplots()
sns.lineplot(x='year', y='base_frequency', data=general_frequencies.reset_index(), label='Base Frequency')
sns.lineplot(x='year', y='turbo_frequency', data=general_frequencies.reset_index(), label='Turbo Frequency')

plt.xlabel('Year')
plt.ylabel('Clock Speed (GHz)')
plt.title('Comparison of Base Clocks and Turbo Clocks by Year')
plt.savefig('Kilali_Faycal_gen3.png', dpi=300) 

## Distribution of base frequency and turbo frequency
figg4 = plt.figure(figsize=(10, 6))
ax = figg4.subplots()
sns.violinplot(x="year", y="value", hue="variable", data=general_frequencies.melt(id_vars='year', value_vars=['base_frequency', 'turbo_frequency']), split=True, ax=ax, palette=palette_general_bt)
base_freq_patch = mpatches.Patch(color='blue', label='Base Frequency')
turbo_freq_patch = mpatches.Patch(color='orange', label='Turbo Frequency')
plt.legend(handles=[base_freq_patch, turbo_freq_patch])
plt.title('Base and Turbo Frequencies by Year')
plt.xlabel('Year')
plt.ylabel('Clock speed (GHz)')
plt.savefig('Kilali_Faycal_gen4.png', dpi=300) 

# General trend of Max temperature and TDP
figg5 = plt.figure(figsize=(10, 6))
ax = figg5.subplots()

sns.lineplot(x='year', y='max_temp', data=general_temp_and_tdp.reset_index(), label='Max temperature (Celsius)')
sns.lineplot(x='year', y='tdp', data=general_temp_and_tdp.reset_index(), label='Thermal Design Power (Watts)')
plt.xlabel('Year')
plt.ylabel('Value')
plt.title('Comparison of Max Temperature & TDP')
plt.savefig('Kilali_Faycal_gen5.png', dpi=300) 

## Distribution of TDP and Max Temperature
figg6 = plt.figure(figsize=(10, 6))
ax = figg6.subplots()
sns.barplot(x="year", y="value", hue="variable", data=general_temp_and_tdp.reset_index().melt(id_vars='year', value_vars=['max_temp', 'tdp']), ax=ax, palette=palette_general_bt)
base_freq_patch = mpatches.Patch(color='blue', label='Max Temperature (Celsius)')
turbo_freq_patch = mpatches.Patch(color='orange', label='Thermal Design Power (Watts)')
plt.legend(handles=[base_freq_patch, turbo_freq_patch])
plt.title('Max temperature and Thermal Design Power by Year')
plt.xlabel('Year')
plt.ylabel('Value')
plt.savefig('Kilali_Faycal_gen6.png', dpi=300) 

# The correlation, we can see some strong correlations here.
corr = general_processors[['cores', 'threads', 'base_frequency', 'turbo_frequency', 'tdp', 'max_temp']].corr()
sns.set(font_scale=1.2)
figg8, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, cbar_kws={'shrink': .8}, square=True, ax=ax,
            xticklabels=['Cores', 'Threads', 'Base Frequency', 'Turbo Frequency', 'TDP', 'Max Temp'], 
            yticklabels=['Cores', 'Threads', 'Base Frequency', 'Turbo Frequency', 'TDP', 'Max Temp'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right')
ax.set_title('Correlation Matrix for General Processors of Intel and AMD', fontsize=18)
plt.tight_layout() # Making the plots tighter so they can fit in nicely.
plt.savefig('Kilali_Faycal_gen8.png', dpi=300)

# Now lets do some comparison between the companies
# Before we do so, we'll do the t-tests so we can annotate the plots for easier readability
#Finding the common years, of course ensuring there are no duplicates
# NOTE: improve the names of the variables
common_years = set(intel_mean_spec_per_year['year']).intersection(set(amd_mean_spec_per_year['year']))
intel_common_years = intel_mean_spec_per_year[intel_mean_spec_per_year['year'].isin(common_years)]
amd_common_years = amd_mean_spec_per_year[amd_mean_spec_per_year['year'].isin(common_years)]

common_years_cpu_and_threads = set(df_intel_spec_cpu_and_threads_mean_per_year['year']).intersection(set(df_amd_spec_cpu_and_threads_mean_per_year['year']))
intel_common_years_cpu_and_threads = df_intel_spec_cpu_and_threads_mean_per_year[df_intel_spec_cpu_and_threads_mean_per_year['year'].isin(common_years_cpu_and_threads)]
amd_common_years_cpu_and_threads = df_amd_spec_cpu_and_threads_mean_per_year[df_amd_spec_cpu_and_threads_mean_per_year['year'].isin(common_years_cpu_and_threads)]

common_years_frequencies = set(df_intel_spec_frequencies_mean_per_year['year']).intersection(set(df_amd_spec_frequencies_mean_per_year['year']))
intel_common_years_frequencies = df_intel_spec_frequencies_mean_per_year[df_intel_spec_frequencies_mean_per_year['year'].isin(common_years_frequencies)]
amd_common_years_frequencies = df_amd_spec_frequencies_mean_per_year[df_amd_spec_frequencies_mean_per_year['year'].isin(common_years_frequencies)]

common_years_temp_and_tdp = set(df_intel_spec_temp_and_tdp_mean_per_year['year']).intersection(set(df_amd_spec_temp_and_tdp_mean_per_year['year']))
intel_common_years_temp_and_tdp = df_intel_spec_temp_and_tdp_mean_per_year[df_intel_spec_temp_and_tdp_mean_per_year['year'].isin(common_years_temp_and_tdp)]
amd_common_years_temp_and_tdp = df_amd_spec_temp_and_tdp_mean_per_year[df_amd_spec_temp_and_tdp_mean_per_year['year'].isin(common_years_temp_and_tdp)]

# T-testing stuff and p-values

# Extract the base frequency and turbo frequency data for each uhh column
intel_cores = intel_common_years_cpu_and_threads['cores']
amd_cores = amd_common_years_cpu_and_threads['cores']
intel_threads = intel_common_years_cpu_and_threads['threads']
amd_turbo_threads = amd_common_years_cpu_and_threads['threads']
intel_base_freq = intel_common_years_frequencies['base_frequency']
amd_base_freq = amd_common_years_frequencies['base_frequency']
intel_turbo_freq = intel_common_years_frequencies['turbo_frequency']
amd_turbo_freq = amd_common_years_frequencies['turbo_frequency']

intel_max_temp = df_intel_spec_temp_and_tdp['max_temp']
amd_max_temp = df_amd_spec_temp_and_tdp['max_temp']
intel_tdp = df_intel_spec_temp_and_tdp['tdp']
amd_tdp = df_amd_spec_temp_and_tdp['tdp']

# t-tests for the attributes we care enough about to compare
base_freq_ttest = ttest_ind(intel_base_freq, amd_base_freq)
turbo_freq_ttest = ttest_ind(intel_turbo_freq, amd_turbo_freq)
cores_ttest = ttest_ind(intel_cores, amd_cores)
threads_ttest = ttest_ind(intel_threads, amd_turbo_threads)
max_temp_ttest = ttest_ind(intel_max_temp, amd_max_temp)
tdp_ttest = ttest_ind(intel_tdp, amd_tdp)



### Cores/Threads and BF/TF graphs + distributions between AMD and Intel
sns.set_style("whitegrid")

fig, axs = plt.subplots(1, 2, figsize=(12, 6))


# Plotting as the first subplot the threads and cores data
axs[0].plot(intel_common_years_cpu_and_threads['year'], intel_common_years_cpu_and_threads['cores'], label='Intel cores')
axs[0].plot(amd_common_years_cpu_and_threads['year'], amd_common_years_cpu_and_threads['cores'], label='AMD cores')
axs[0].plot(intel_common_years_cpu_and_threads['year'], intel_common_years_cpu_and_threads['threads'], label='Intel threads')
axs[0].plot(amd_common_years_cpu_and_threads['year'], amd_common_years_cpu_and_threads['threads'], label='AMD threads')
axs[0].set_xlabel('Year')
axs[0].set_ylabel('Average number of cores/threads')
axs[0].set_title('Mean number of cores/threads of Intel and AMD')
axs[0].legend()

# We'll plot the base frequency and turbo frequency data in the second subplot
axs[1].plot(intel_common_years_frequencies['year'], intel_common_years_frequencies['base_frequency'], label='Intel base frequency')
axs[1].plot(amd_common_years_frequencies['year'], amd_common_years_frequencies['base_frequency'], label='AMD base frequency')
axs[1].plot(intel_common_years_frequencies['year'], intel_common_years_frequencies['turbo_frequency'], label='Intel turbo frequency')
axs[1].plot(amd_common_years_frequencies['year'], amd_common_years_frequencies['turbo_frequency'], label='AMD turbo frequency')
axs[1].set_xlabel('Year')
axs[1].set_ylabel('Average clock speed (GHz)')
axs[1].set_title('Mean base and turbo clock speed of Intel and AMD')
axs[1].legend()

axs[0].annotate(f'Cores t-value: {cores_ttest.statistic:.2f}\n p-value: {cores_ttest.pvalue:.2e}\n\n'
                 f'Threads t-value: {threads_ttest.statistic:.2f}\n p-value: {threads_ttest.pvalue:.2e}', 
                 xy=(0.05, 0.65), xycoords='axes fraction', fontsize=12, ha='left', va='top', color=colorblind_friendly[0])
axs[1].annotate(f'Base frequency t-value: {base_freq_ttest.statistic:.2f}\n p-value: {base_freq_ttest.pvalue:.2e}\n\n'
                 f'Turbo frequency t-value: {turbo_freq_ttest.statistic:.2f}\n p-value: {turbo_freq_ttest.pvalue:.2e}', 
                 xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, ha='left', va='top', color=colorblind_friendly[1])

plt.savefig('Kilali_Faycal_0.png', dpi=300)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

## DISTRIBUTION PLOT OF CORES/THREADS

# Cores and threads subplot
sns.histplot(data=intel_common_years_cpu_and_threads, x="cores", ax=axs[0], color='blue', alpha=0.5, label="Intel Cores")
sns.histplot(data=amd_common_years_cpu_and_threads, x="cores", ax=axs[0], color='orange', alpha=0.5, label="AMD Cores")
sns.histplot(data=intel_common_years_cpu_and_threads, x="threads", ax=axs[0], color='blue', alpha=0.3, label="Intel Threads", kde=True)
sns.histplot(data=amd_common_years_cpu_and_threads, x="threads", ax=axs[0], color='orange', alpha=0.3, label="AMD Threads", kde=True)
axs[0].set_xlabel('Number of cores and threads')
axs[0].set_ylabel('Density')
axs[0].set_title('Distribution of cores and threads of Intel and AMD')
axs[0].legend()

# Base and turbo frequency subplot
sns.kdeplot(data=intel_common_years_frequencies, x = "base_frequency", ax=axs[1], color='blue', alpha=0.5, label="Intel Base Frequency")
sns.kdeplot(data=amd_common_years_frequencies, x = "base_frequency", ax=axs[1], color='orange', alpha=0.5, label="AMD Base Frequency")
sns.kdeplot(data=intel_common_years_frequencies, x = "turbo_frequency", ax=axs[1], color='blue', alpha=0.3, label="Intel Turbo Frequency", linestyle='--')
sns.kdeplot(data=amd_common_years_frequencies, x = "turbo_frequency", ax=axs[1], color='orange', alpha=0.3, label="AMD Turbo Frequency", linestyle='--')
axs[1].set_xlabel('Clock speed (GHz)')
axs[1].set_ylabel('Density')
axs[1].set_title('Distribution of base and turbo clock speed of Intel and AMD')
axs[1].legend()
plt.tight_layout()
plt.savefig('Kilali_Faycal_1.png', dpi=300)

### Max_temp/TDP and Distribution for it between AMD and Intel
sns.set_style("whitegrid") # Kind of a fan of this one
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plotting the max_temp and TDP data columns in the first subplot of the figure
axs[0].plot(intel_common_years_temp_and_tdp['year'], intel_common_years_temp_and_tdp['max_temp'], label='Intel Max Temperature')
axs[0].plot(amd_common_years_temp_and_tdp['year'], amd_common_years_temp_and_tdp['max_temp'], label='AMD Max Temperature')
axs[0].plot(intel_common_years_temp_and_tdp['year'], intel_common_years_temp_and_tdp['tdp'], label='Intel TDP')
axs[0].plot(amd_common_years_temp_and_tdp['year'], amd_common_years_temp_and_tdp['tdp'], label='AMD TDP')
axs[0].set_xlabel('Year')
axs[0].set_ylabel('Temperature (C) / TDP (W)')
axs[0].set_title('Max Temperature and TDP of Intel and AMD')
axs[0].legend()

axs[0].annotate(f'Max temp t-value: {max_temp_ttest.statistic:.2f}\n p-value: {max_temp_ttest.pvalue:.2e}\n\n'
                 f'TDP t-value: {tdp_ttest.statistic:.2f}\n p-value: {tdp_ttest.pvalue:.2e}', 
                 xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, ha='left', va='top', color=colorblind_friendly[2])

# Generating a distribution plot of the max_temp and TDP columns in the second subplot within the same figure
sns.distplot(intel_common_years_temp_and_tdp['max_temp'], ax=axs[1], label='Intel Max temperature')
sns.distplot(amd_common_years_temp_and_tdp['max_temp'], ax=axs[1], label='AMD Max Temperature')
sns.distplot(intel_common_years_temp_and_tdp['tdp'], ax=axs[1], label='Intel TDP')
sns.distplot(amd_common_years_temp_and_tdp['tdp'], ax=axs[1], label='AMD TDP')
axs[1].set_xlabel('Temperature (C) / TDP (W)')
axs[1].set_ylabel('Density')
axs[1].set_title('Distribution of Max Temperature and TDP of Intel and AMD')
axs[1].legend()


plt.tight_layout()
plt.savefig('Kilali_Faycal_2.png', dpi=300)


# Correlations

# Compute correlation coefficients for AMD and Intel
amd_corr = amd_common_years[['cores', 'threads', 'base_frequency', 'turbo_frequency', 'tdp', 'max_temp']].corr()
intel_corr = intel_common_years[['cores', 'threads', 'base_frequency', 'turbo_frequency', 'tdp', 'max_temp']].corr()

# Create a figure with two subplots, again.
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot heatmap of correlation coefficients for AMD
sns.heatmap(amd_corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, cbar=False, square=True, ax=axs[0],
            xticklabels=['Cores', 'Threads', 'Base Freq.', 'Turbo Freq.', 'TDP', 'Max Temp'], 
            yticklabels=['Cores', 'Threads', 'Base Freq.', 'Turbo Freq.', 'TDP', 'Max Temp'])
axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=20, ha='right')
axs[0].set_yticklabels(axs[0].get_yticklabels(), rotation=0, ha='right')
axs[0].set_title("AMD's correlation coefficients", fontsize=14)


# Plot heatmap of correlation coefficients for Intel
sns.heatmap(intel_corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, cbar=False, square=True, ax=axs[1],
            xticklabels=['Cores', 'Threads', 'Base Freq.', 'Turbo Freq.', 'TDP', 'Max Temp'], 
            yticklabels=['Cores', 'Threads', 'Base Freq.', 'Turbo Freq.', 'TDP', 'Max Temp'])
axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=20, ha='right')
axs[1].set_yticklabels(axs[1].get_yticklabels(), rotation=0, ha='right')
axs[1].set_title("Intel's correlation coefficients", fontsize=14)


plt.tight_layout()
plt.savefig('Kilali_Faycal_3.png', dpi=300)


# Predictions
future_years = np.arange(intel_common_years_temp_and_tdp['year'].max() + 1, intel_common_years_temp_and_tdp['year'].max() + 11)


# General Predictions

# Predicted number of threads for general processors
x = np.array(general_cpu_and_threads.index)
y = np.array(general_cpu_and_threads['threads'])
params = Parameters()
params.add('a0', value=0)
params.add('a1', value=0)
result = minimize(least_squares, params, args=(x, y))
params = result.params
future_predictions_general_threads = linear_model(future_years, params)

# Predicted number of cores for general processors
y = np.array(general_cpu_and_threads['cores'])
result = minimize(least_squares, params, args=(x, y))
params = result.params
future_predictions_general_cores = linear_model(future_years, params)

# Predicted max temperature for General Processors (that is, both Intel and AMD treated as one)
x = np.array(general_temp_and_tdp['year'])
y = np.array(general_temp_and_tdp['max_temp'])
params = Parameters()
params.add('a0', value=0)
params.add('a1', value=0)
result = minimize(least_squares, params, args=(x, y))
params = result.params
future_predictions_general_temp = linear_model(future_years, params)

# Predicted TDP for General Processors
x = np.array(general_temp_and_tdp['year'])
y = np.array(general_temp_and_tdp['tdp'])
params = Parameters()
params.add('a0', value=0)
params.add('a1', value=0)
result = minimize(least_squares, params, args=(x, y))
params = result.params
future_predictions_general_TDP = linear_model(future_years, params)

# Predicted Base Frequency for General Processors
x = np.array(general_frequencies['year'])
y = np.array(general_frequencies['base_frequency'])
params = Parameters()
params.add('a0', value=0)
params.add('a1', value=0)
result = minimize(least_squares, params, args=(x, y))
params = result.params
future_predictions_general_base_freq = linear_model(future_years, params)

# Predicted Turbo Frequency for General Processors
x = np.array(general_frequencies['year'])
y = np.array(general_frequencies['turbo_frequency'])
params = Parameters()
params.add('a0', value=0)
params.add('a1', value=0)
result = minimize(least_squares, params, args=(x, y))
params = result.params
future_predictions_general_turbo_freq = linear_model(future_years, params)


### General trend of base cores and threads
#figg1 = plt.figure(figsize=(10, 6))
#ax = figg1.subplots()
#sns.lineplot(x='year', y='cores', data=general_cpu_and_threads.reset_index(), label='Cores', color='teal')
#sns.lineplot(x='year', y='threads', data=general_cpu_and_threads.reset_index(), label='Threads', color='coral')
#sns.lineplot(x=future_years, y=future_predictions_general_cores, label="Predicted cores", color=colorblind_friendly[0])
#sns.lineplot(x=future_years, y=future_predictions_general_threads, label="Predicted threads", color=colorblind_friendly[1])
#plt.xlabel('Year')
#plt.ylabel('Number of Cores/Threads')
#plt.title('Comparison of Cores and Threads by Year')
#xmin, xmax = general_cpu_and_threads['year'].min(), general_cpu_and_threads['year'].max() + 10
#plt.xlim(xmin, xmax)
#plt.savefig('Kilali_Faycal_gen1.png') 
#
### Distribution of cores and threads
#
#sns.catplot(x="year", y="value", hue="variable", kind = 'bar', data=general_cpu_and_threads.melt(id_vars='year', value_vars=['cores', 'threads']), ax=ax, palette=palette_general_ct, legend= False)
#cores_patch = mpatches.Patch(color='teal', label='Cores')
#threads_patch = mpatches.Patch(color='coral', label='Threads')
#plt.legend(handles=[cores_patch, threads_patch])
#sns.lineplot(x=future_years, y=future_predictions_general_cores, label="Predicted cores", color=colorblind_friendly[0])
#sns.lineplot(x=future_years, y=future_predictions_general_threads, label="Predicted threads", color=colorblind_friendly[1])
#plt.title('Mean Cores and Threads per processor by Year')
#plt.xlabel('Year')
#plt.ylabel('Mean Number of Cores/Threads')
#xmin, xmax = general_cpu_and_threads['year'].min(), general_cpu_and_threads['year'].max() + 10
#plt.xlim(xmin, xmax)
#plt.savefig('Kilali_Faycal_gen2.png')


# Comparison predictions


fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Predicted max temperature for Intel
x = np.array(intel_common_years_temp_and_tdp['year'])
y = np.array(intel_common_years_temp_and_tdp['max_temp'])
params = Parameters()
params.add('a0', value=0)
params.add('a1', value=0)
result = minimize(least_squares, params, args=(x, y))
params = result.params
future_predictions_intel_temp = linear_model(future_years, params)

# Predicted max temperature for AMD
x = np.array(amd_common_years_temp_and_tdp['year'])
y = np.array(amd_common_years_temp_and_tdp['max_temp'])
params = Parameters()
params.add('a0', value=0)
params.add('a1', value=0)
result = minimize(least_squares, params, args=(x, y))
params = result.params
future_predictions_amd_temp = linear_model(future_years, params)

# Predicted TDP for Intel
x = np.array(intel_common_years_temp_and_tdp['year'])
y = np.array(intel_common_years_temp_and_tdp['tdp'])
params = Parameters()
params.add('a0', value=0)
params.add('a1', value=0)
result = minimize(least_squares, params, args=(x, y))
params = result.params
future_predictions_intel_tdp = linear_model(future_years, params)

# Predicted TDP for AMD
x = np.array(amd_common_years_temp_and_tdp['year'])
y = np.array(amd_common_years_temp_and_tdp['tdp'])
params = Parameters()
params.add('a0', value=0)
params.add('a1', value=0)
result = minimize(least_squares, params, args=(x, y))
params = result.params
future_predictions_amd_tdp = linear_model(future_years, params)

# Plotting time
axs[0].plot(intel_common_years_temp_and_tdp['year'], intel_common_years_temp_and_tdp['max_temp'], 'o', color=colorblind_friendly[0], label="Intel's actual mean max temperature")
axs[0].plot(amd_common_years_temp_and_tdp['year'], amd_common_years_temp_and_tdp['max_temp'], 'o', color=colorblind_friendly[1], label="AMD's actual mean max temperature")
axs[0].plot(future_years, future_predictions_intel_temp, color=colorblind_friendly[0], label="Intel's predicted mean max temperature")
axs[0].plot(future_years, future_predictions_amd_temp, color=colorblind_friendly[1], label="AMD's predicted mean max temperature")
axs[0].plot(intel_common_years_temp_and_tdp['year'], intel_common_years_temp_and_tdp['tdp'], 's', color=colorblind_friendly[2], label="Intel's actual mean TDP")
axs[0].plot(amd_common_years_temp_and_tdp['year'], amd_common_years_temp_and_tdp['tdp'], 's', color=colorblind_friendly[3], label="AMD's actual mean TDP")
axs[0].plot(future_years, future_predictions_intel_tdp, '--', color=colorblind_friendly[2], label="Intel's predicted mean TDP")
axs[0].plot(future_years, future_predictions_amd_tdp, '--', color=colorblind_friendly[3], label="AMD's predicted mean TDP")
axs[0].set_xlabel('Year')
axs[0].set_ylabel('Temperature (C) / TDP (W)')
axs[0].set_title('Max Temperature and TDP of Intel and AMD')

# The distribution plots
sns.kdeplot(intel_common_years_temp_and_tdp['max_temp'], ax=axs[1], label="Intel's actual mean max temperature", color=colorblind_friendly[0])
sns.kdeplot(future_predictions_intel_temp, ax=axs[1], label="Intel's predicted mean max temperature", linestyle='--', color=colorblind_friendly[0])
sns.kdeplot(amd_common_years_temp_and_tdp['max_temp'], ax=axs[1], label="AMD's actual mean max temperature", color=colorblind_friendly[1])
sns.kdeplot(future_predictions_amd_temp, ax=axs[1], label="AMD's predicted mean max temperature", linestyle='--', color=colorblind_friendly[1])
sns.kdeplot(intel_common_years_temp_and_tdp['tdp'], ax=axs[1], label="Intel's actual mean TDP", color=colorblind_friendly[2])
sns.kdeplot(future_predictions_intel_tdp, ax=axs[1], label="Intel's predicted mean TDP", linestyle='--', color=colorblind_friendly[2])
sns.kdeplot(amd_common_years_temp_and_tdp['tdp'], ax=axs[1], label="AMD's actual mean TDP", color=colorblind_friendly[3])
sns.kdeplot(future_predictions_amd_tdp, ax=axs[1], label="AMD's predicted mean TDP", linestyle='--', color=colorblind_friendly[3])

axs[1].set_xlabel('Temperature (C) / TDP (W)')
axs[1].set_ylabel('Density')
axs[1].set_title('Distribution of mean max temperature & mean TDP')
axs[0].legend(fontsize=8)
axs[1].legend(fontsize=8)

plt.savefig('Kilali_Faycal_4.png', dpi=300)


## Next plot
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Predicted threads for Intel
x = np.array(intel_common_years_cpu_and_threads['year'])
y = np.array(intel_common_years_cpu_and_threads['threads'])
params = Parameters()
params.add('a0', value=0)
params.add('a1', value=0)
result = minimize(least_squares, params, args=(x, y))
params = result.params
future_predictions_intel_threads = linear_model(future_years, params)

# Predicted threads for AMD
x = np.array(amd_common_years_cpu_and_threads['year'])
y = np.array(amd_common_years_cpu_and_threads['threads'])
params = Parameters()
params.add('a0', value=0)
params.add('a1', value=0)
result = minimize(least_squares, params, args=(x, y))
params = result.params
future_predictions_amd_threads = linear_model(future_years, params)

# Predicted cores for Intel
x = np.array(intel_common_years_cpu_and_threads['year'])
y = np.array(intel_common_years_cpu_and_threads['cores'])
params = Parameters()
params.add('a0', value=0)
params.add('a1', value=0)
result = minimize(least_squares, params, args=(x, y))
params = result.params
future_predictions_intel_cores = linear_model(future_years, params)

# Predicted cores for AMD
x = np.array(amd_common_years_cpu_and_threads['year'])
y = np.array(amd_common_years_cpu_and_threads['cores'])
params = Parameters()
params.add('a0', value=0)
params.add('a1', value=0)
result = minimize(least_squares, params, args=(x, y))
params = result.params
future_predictions_amd_cores = linear_model(future_years, params)


# Predicted base frequency for Intel
x = np.array(intel_common_years_frequencies['year'])
y = np.array(intel_common_years_frequencies['base_frequency'])
params = Parameters()
params.add('a0', value=0)
params.add('a1', value=0)
result = minimize(least_squares, params, args=(x, y))
params = result.params
future_predictions_intel_base_freq = linear_model(future_years, params)

# Predicted base frequency for AMD
x = np.array(amd_common_years_frequencies['year'])
y = np.array(amd_common_years_frequencies['base_frequency'])
params = Parameters()
params.add('a0', value=0)
params.add('a1', value=0)
result = minimize(least_squares, params, args=(x, y))
params = result.params
future_predictions_amd_base_freq = linear_model(future_years, params)

# Predicted turbo frequency for Intel
x = np.array(intel_common_years_frequencies['year'])
y = np.array(intel_common_years_frequencies['turbo_frequency'])
params = Parameters()
params.add('a0', value=0)
params.add('a1', value=0)
result = minimize(least_squares, params, args=(x, y))
params = result.params
future_predictions_intel_turbo_freq = linear_model(future_years, params)

# Predicted turbo frequency for AMD
x = np.array(amd_common_years_frequencies['year'])
y = np.array(amd_common_years_frequencies['turbo_frequency'])
params = Parameters()
params.add('a0', value=0)
params.add('a1', value=0)
result = minimize(least_squares, params, args=(x, y))
params = result.params
future_predictions_amd_turbo_freq = linear_model(future_years, params)



axs[0].plot(intel_common_years_cpu_and_threads['year'], intel_common_years_cpu_and_threads['threads'], 'o', color=colorblind_friendly[0], label="Intel's actual mean threads")
axs[0].plot(amd_common_years_cpu_and_threads['year'], amd_common_years_cpu_and_threads['threads'], 'o', color=colorblind_friendly[1], label="AMD's actual mean threads")
axs[0].plot(future_years, future_predictions_intel_threads, color=colorblind_friendly[0], label="Intel's predicted mean threads")
axs[0].plot(future_years, future_predictions_amd_threads, color=colorblind_friendly[1], label="AMD's predicted mean threads")
axs[0].plot(intel_common_years_cpu_and_threads['year'], intel_common_years_cpu_and_threads['cores'], 's', color=colorblind_friendly[2], label="Intel's actual mean cores")
axs[0].plot(amd_common_years_cpu_and_threads['year'], amd_common_years_cpu_and_threads['cores'], 's', color=colorblind_friendly[3], label="AMD's actual mean cores")
axs[0].plot(future_years, future_predictions_intel_cores, '--', color=colorblind_friendly[2], label="Intel's predicted mean cores")
axs[0].plot(future_years, future_predictions_amd_cores, '--', color=colorblind_friendly[3], label="AMD's predicted mean cores")
axs[0].set_xlabel('Year')
axs[0].set_ylabel('Threads / Cores')
axs[0].set_title('Mean Cores and Threads of Intel and AMD')


axs[1].plot(intel_common_years_frequencies['year'], intel_common_years_frequencies['base_frequency'], 'o', color=colorblind_friendly[0], label="Intel's actual base frequency")
axs[1].plot(amd_common_years_frequencies['year'], amd_common_years_frequencies['base_frequency'], 'o', color=colorblind_friendly[1], label="AMD's actual base frequency")
axs[1].plot(future_years, future_predictions_intel_base_freq, color=colorblind_friendly[0], label="Intel's predicted base frequency")
axs[1].plot(future_years, future_predictions_amd_base_freq, color=colorblind_friendly[1], label="AMD's predicted base frequency")
axs[1].plot(intel_common_years_frequencies['year'], intel_common_years_frequencies['turbo_frequency'], 's', color=colorblind_friendly[2], label="Intel's actual turbo frequency")
axs[1].plot(amd_common_years_frequencies['year'], amd_common_years_frequencies['turbo_frequency'], 's', color=colorblind_friendly[3], label="AMD's actual turbo frequency")
axs[1].plot(future_years, future_predictions_intel_turbo_freq, '--', color=colorblind_friendly[2], label="Intel's predicted turbo frequency")
axs[1].plot(future_years, future_predictions_amd_turbo_freq, '--', color=colorblind_friendly[3], label="AMD's predicted turbo frequency")
axs[1].set_xlabel('Year')
axs[1].set_ylabel('Clock speed (GHz)')
axs[0].legend(fontsize=8)
axs[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig('Kilali_Faycal_5.png', dpi=300)



## Now for the distributions, as a separate graph, for some reason this is giving me issues so we'll skip it
#fig, axs = plt.subplots(1, 2, figsize=(12, 6))
#
#
## Cores and threads subplot
#sns.histplot(data=intel_common_years_cpu_and_threads, x="cores", ax=axs[0], color='blue', alpha=0.5, label="Intel Cores")
#sns.histplot(data=amd_common_years_cpu_and_threads, x="cores", ax=axs[0], color='orange', alpha=0.5, label="AMD Cores")
#axs[0].plot(future_years, future_predictions_intel_cores, color=colorblind_friendly[0], label="Intel's predicted cores")
#axs[0].plot(future_years, future_predictions_amd_cores, color=colorblind_friendly[1], label="AMD's predicted cores")
#sns.histplot(data=intel_common_years_cpu_and_threads, x="threads", ax=axs[0], color='blue', alpha=0.3, label="Intel Threads", kde=True)
#sns.histplot(data=amd_common_years_cpu_and_threads, x="threads", ax=axs[0], color='orange', alpha=0.3, label="AMD Threads", kde=True)
#axs[0].plot(future_years, future_predictions_intel_threads, '--', color=colorblind_friendly[0], label="Intel's predicted threads")
#axs[0].plot(future_years, future_predictions_amd_threads, '--', color=colorblind_friendly[1], label="AMD's predicted threads")
#axs[0].set_xlabel('Number of cores and threads')
#axs[0].set_ylabel('Density')
#axs[0].set_title('Distribution of cores and threads of Intel and AMD')
#
## Base and turbo frequency subplot
#sns.kdeplot(data=intel_common_years_frequencies, x="base_frequency", ax=axs[1], color='blue', alpha=0.5, label="Intel Base Frequency")
#sns.kdeplot(data=amd_common_years_frequencies, x="base_frequency", ax=axs[1], color='orange', alpha=0.5, label="AMD Base Frequency")
#axs[1].plot(future_years, future_predictions_intel_base_freq, color=colorblind_friendly[0], label="Intel's predicted base frequency")
#axs[1].plot(future_years, future_predictions_amd_base_freq, color=colorblind_friendly[1], label="AMD's predicted base frequency")
#sns.kdeplot(data=intel_common_years_frequencies, x="turbo_frequency", ax=axs[1], color='blue', alpha=0.3, label="Intel Turbo Frequency", linestyle='--')
#sns.kdeplot(data=amd_common_years_frequencies, x = "turbo_frequency", ax = axs[1], color = 'orange', alpha = 0.3, label = "AMD Turbo Frequency", linestyle = '--')
#axs[1].plot(future_years, future_predictions_intel_turbo_freq, '--', color=colorblind_friendly[0], label="Intel's predicted turbo frequency")
#axs[1].plot(future_years, future_predictions_amd_turbo_freq, '--', color=colorblind_friendly[1], label="AMD's predicted turbo frequency")
#
#axs[1].set_xlabel('Clock speed (GHz)')
#axs[1].set_ylabel('Density')
#axs[1].set_title('Distribution of base and turbo clock speed of Intel and AMD')
#plt.tight_layout()
#axs[0].legend(fontsize=8)
#axs[1].legend(fontsize=8)
#plt.savefig('Kilali_Faycal_7.png', dpi=300)





plt.show()

