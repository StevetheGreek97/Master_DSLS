"""
Programming_1 finall assignment
Author: Stylianos Mavrianos
Date: 25/01/2022
Version: 2

Supporting file for finall assignment
"""

#Importing Libralies
import pandas as pd
import numpy as np
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
from bokeh.transform import jitter
from bokeh.models import ColumnDataSource,Range1d
from bokeh.layouts import row 
import matplotlib.pyplot as plt

def get_NP_level(df, np):
     #######################################################
    #
    # Arguments:
    #
    # df                  pd.dataframe
    # np                  level of nanoplastic concentration
    #
    # Author:            Stylianos Mavrianos
    # Date:              2021-11-06
    #
    ########################################################
    
    df_np_level = df[df['NP_treatment'] == np]
    
    return df_np_level


def get_INF_level(df, inf):
     #######################################################
    #
    # Arguments:
    #
    # df                  pd.dataframe
    # np                  level of infection
    #
    # Author:            Stylianos Mavrianos
    # Date:              2021-11-06
    #
    ########################################################
    df_inf_level = df[df['Inf_treatment'] == inf] 
    
    return df_inf_level


def get_values(df, var):
     #######################################################
    #
    # Arguments:
    #
    # df                  pd.dataframe
    # var                 variable of interest
    #
    # Author:            Stylianos Mavrianos
    # Date:              2021-11-06
    #
    ########################################################
    if var == 'Age_death':
        x = 3
    if var == 'Total_juv':
        x = 4
    if var == 'Spore_yield':
        x = 8
    values = np.array(df.iloc[0:, x]) 
    
    return values

def get_means(df, var):
    #######################################################
    #
    # Arguments:
    #
    # df                  pd.dataframe
    # var                 variable of interest
    #
    # Author:            Stylianos Mavrianos
    # Date:              2021-11-06
    #
    ########################################################
    means = pd.DataFrame(data=df.groupby(['Inf_treatment','NP_treatment'])[var].mean())
    means = means.reset_index()
    means = means.drop(columns=['Inf_treatment'])
    
    return means

def get_std(df, var):
    #######################################################
    #
    # Arguments:
    #
    # df                  pd.dataframe
    # var                 variable of interest
    #
    # Author:            Stylianos Mavrianos
    # Date:              2021-11-06
    #
    ########################################################
    high, low, zero = df.groupby('NP_treatment')
    
    std_zero = np.std(zero[1][var], ddof=1) / np.sqrt(np.size(zero[1][var]))
    std_low = np.std(low[1][var], ddof=1) / np.sqrt(np.size(low[1][var]))
    std_high = np.std(high[1][var], ddof=1) / np.sqrt(np.size(high[1][var]))
    
    d = {'NP_treatment': ['HIGH','LOW','ZERO'], 'std': [std_high,std_low,  std_zero,]}
    std =  pd.DataFrame(data=d)
    
    return std

def errorbar_df(df, var):
    #######################################################
    #
    # Arguments:
    #
    # df                  pd.dataframe
    # var                 variable of interest
    #
    # Author:            Stylianos Mavrianos
    # Date:              2021-11-06
    #
    ########################################################
    mean = get_means(df, var)
    std = get_std(df, var)
    mean['std_lower'] = mean[var] - std['std']
    mean['std_upper'] = mean[var] + std['std']
    
    return mean


def make_plot(df_main, df_errorbar, var, title,axis_label, num):
    #######################################################
    #
    # Arguments:
    #
    # df_main                  pd.dataframe
    # df_errorbar              pd.dataframe
    # var                      variable of interest
    # title                    title of the plot
    # axis_label               label of the plot
    # num                      y_max_range
    #
    # Author:            Stylianos Mavrianos
    # Date:              2021-11-06
    #
    ########################################################   
    np_levels = ['ZERO', 'LOW', 'HIGH']

    source = ColumnDataSource(df_main)
    source_1 = ColumnDataSource(df_errorbar)

    p = figure(tools="pan, hover")
    p = figure(x_range= np_levels,y_range = (0,num), 
               toolbar_location=None,
               title=title,
               plot_height=300, plot_width=300)
    p.yaxis.axis_label = axis_label


    #Scatter data
    p.dot(x= jitter('NP_treatment',width=0.2, range=p.x_range), y=jitter(var, width=1, range=p.y_range), source= source , alpha=0.3, size = 35)


    p.line(x = 'NP_treatment', y = var, source= source_1, color = 'black')

    # Error asterisks
    p.asterisk(x = 'NP_treatment', y = 'std_lower', source= source_1, color = 'black', size = 3)
    p.asterisk(x = 'NP_treatment', y = 'std_upper', source= source_1, color = 'black', size = 3)

    # Mean
    p.dot(x = 'NP_treatment', y = var, source= source_1, size = 35, color = 'red')

    # p.ygrid.grid_line_color = None

    return p


def make_barplot(df_main, df_errorbar, var, title, axis_label):
    #######################################################
    #
    # Arguments:
    #
    # df_main                  pd.dataframe
    # df_errorbar              pd.dataframe
    # var                 variable of interest
    # title                    title of the plot
    # axis_label               label of the plot
    #
    # Author:            Stylianos Mavrianos
    # Date:              2021-11-06
    #
    ########################################################
    from bokeh.palettes import Spectral3
    
    h, l, z = df_main.groupby(['NP_treatment'])[var].sum() / df_main.groupby(['NP_treatment'])[var].count()
    
    NP_treatment = ['ZERO', 'LOW', 'HIGH']
    counts = [z, l, h]
    
    source = ColumnDataSource(data=dict(NP_treatment = NP_treatment, counts = counts, color=Spectral3))
    source_1 = ColumnDataSource(df_errorbar)

    p = figure(x_range = NP_treatment, y_range = (0,1.05), height = 250, title = title,
               toolbar_location = None, tools = "")

    p.vbar(x='NP_treatment', top = 'counts', width=0.2, color='color', source = source)
    p.xgrid.grid_line_color = None
    p.yaxis.axis_label = axis_label

    # Error asterisks
    p.asterisk(x = 'NP_treatment', y = 'std_lower', source= source_1, color = 'black', size = 3)
    p.asterisk(x = 'NP_treatment', y = 'std_upper', source= source_1, color = 'black', size = 3)

    return p 


def anova(df, Response_var, *Explanatory_var ):
    #######################################################
    #
    # Arguments:
    #
    # df                  pd.dataframe
    # Response_var        Response variable 
    # Explanatory_var     Explanatory variables
    #
    # Author:            Stylianos Mavrianos
    # Date:              2021-11-06
    #
    ########################################################

    import pingouin as pg
    
    aov = pg.anova(dv=Response_var, between=[*Explanatory_var], data= df,
               detailed=True)
    
    return aov.round(4)
 
def figures():  

    zero_counts = np.array([11, 14])
    zero_labels = ["Infected", "Not infected",]
    zero_colors = ['r', 'g']
    plt.pie(zero_counts, labels = zero_labels, colors = zero_colors, shadow = True)
    plt.title("Zero concentration")  
    plt.show()
  
    low_counts = np.array([1, 2, 6, 16])
    low_labels = ['excluded', 'early death', "Infected", "Not infected"]
    low_colors = ['k', 'gray', 'r', 'g']
    plt.title("Low concentration")  
    plt.pie(low_counts, labels = low_labels, colors = low_colors, shadow = True, counterclock = True)
    plt.show()

    high_counts = np.array([2, 6, 3, 14])
    high_labels = ['excluded', 'early death', "Infected", "Not infected"]
    high_colors = ['k', 'gray', 'r', 'g']
    plt.title("High concentration")
    plt.pie(high_counts, labels = high_labels, colors = high_colors, shadow = True)
    plt.show()
    
    
    
def Q_Q_Plot(y, est = 'robust', **kwargs):
    
    ################################################################################
    #
    # Arguments:
    #
    # y                  data array
    # est                Estimation method for normal parameters mu and sigma:
    #                    either 'robust' (default), or 'ML' (Maximum Likelihood),
    #                    or 'preset' (given values)
    # If est='preset' than the optional parameters mu, sigma must be provided
    #
    # Author:            M.E.F. Apol
    # Date:              2020-01-06
    #
    ################################################################################
    
    import numpy as np
    from scipy.stats import iqr # iqr is the Interquartile Range function
    import matplotlib.pyplot as plt
    
    # First, get the optional arguments mu and sigma:
    mu_0 = kwargs.get('mu', None)
    sigma_0 = kwargs.get('sigma', None)
    
    n = len(y)
    
    # Calculate order statistic:
    y_os = np.sort(y)
  
    # Estimates of mu and sigma:
    # ML estimates:
    mu_ML = np.mean(y)
    sigma2_ML = np.mean((y - mu_ML)**2)
    sigma_ML = np.sqrt(sigma2_ML) # biased estimate
    s2 = n/(n-1) * sigma2_ML
    s = np.sqrt(s2) # unbiased estimate
    # Robust estimates:
    mu_R = np.median(y)
    sigma_R = iqr(y)/1.349

    # Assign values of mu and sigma for z-transform:
    if est == 'ML':
        mu, sigma = mu_ML, s
    elif est == 'robust':
        mu, sigma = mu_R, sigma_R
    elif est == 'preset':
        mu, sigma = mu_0, sigma_0
    else:
        print('Wrong estimation method chosen!')
        
    print('Estimation method: ' + est)
    print('mu = ',mu,', sigma = ',sigma)
        
    # Perform z-transform: sample quantiles z.i
    z_i = (y_os - mu)/sigma

    # Calculate cumulative probabilities p.i:
    i = np.array(range(n)) + 1
    p_i = (i - 0.5)/n

    # Calculate theoretical quantiles z.(i):
    from scipy.stats import norm
    z_th = norm.ppf(p_i, 0, 1)

    # Calculate SE or theoretical quantiles:
    SE_z_th = (1/norm.pdf(z_th, 0, 1)) * np.sqrt((p_i * (1 - p_i)) / n)

    # Calculate 95% CI of diagonal line:
    CI_upper = z_th + 1.96 * SE_z_th
    CI_lower = z_th - 1.96 * SE_z_th

    # Make Q-Q plot:
    plt.plot(z_th, z_i, 'o', color='k', label='experimental data')
    plt.plot(z_th, z_th, '--', color='r', label='normal line')
    plt.plot(z_th, CI_upper, '--', color='b', label='95% CI')
    plt.plot(z_th, CI_lower, '--', color='b')
    plt.xlabel('Theoretical quantiles, $z_{(i)}$')
    plt.ylabel('Sample quantiles, $z_i$')
    plt.title('Q-Q plot (' + est + ')')
    plt.legend(loc='best')
    plt.show()


def Q_Q_Hist(y, est='robust', **kwargs):
    
    ################################################################################
    #
    # Arguments:
    #
    # y                  data array
    # est                Estimation method for normal parameters mu and sigma:
    #                    either 'robust' (default), or 'ML' (Maximum Likelihood),
    #                    or 'preset' (given values)
    # If est='preset' than the optional parameters mu, sigma must be provided  
    #
    # Author:            M.E.F. Apol
    # Date:              2020-01-06
    #
    ################################################################################
    
    import numpy as np
    from scipy.stats import iqr # iqr is the Interquartile Range function
    from scipy.stats import norm
    import matplotlib.pyplot as plt
    
    # First, get the optional arguments mu and sigma:
    mu_0 = kwargs.get('mu', None)
    sigma_0 = kwargs.get('sigma', None)
    
    n = len(y)
    
    # Estimates of mu and sigma:
    # ML estimates:
    mu_ML = np.mean(y)
    sigma2_ML = np.mean((y - mu_ML)**2)
    sigma_ML = np.sqrt(sigma2_ML) # biased estimate
    s2 = n/(n-1) * sigma2_ML
    s = np.sqrt(s2) # unbiased estimate
    # Robust estimates:
    mu_R = np.median(y)
    sigma_R = iqr(y)/1.349

    # Assign values of mu and sigma for z-transform:
    if est == 'ML':
        mu, sigma = mu_ML, s       
    elif est == 'robust':
        mu, sigma = mu_R, sigma_R
    elif est == 'preset':
        mu, sigma = mu_0, sigma_0
    else:
        print('Wrong estimation method chosen!')
    print('Estimation method: ' + est)
    print('mu = ',mu,', sigma = ',sigma)
        
    # Calculate the CLT normal distribution:
    x = np.linspace(np.min(y), np.max(y), 501)
    rv = np.array([norm.pdf(xi, loc = mu, scale = sigma) for xi in x])
    
    # Make a histogram with corresponding normal distribution:
    nn, bins, patches = plt.hist(x=y, density=True,
                                 bins='auto', 
                                 color='darkgrey',alpha=1, rwidth=1, label='experimental')
    h = plt.plot(x, rv, 'r', label='normal approximation')
    plt.grid(axis='y', alpha=0.5)
    plt.xlabel('Values, $y$')
    plt.ylabel('Probability $f(y)$')
    plt.title('Histogram with corresponding normal distribution (' + est + ')')
    plt.legend(loc='best')
    plt.show()
    
if __name__ == "__main__":
    main()

