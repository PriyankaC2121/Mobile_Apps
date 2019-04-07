#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:34:55 2019

@author: pc
"""

"""
This analysis aims to use PCA and K-means clustering to help analyse the 
survey results. The hope is at the end of the analysis, we would be able to 
see which customer needs and groups should be focused on for our upcoming 
marketing strategy.
"""



# Importing known libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importing new libraries
from sklearn.preprocessing import StandardScaler # standard scaler
from sklearn.decomposition import PCA # principal component analysis
from sklearn.cluster import KMeans # k-means clustering


# Setting pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


# Importing dataset
survey_df = pd.read_excel('Mobile_App_Survey_Data.xlsx')


###############################################################################
# Fundamental Dataset Exploration
###############################################################################


# Dimensions of the DataFrame
survey_df.shape

# Information about each variable
survey_df.info()


# Descriptive statistics
desc = survey_df.describe().round(2)

print(desc)

print(
      survey_df
      .isnull()
      .sum()
      )


survey_df.columns

for col in enumerate(survey_df):
    print(col)

# Rename column names

survey_df.columns = ['caseid',
                 'age' ,
                 '2_iphone',
                 '2_ipod',
                 '2_android',
                 '2_blackberry',
                 '2_nokia',
                 '2_windows',
                 '2_hp',
                 '2_tablet',
                 '2_others',
                 '2_none',
                 '4_music',   
                 '4_tv_checkin',
                 '4_entertainment',
                 '4_tv_show',
                 '4_gaming',
                 '4_socialnet',
                 '4_generalnews',
                 '4_shopping',
                 '4_specificnews',
                 '4_other',
                 '4_none',
                 'num_apps',
                 'free_apps',
                 'visit_facebook', 
                 'visit_twitter',
                 'visit_myspace',
                 'visit_pandoraradio',
                 'visit_vevo',
                 'visit_youtube',
                 'visit_aolradio',
                 'visit_lastfm',
                 'visit_yahoo',
                 'visit_imdb',
                 'visit_linkedin',
                 'visit_netflix',
                 '24_keepup_techdev',
                 '24_advise_tech_elec',
                 '24_like_buying_newgadgets',
                 '24_n_toomuch_tech',
                 '24_use_tech_control',
                 '24_use_tech_time',
                 '24_music_imp',
                 '24_learn_tvshow',
                 '24_n_toomuch_info',
                 '24_check_ff_socialnet',
                 '24_intouch_ff_internet',
                 '24_n_intouch_ff_internet',
                 '25_opinion_leader',
                 '25_stand_out',
                 '25_offer_advice',
                 '25_take_lead',
                 '25_try_new',
                 '25_told_what_to_do',
                 '25_control_freak',
                 '25_risk_taker',
                 '25_creative',
                 '25_optimistic',
                 '25_active',
                 '25_not_enough_time',
                 '26_like_luxurybrands',  
                 '26_discounts',  
                 '26_shopping',
                 '26_package_deals',  
                 '26_shopping_online',
                 '26_designer_brands',
                 '26_cannot_get_enough_apps',
                 '26_cool_apps', 
                 '26_boasting_new_apps',
                 '26_influence_children',
                 '26_paymore_for_features',
                 '26_likes_to_spend',
                 '26_influence_hot_nothot',
                 '26_reflect_style',
                 '26_impulse_purchases',
                 '26_entertainment',
                 'education',
                 'marital',
                 '50_children_zero', 
                 '50_children_below6',
                 '50_children_below12',
                 '50_children_below17',
                 '50_children_above18',
                 'race',
                 'hispanic_latino',
                 'annual_income',
                 'gender',
        ]

#############################################
#  Create new columns 
#############################################
""" 
(1) To reverse questions - for ease of subsequent analysis
(2) Group for questions where more than one answer can be selected - 
 - for potential insights 
"""

##################
#  Reversing questions
##################
    

# create a new column for question on number of free apps
survey_df['free_apps_2'] = -100

# to replace indexes 
survey_df['free_apps_2'][survey_df['free_apps'] ==1] = 6
survey_df['free_apps_2'][survey_df['free_apps'] ==2] = 5
survey_df['free_apps_2'][survey_df['free_apps'] ==3] = 4
survey_df['free_apps_2'][survey_df['free_apps'] ==4] = 3
survey_df['free_apps_2'][survey_df['free_apps'] ==5] = 2
survey_df['free_apps_2'][survey_df['free_apps'] ==6] = 1


survey_df['free_apps'].value_counts()

survey_df['free_apps_2'].value_counts()

# create a new column for question on martial status

survey_df['marital_2'] = -100

# to replace indexes 
survey_df['marital_2'][survey_df['marital'] ==1] = 3
survey_df['marital_2'][survey_df['marital'] ==2] = 1
survey_df['marital_2'][survey_df['marital'] ==3] = 2
survey_df['marital_2'][survey_df['marital'] ==4] = 4

survey_df['marital'].value_counts()

survey_df['marital_2'].value_counts()


# Drop old columns

survey_df = survey_df.drop(columns = ['free_apps'], 
                           axis = 1 )

survey_df = survey_df.drop(columns = ['marital'], 
                           axis = 1 )

survey_df.shape


##################
# New columns for features
##################
"""
These are for questions where survey says you can select more than 1 option in
the answer.
"""

# Total number of apps 

survey_df['2_total_apps'] = ( survey_df['2_iphone']  
                            + survey_df['2_ipod'] 
                            + survey_df['2_android'] 
                            + survey_df['2_blackberry']                                                    
                            + survey_df['2_nokia']
                            + survey_df['2_windows']
                            + survey_df['2_hp']
                            + survey_df['2_tablet']
                            + survey_df['2_others']
                            + survey_df['2_none']
                            )

# Total diffferent types of visits
survey_df['4_total_visits'] = ( survey_df['visit_facebook']  
                             + survey_df['visit_twitter'] 
                             + survey_df['visit_myspace'] 
                             + survey_df['visit_pandoraradio']                                                    
                             + survey_df['visit_vevo']
                             + survey_df['visit_youtube']
                             + survey_df['visit_aolradio']
                             + survey_df['visit_lastfm']
                             + survey_df['visit_imdb']
                             + survey_df['visit_linkedin']
                             + survey_df['visit_netflix']
                                )
""" 
^ Note: Since scale is reversed : 1-very often, 4-almost never.
## So higher the score, they almost never visit anything. 
Look at value counts.

"""
survey_df['4_total_visits'].value_counts()



# Number of children

survey_df['total_num_children'] = (survey_df['50_children_zero']
                                 + survey_df['50_children_below6']
                                 + survey_df['50_children_below12']
                                 + survey_df['50_children_below17']
                                 + survey_df['50_children_above18']
                                 )




###############
#  Check for data consistency
###############

"""
For q50 on number of children, since many values can be selected, 
check to ensure no-one says "no children" and "has children". Will be 
contradictory. 
"""

survey_df['50_children_below6'][survey_df['50_children_zero']==1].sum()
survey_df['50_children_below12'][survey_df['50_children_zero']==1].sum()
survey_df['50_children_below17'][survey_df['50_children_zero']==1].sum()
survey_df['50_children_above18'][survey_df['50_children_zero']==1].sum()


###############
#  Correleation
###############

survey_df.corr()



###############################################################################
###############################################################################
###############################################################################

                                #  PCA   #

###############################################################################
###############################################################################
###############################################################################

########################
# Step 1: Remove demographics
########################


survey_df.columns 

"""
Columns identified to be demographics:
    - case id, age, education, race, hispanic/latino, income, gender, maritial,
    number of children (zero, below 6, below12, below17, above 18, total
    number of children)
"""

## Step 2: remove demographics

survey_df_reduced = survey_df
survey_df_reduced = survey_df.drop(['caseid',
                                    'age' , 
                                    'education', 
                                    'race',
                                    'hispanic_latino',
                                    'annual_income',
                                    'gender',
                                    'marital_2',
                                    '50_children_zero',
                                    '50_children_below6',
                                    '50_children_below12',
                                    '50_children_below17',
                                    '50_children_above18',
                                    'total_num_children',
                                    ],
                                    axis = 1)
                                   

survey_df_reduced.shape


########################
# Step 2: Scale to get equal variance
########################


scaler = StandardScaler()


scaler.fit(survey_df_reduced)


X_scaled_reduced = scaler.transform(survey_df_reduced)


########################
# Step 3: Run PCA without limiting the number of components
########################

customer_pca_reduced = PCA(n_components = None,
                           random_state = 508)


customer_pca_reduced.fit(X_scaled_reduced)

X_pca_reduced = customer_pca_reduced.transform(X_scaled_reduced)

########################
# Step 4: Analyze the scree plot to determine how many components to retain
########################


fig, ax = plt.subplots(figsize=(10, 8))

features = range(customer_pca_reduced.n_components_)
# here your number of components is 6

plt.plot(features,
         customer_pca_reduced.explained_variance_ratio_,
         linewidth = 2,
         marker = 'o',
         markersize = 10,
         markeredgecolor = 'black',
         markerfacecolor = 'grey')


plt.title('Reduced Customer Survey Scree Plot')
plt.xlabel('PCA feature')
plt.ylabel('Explained Variance')
plt.xticks(features)
plt.show()


# Explained variance as a ratio of total variance
customer_pca_reduced.explained_variance_ratio_


########################
# Step 5: Run PCA again based on the desired number of components
########################
"""
Put in number of optimal components    !!!!!!!!!!!!!!!!!!!!!!!
"""

customer_pca_reduced = PCA(n_components = 3,
                           random_state = 508)


customer_pca_reduced.fit(X_scaled_reduced)


########################
# Step 6: Analyze factor loadings to understand principal components
########################

factor_loadings_df = pd.DataFrame(pd.np.transpose
                                  (customer_pca_reduced.components_))

#renaming columns
factor_loadings_df = factor_loadings_df.set_index(survey_df_reduced.columns)


print(factor_loadings_df)
print(factor_loadings_df).round(2)

factor_loadings_df.to_excel('PCA_factor_loadings.xlsx')


########################
# Step 7: Analyze factor strengths per customer
########################

X_pca_reduced = customer_pca_reduced.transform(X_scaled_reduced)


X_pca_df = pd.DataFrame(X_pca_reduced)


########################
# Step 8: Rename your principal components and reattach demographic information
########################

"""
The naming can be anything from your analysis. Put in demographic data so that
you can analyse further. 
"""
X_pca_df.columns = ['PCA1', 'PCA2', 'PCA3']


final_pca_df = pd.concat(
                        [survey_df.loc[ : , [
                                            'caseid',
                                            'age' , 
                                            'education', 
                                            'race',
                                            'hispanic_latino',
                                            'annual_income',
                                            'gender',
                                            'marital_2',
                                            '50_children_zero',
                                            '50_children_below6',
                                            '50_children_below12',
                                            '50_children_below17',
                                            '50_children_above18',
                                            'total_num_children',
                                            ]] , 
                         X_pca_df],
                         axis = 1)



###############################################################################
###############################################################################
###############################################################################

                     # Combining PCA and Clustering!!!

###############################################################################
###############################################################################
###############################################################################

########################
# Step 1: Take your transformed dataframe
########################

print(X_pca_df.head(n = 5))


print(pd.np.var(X_pca_df))


########################
# Step 2: Scale to get equal variance
########################

scaler = StandardScaler()

"""
Note scale the pca df -- not anything else!
"""

scaler.fit(X_pca_df)


X_pca_clust = scaler.transform(X_pca_df)


X_pca_clust_df = pd.DataFrame(X_pca_clust)


print(pd.np.var(X_pca_clust_df))


X_pca_clust_df.columns = X_pca_df.columns


########################
# Step 3: Experiment with different numbers of clusters 
########################

"""
Selecting best number of clusters
"""

ks = range(1, 50)
inertias = []


for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters = k)

    # Fit model to samples
    model.fit(X_scaled_reduced)

    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)


# Plot ks vs inertias
fig, ax = plt.subplots(figsize = (12, 8))
plt.plot(ks, inertias, '-o')


plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)

plt.show()


"""
Using optimal number of clusters
"""

customers_k_pca = KMeans(n_clusters = 5,
                         random_state = 508)


customers_k_pca.fit(X_pca_clust_df)


customers_kmeans_pca = pd.DataFrame({'cluster': customers_k_pca.labels_})


print(customers_kmeans_pca.iloc[: , 0].value_counts())

372.0/1552.0

# ^ use value counts to see if the clustering has a proportionate amount 
# samples.



########################
# Step 4: Analyze cluster centers
########################

centroids_pca = customers_k_pca.cluster_centers_


centroids_pca_df = pd.DataFrame(centroids_pca)


# Rename your principal components (PCA1, PCA2, PCA3)
centroids_pca_df.columns = ['indoor', 'me-time', 'outdoor']


print(centroids_pca_df)


# Sending data to Excel
centroids_pca_df.to_excel('customers_pca_centriods.xlsx')

"""
Analyse at this stage. especially compare with the value counts. 
"""


########################
# Step 5: Analyze cluster memberships
########################

clst_pca_df = pd.concat([customers_kmeans_pca,
                         X_pca_clust_df],
                         axis = 1)

clst_pca_df.columns = ['cluster', 'indoor', 'me-time', 'outdoor' ]


print(clst_pca_df)



########################
# Step 6: Reattach demographic information
########################

final_pca_clust_df = pd.concat(
                               [survey_df.loc[ : , [
                                            'caseid',
                                            'age' , 
                                            'education', 
                                            'race',
                                            'hispanic_latino',
                                            'annual_income',
                                            'gender',
                                            'marital_2',
                                            '50_children_zero',
                                            '50_children_below6',
                                            '50_children_below12',
                                            '50_children_below17',
                                            '50_children_above18',
                                            'total_num_children',
                                            ]] , 
                                clst_pca_df],
                                axis = 1)


print(final_pca_clust_df.head(n = 5))


###############################################################################
###############################################################################
###############################################################################

                     # ANALYSIS # 

###############################################################################
###############################################################################
###############################################################################

########################
# PCA + KMeans
########################

# Renaming age
age_names = {1  : 'Under 18',
             2  : '18-24',
             3  : '25-29',
             4  : '30-34',
             5  : '35-39',
             6  : '40-44',
             7  : '45-49',
             8  : '50-54',
             9  : '55-59',
             10 : '60-64',
             11 : 'Above 65',
             }

final_pca_clust_df['age'].replace(age_names, inplace = True)                     
                                       
       
# Renaming education  
education_names = {1 : 'Some high school',
                   2 : 'High School Graduate',
                   3 : 'Some College',
                   4 : 'College Graduate',
                   5 : 'Some post-graduate studies',
                   6 : 'Post_graduate degree'}

final_pca_clust_df['education'].replace(education_names, inplace = True)                     
                     
# Renaming race
race_names = {1 : 'White / Caucasian',
              2 : 'Black or African American',
              3 : 'Asian',
              4 : 'Native Hawaiian / Other Pacific Islander',
              5 : 'American Indian / Alaska Native',
              6 : 'Other race'}

final_pca_clust_df['race'].replace(race_names, inplace = True)   
                     
                     
# Renaming hispanic_latino
hl_names = {1 : 'Yes',
            2 : 'No'}

final_pca_clust_df['hispanic_latino'].replace(hl_names, inplace = True)
                     
# Renaming annual income    
      
income_names = { 1  : 'Under 10 K',
                 2  : 'Under 15 K',
                 3  : 'Under 20 K',
                 4  : 'Under 30 K',
                 5  : 'Under 40 K',
                 6  : 'Under 50 K',
                 7  : 'Under 60 K',
                 8  : 'Under 70 K',
                 9  : 'Under 80 K',
                 10 : 'Under 90 K',
                 11 : 'Under 100 K',
                 12 : 'Under 125 K',
                 13 : 'Under 150 K',
                 14 : 'More than 150 K',
         
             }

final_pca_clust_df['annual_income'].replace(income_names, inplace = True)                     
                                       
           
                     
# Renaming gender                     
gender_names = {1 : 'Male',
                2 : 'Female'}

final_pca_clust_df['gender'].replace(gender_names, inplace = True)
           
                     
# Renaming marital
marital_names = {1 : 'Single',
                 2 : 'Single + partner',
                 3 : 'Married',
                 4 : 'Seperated/Widowed/Divorced'}

final_pca_clust_df['marital_2'].replace(marital_names, inplace = True)


# Renaming total children
total_children_names = { 0 : 'No Children',
                         1 : '>= 1 child in same age group',
                         2 : '>= 2 children, in diff age groups',
                         3 : '>= 3 children, in diff age groups',
                         4 : '>= 3 children, in diff age groups',
                        }

final_pca_clust_df['total_num_children'].replace(
                                                total_children_names, 
                                                inplace = True)




##############################
# Additional grouping for age and income
##############################
                  
# Renaming age_2
"""
Rationale: Regrouping to have better visual boxplots for analysis
           Under 25: mostly students 
           25-40: working professionals
           Above 55 : old people/approaching retired   
"""
age_names_2 = {1  : 'Under 25',
             2  : 'Under 25',
             3  : '25-40',
             4  : '25-40',
             5  : '25-40',
             6  : '40-55',
             7  : '40-55',
             8  : '40-55',
             9  : 'Above 55',
             10 : 'Above 55',
             11 : 'Above 55',
             }

final_pca_clust_df['age'].replace(age_names_2, inplace = True)                   


# Renaming annual income_2 
"""
Rationale: Regrouping to have better visual boxplots for analysis
           based on income distribution (low income, lower middle, upper
           midle, high income groups)
        

"""   
      
income_names_2 = { 1  : 'Under 40 K',
                 2  : 'Under 40 K',
                 3  : 'Under 40 K',
                 4  : 'Under 40 K',
                 5  : 'Under 40 K',
                 6  : 'Under 80 K',
                 7  : 'Under 80 K',
                 8  : 'Under 80 K',
                 9  : 'Under 80 K',
                 10 : 'Under 120 K',
                 11 : 'Above 120 K',
                 12 : 'Above 120 K',
                 13 : 'Above 120 K',
                 14 : 'Above 120 K',
         
             }

final_pca_clust_df['annual_income'].replace(income_names_2, inplace = True)                     
                                       



# Adding a productivity step
data_df = final_pca_clust_df

########################################################################
# Boxplots
########################################################################

########################
# Age  - FULL
########################

print ( pd.pivot_table(data_df,
               index='age',
               columns='cluster',
               values='indoor',
               aggfunc=np.count_nonzero))

# PCA1 : 
fig, ax = plt.subplots(figsize = (8, 6))
sns.boxplot(x = 'age',
            y = final_pca_clust_df.iloc[:,-3],
            hue = 'cluster',
            order = ['Under 18',
                     '18-24',
                     '25-29',
                     '30-34',
                     '35-39',
                     '40-44',
                     '45-49',
                     '50-54',
                     '55-59',
                     '60-64',
                     'Above 65'                  
                     ],
            data = data_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()


# PCA2: 
fig, ax = plt.subplots(figsize = (8, 6))
sns.boxplot(x = 'age',
            y = final_pca_clust_df.iloc[:,-2],
            hue = 'cluster',
            order = ['Under 18',
                     '18-24',
                     '25-29',
                     '30-34',
                     '35-39',
                     '40-44',
                     '45-49',
                     '50-54',
                     '55-59',
                     '60-64',
                     'Above 65'                  
                     ],
            data = data_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()

# PCA3: 
fig, ax = plt.subplots(figsize = (8,6))
sns.boxplot(x = 'age',
            y = final_pca_clust_df.iloc[:,-1],
            hue = 'cluster',
            order = ['Under 18',
                     '18-24',
                     '25-29',
                     '30-34',
                     '35-39',
                     '40-44',
                     '45-49',
                     '50-54',
                     '55-59',
                     '60-64',
                     'Above 65'                  
                     ],
            data = data_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()


########################
# Age  - GROUPED
########################

print ( pd.pivot_table(data_df,
               index='age',
               columns='cluster',
               values='indoor',
               aggfunc=np.count_nonzero))


# PCA1 : 
fig, ax = plt.subplots(figsize = (8, 6))
sns.boxplot(x = 'age',
            y = final_pca_clust_df.iloc[:,-3],
            hue = 'cluster',
            order = ['Under 25',
                     '25-40',
                     '40-55',
                     'Above 55',
                     ],
            data = data_df)

plt.ylim(-4, 4)
plt.tight_layout()
plt.show()

# PCA2: 
fig, ax = plt.subplots(figsize = (8, 6))
sns.boxplot(x = 'age',
            y = final_pca_clust_df.iloc[:,-2],
            hue = 'cluster',
            order = ['Under 25',
                     '25-40',
                     '40-55',
                     'Above 55',
                     ],
            data = data_df)

plt.ylim(-4, 4)
plt.tight_layout()
plt.show()

# PCA3: 
fig, ax = plt.subplots(figsize = (8,6))
sns.boxplot(x = 'age',
            y = final_pca_clust_df.iloc[:,-1],
            hue = 'cluster',
            order = ['Under 25',
                     '25-40',
                     '40-55',
                     'Above 55',
                     ],
            data = data_df)

plt.ylim(-4, 4)
plt.tight_layout()
plt.show()




########################
# Education
########################

print ( pd.pivot_table(data_df,
               index='education',
               columns='cluster',
               values='indoor',
               aggfunc=np.count_nonzero))
# PCA1 : 
fig, ax = plt.subplots(figsize = (8, 6))
sns.boxplot(x = 'education',
            y = final_pca_clust_df.iloc[:,-3],
            hue = 'cluster',
            order = ['Some high school',
                     'High School Graduate',
                     'Some College',
                     'College Graduate',
                     'Some post-graduate studies',
                     'Post_graduate degree',
                     ],
            data = data_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()



# PCA2: 
fig, ax = plt.subplots(figsize = (8, 6))
sns.boxplot(x = 'education',
            y = final_pca_clust_df.iloc[:,-2],
            hue = 'cluster',
            order = ['Some high school',
                     'High School Graduate',
                     'Some College',
                     'College Graduate',
                     'Some post-graduate studies',
                     'Post_graduate degree',
                     ],
            data = data_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()

# PCA3: 
fig, ax = plt.subplots(figsize = (8,6))
sns.boxplot(x = 'education',
            y = final_pca_clust_df.iloc[:,-1],
            hue = 'cluster',
            order = ['Some high school',
                     'High School Graduate',
                     'Some College',
                     'College Graduate',
                     'Some post-graduate studies',
                     'Post_graduate degree',
                     ],
            data = data_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()



########################
# Race
########################

print ( pd.pivot_table(data_df,
               index='race',
               columns='cluster',
               values='indoor',
               aggfunc=np.count_nonzero))


# PCA1 : 
fig, ax = plt.subplots(figsize = (8, 6))
sns.boxplot(x = 'race',
            y = final_pca_clust_df.iloc[:,-3],
            hue = 'cluster',
             order = ['White / Caucasian',
                     'Black or African American',
                     'Asian',
                     'Native Hawaiian / Other Pacific Islander',
                     'American Indian / Alaska Native',
                     'Other race'
                     ],
            data = data_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()



# PCA2: 
fig, ax = plt.subplots(figsize = (12, 6))
sns.boxplot(x = 'race',
            y = final_pca_clust_df.iloc[:,-2],
            hue = 'cluster',
             order = ['White / Caucasian',
                     'Black or African American',
                     'Asian',
                     'Native Hawaiian / Other Pacific Islander',
                     'American Indian / Alaska Native',
                     'Other race'
                     ],
            data = data_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()

# PCA3: 
fig, ax = plt.subplots(figsize = (8,6))
sns.boxplot(x = 'race',
            y = final_pca_clust_df.iloc[:,-1],
            hue = 'cluster',
            order = ['White / Caucasian',
                     'Black or African American',
                     'Asian',
                     'Native Hawaiian / Other Pacific Islander',
                     'American Indian / Alaska Native',
                     'Other race'
                     ],
            data = data_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()


########################
# Hispanic/Latino
########################

print ( pd.pivot_table(data_df,
               index='hispanic_latino',
               columns='cluster',
               values='indoor',
               aggfunc=np.count_nonzero))


# PCA1 : 
fig, ax = plt.subplots(figsize = (8, 6))
sns.boxplot(x = 'hispanic_latino',
            y = final_pca_clust_df.iloc[:,-3],
            hue = 'cluster',
            data = data_df)

plt.ylim(-4, 4)
plt.tight_layout()
plt.show()



# PCA2: 
fig, ax = plt.subplots(figsize = (8, 6))
sns.boxplot(x = 'hispanic_latino',
            y = final_pca_clust_df.iloc[:,-2],
            hue = 'cluster',
            data = data_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()

# PCA3: 
fig, ax = plt.subplots(figsize = (8,6))
sns.boxplot(x = 'hispanic_latino',
            y = final_pca_clust_df.iloc[:,-1],
            hue = 'cluster',
            data = data_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()



########################
# Income Group
########################


print ( pd.pivot_table(data_df,
               index='annual_income',
               columns='cluster',
               values='indoor',
               aggfunc=np.count_nonzero))

# PCA1 : 
fig, ax = plt.subplots(figsize = (12, 6))
sns.boxplot(x = 'annual_income',
            y = final_pca_clust_df.iloc[:,-3],
            hue = 'cluster',
             order = ['Under 15 K',
                     'Under 20 K',
                     'Under 30 K',
                     'Under 40 K',
                     'Under 50 K',
                     'Under 60 K',
                     'Under 70 K',
                     'Under 80 K',
                     'Under 90 K',
                     'Under 100 K',
                     'Under 125 K',
                     'Under 150 K',
                     'More than 150 K',                  
                     ],
            data = data_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()


# PCA2: 
fig, ax = plt.subplots(figsize = (12, 6))
sns.boxplot(x = 'annual_income',
            y = final_pca_clust_df.iloc[:,-2],
            hue = 'cluster',
             order = ['Under 15 K',
                     'Under 20 K',
                     'Under 30 K',
                     'Under 40 K',
                     'Under 50 K',
                     'Under 60 K',
                     'Under 70 K',
                     'Under 80 K',
                     'Under 90 K',
                     'Under 100 K',
                     'Under 125 K',
                     'Under 150 K',
                     'More than 150 K',                  
                     ],
            data = data_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()

# PCA3: 
fig, ax = plt.subplots(figsize = (12,6))
sns.boxplot(x = 'annual_income',
            y = final_pca_clust_df.iloc[:,-1],
            hue = 'cluster',
            order = ['Under 15 K',
                     'Under 20 K',
                     'Under 30 K',
                     'Under 40 K',
                     'Under 50 K',
                     'Under 60 K',
                     'Under 70 K',
                     'Under 80 K',
                     'Under 90 K',
                     'Under 100 K',
                     'Under 125 K',
                     'Under 150 K',
                     'More than 150 K',
                     
                     ],
            data = data_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()


########################
# Income Group - Grouped
########################


print ( pd.pivot_table(data_df,
               index='annual_income',
               columns='cluster',
               values='indoor',
               aggfunc=np.count_nonzero))

# PCA1 : 
fig, ax = plt.subplots(figsize = (12, 6))
sns.boxplot(x = 'annual_income',
            y = final_pca_clust_df.iloc[:,-3],
            hue = 'cluster',
             order = ['Under 40 K',
                     'Under 80 K',
                     'Under 120 K',
                     'Above 120 K',
                     ],
            data = data_df)

plt.ylim(-4, 4)
plt.tight_layout()
plt.show()


# PCA2: 
fig, ax = plt.subplots(figsize = (12, 6))
sns.boxplot(x = 'annual_income',
            y = final_pca_clust_df.iloc[:,-2],
            hue = 'cluster',
             order = ['Under 40 K',
                     'Under 80 K',
                     'Under 120 K',
                     'Above 120 K',
                     ],
            data = data_df)

plt.ylim(-4, 4)
plt.tight_layout()
plt.show()

# PCA3: 
fig, ax = plt.subplots(figsize = (12,6))
sns.boxplot(x = 'annual_income',
            y = final_pca_clust_df.iloc[:,-1],
            hue = 'cluster',
            order = ['Under 40 K',
                     'Under 80 K',
                     'Under 120 K',
                     'Above 120 K',
                     ],
            data = data_df)

plt.ylim(-4, 4)
plt.tight_layout()
plt.show()







########################
# Gender
########################

print ( pd.pivot_table(data_df,
               index='gender',
               columns='cluster',
               values='indoor',
               aggfunc=np.count_nonzero))


# PCA1 : 
fig, ax = plt.subplots(figsize = (8, 6))
sns.boxplot(x = 'gender',
            y = final_pca_clust_df.iloc[:,-3],
            hue = 'cluster',
            data = data_df)

plt.ylim(-4, 4)
plt.tight_layout()
plt.show()



# PCA2: 
fig, ax = plt.subplots(figsize = (8, 6))
sns.boxplot(x = 'gender',
            y = final_pca_clust_df.iloc[:,-2],
            hue = 'cluster',
            data = data_df)

plt.ylim(-4, 6)
plt.tight_layout()
plt.show()

# PCA3: 
fig, ax = plt.subplots(figsize = (8,6))
sns.boxplot(x = 'gender',
            y = final_pca_clust_df.iloc[:,-1],
            hue = 'cluster',
            data = data_df)

plt.ylim(-4, 6)
plt.tight_layout()
plt.show()


########################
# Marital Status
########################

print ( pd.pivot_table(data_df,
               index='marital_2',
               columns='cluster',
               values='indoor',
               aggfunc=np.count_nonzero))


# PCA1 : 
fig, ax = plt.subplots(figsize = (8, 6))
sns.boxplot(x = 'marital_2',
            y = final_pca_clust_df.iloc[:,-3],
            hue = 'cluster',
            order = ['Single',
                     'Single + partner',
                     'Married',
                     'Seperated/Widowed/Divorced',
                     ],
            data = data_df)

plt.ylim(-5, 5)
plt.tight_layout()
plt.show()



# PCA2: 
fig, ax = plt.subplots(figsize = (8, 6))
sns.boxplot(x = 'marital_2',
            y = final_pca_clust_df.iloc[:,-2],
            hue = 'cluster',
            order = ['Single',
                     'Single + partner',
                     'Married',
                     'Seperated/Widowed/Divorced',
                     ],
            data = data_df)

plt.ylim(-4, 4)
plt.tight_layout()
plt.show()

# PCA3: 
fig, ax = plt.subplots(figsize = (8,6))
sns.boxplot(x = 'marital_2',
            y = final_pca_clust_df.iloc[:,-1],
            hue = 'cluster',
            order = ['Single',
                     'Single + partner',
                     'Married',
                     'Seperated/Widowed/Divorced',
                     ],
            data = data_df)

plt.ylim(-4, 4)
plt.tight_layout()
plt.show()



