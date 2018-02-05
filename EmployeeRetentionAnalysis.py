
# First, we import the neccessary modules for data manipulation
# and visual representation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot
import seaborn as sns

#Read the analytics csv file and store our dataset into a dataframe
#called "employee"
employee = pd.DataFrame.from_csv('HR_comma_sep.csv', index_col=None)

# Let's check to see if there are any missing values in our data set
employee.isnull().any()


# Let's get a quick overview of what we are dealing with in our dataset
employee.head()

# Let's rename certain columns for better readability
employee = employee.rename(columns={'satisfaction_level': 'satisfaction',
                        'last_evaluation': 'evaluation',
                        'number_project': 'projectCount',
                        'average_montly_hours': 'averageMonthlyHours',
                        'time_spend_company': 'yearsAtCompany',
                        'Work_accident': 'workAccident',
                        'promotion_last_5years': 'promotion',
                        'sales' : 'department',
                        'left' : 'turnover'
                        })


# Now let's move the reponse variable "turnover" to the front of the table
front = employee['turnover']
employee.drop(labels=['turnover'], axis=1,inplace = True)
employee.insert(0, 'turnover', front)
employee.head()

# The dataset contains 10 columns and 14999 observations
employee.shape

# Check the type of our features.
employee.dtypes

#What is the amount of turnover for employees?
turnover = employee.turnover.value_counts() / 14999
turnover
# Looks like about 76% of employees stayed and 24% of employees left.

# Let's display the statistical overview of the employees
employee.describe()

# Let's get an overview of summary (Turnover V.S. Non-turnover)
turnoversummary = employee.groupby('turnover')
turnoversummary.mean()

#We want to see which variables are most closely related.
#We do this using a Correlation Matrix.
corr = employee.corr()
corr = (corr)
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
sns.plt.title('Heatmap of Correlation Matrix')
corr

# Let's set up the matplotlib figure
f, axes = plt.subplots(ncols=3, figsize=(15, 6))

# First, let's graph Employee Satisfaction
sns.distplot(employee.satisfaction, kde=False, color="g", ax=axes[0]).set_title('Employee Satisfaction Distribution')

# Then, let's graph Employee Evaluation
sns.distplot(employee.evaluation, kde=False, color="r", ax=axes[1]).set_title('Employee Evaluation Distribution')

# Now, let's graph Employee Average Monthly Hours
sns.distplot(employee.averageMonthlyHours, kde=False, color="b", ax=axes[2]).set_title('Employee Average Monthly Hours Distribution')

#Let's see the distribution of Employee Salary Turnover
f, ax = plt.subplots(figsize=(15, 4))
sns.countplot(y="salary", hue='turnover', data=employee).set_title('Employee Salary Turnover Distribution');

#Let's see the distribution of Employee Department Turnover
f, ax = plt.subplots(figsize=(15, 5))
sns.countplot(y="department", hue='turnover', data=employee).set_title('Employee Department Turnover Distribution');

#Lets distiguish only departments that had turnover
employeecolors = ['#78C850',  # Green
                    '#F08030',  # Red
                    '#6890F0',  # Blue
                    '#A8B820',  # Olive
                    '#A8A878',  # Grey
                    '#A040A0',  # Purple
                    '#F8D030',  # Yellow
                    '#E0C068',  # Brown
                    '#EE99AC',  # Pink
                    '#C03028',  # Maroon
                    '#F85888',  # Light Purple
                    '#B8A038',  # Grey
                    '#705898',  # Faded Purple
                    '#98D8D8',  # Light Blue
                    '#7038F8',  # Other Purple
                   ]

# Show the plot of only tunrover in departments
sns.countplot(x='department', data=employee, palette=employeecolors)
# Rotate x-labels
plt.xticks(rotation=-45)

#Plot turnover in relation to number of projects being taken on
ax = sns.barplot(x="projectCount", y="projectCount", hue="turnover", data=employee, estimator=lambda x: len(x) / len(df) * 100)
ax.set(ylabel="Percent")

#Let's plot turnover as a function of evaluation score
fig = plt.figure(figsize=(15,4),)
ax=sns.kdeplot(employee.loc[(df['turnover'] == 0),'evaluation'] , color='b',shade=True,label='no turnover')
ax=sns.kdeplot(employee.loc[(df['turnover'] == 1),'evaluation'] , color='r',shade=True, label='turnover')
plt.title('Employee Evaluation Distribution - Turnover V.S. No Turnover')
plt.show()

#Let's plot turnover as a function of average monthly hours worked
fig = plt.figure(figsize=(15,4))
ax=sns.kdeplot(employee.loc[(df['turnover'] == 0),'averageMonthlyHours'] , color='b',shade=True, label='no turnover')
ax=sns.kdeplot(employee.loc[(df['turnover'] == 1),'averageMonthlyHours'] , color='r',shade=True, label='turnover')
plt.title('Employee AverageMonthly Hours Distribution - Turnover V.S. No Turnover')
plt.show()

#LEts plot ProjectCount VS AverageMonthlyHours
sns.boxplot(x="projectCount", y="averageMonthlyHours", hue="turnover", data=employee).set_title('Employee ProjectCount V.S. Average Monthly Hours')

#Looks like the average employees who stayed worked about 200hours/month. Those that had a turnover worked about 250hours/month and 150hours/month


#Now, lets plot ProjectCount VS Evaluation
sns.boxplot(x="projectCount", y="evaluation", hue="turnover", data=employee)
#Looks like employees who did not leave the company had an average evaluation of around 70% even with different projectCounts
#There is a huge skew in employees who had a turnover though. It drastically changes after 3 projectCounts.
#Employees that had two projects and a horrible evaluation left. Employees with more than 3 projects and super high evaluations left

#Now let's plot the distribution of satisfaction and evaluation score
sns.lmplot(x='satisfaction', y='evaluation', data=employee,
           fit_reg=False, # No regression line
           hue='turnover')   # Color by evolution stage

#Looks like there are three distinction clusters. LEt's do a cluster model!

# Import KMeans Model
from sklearn.cluster import KMeans

# Graph and create 3 clusters of Employee Turnover
kmeans = KMeans(n_clusters=3,random_state=2)
kmeans.fit(employee[employee.turnover==1][["satisfaction","evaluation"]])

kmeanscolors = ['green' if c == 0 else 'blue' if c == 2 else 'red' for c in kmeans.labels_]

fig = plt.figure(figsize=(10, 6))
plt.scatter(x="satisfaction",y="evaluation", data=employee[employee.turnover==1],
            alpha=0.25,color = kmeanscolors)
plt.xlabel("Satisfaction")
plt.ylabel("Evaluation")
plt.scatter(x=kmeans.cluster_centers_[:,0],y=kmeans.cluster_centers_[:,1],color="black",marker="X",s=100)
plt.title("Clusters of Employee Turnover")
plt.show()
