import graphlab

# Limit number of worker processes. This preserves system memory, which prevents
# hosted notebooks from crashing.
graphlab.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 4)

# Load some house sales data
# Dataset is from house sales in King County, the region where the city of Seattle,
# WA is located.
sales = graphlab.SFrame('Week 2/home_data.gl/')
sales

# Exploring the data for housing sales
# The house price is correlated with the number of square feet of living space.
graphlab.canvas.set_target('ipynb')
sales.show(view="Scatter Plot", x="sqft_living", y="price")

# Create a simple regression model of sqft_living to price
# Split data into training and testing.
# We use seed=0 so that everyone running this notebook gets the same results.
# In practice, you may set a random seed (or let GraphLab Create pick a random
# seed for you).
train_data,test_data = sales.random_split(.8,seed=0)

# Build the regression model using only sqft_living as a feature
sqft_model = graphlab.linear_regression.create(train_data, target='price', features=['sqft_living'],validation_set=None)

# Evaluate the simple model
print test_data['price'].mean()
print sqft_model.evaluate(test_data)

# Let's show what our predictions look like
# Matplotlib is a Python plotting library that is also useful for plotting.
# You can install it with:
# 'pip install matplotlib'
import matplotlib.pyplot as plt
plt.plot(test_data['sqft_living'],test_data['price'],'.',
        test_data['sqft_living'],sqft_model.predict(test_data),'-')

sqft_model.get('coefficients')

# Explore other features in the data
# To build a more elaborate model, we will explore using more features.
my_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']
sales[my_features].show()
sales.show(view='BoxWhisker Plot', x='zipcode', y='price')

# Build a regression model with more features
my_features_model = graphlab.linear_regression.create(train_data,target='price',features=my_features,validation_set=None)
print my_features

# Comparing the results of the simple model with adding more features
print sqft_model.evaluate(test_data)
print my_features_model.evaluate(test_data)

# Apply learned models to predict prices of 3 houses
house1 = sales[sales['id']=='5309101200']
print house1['price']
print sqft_model.predict(house1)
print my_features_model.predict(house1)

# Prediction for a second, fancier house
house2 = sales[sales['id']=='1925069082']
print sqft_model.predict(house2)
print my_features_model.predict(house2)

# Last house, super fancy
bill_gates = {'bedrooms':[8],
              'bathrooms':[25],
              'sqft_living':[50000],
              'sqft_lot':[225000],
              'floors':[4],
              'zipcode':['98039'],
              'condition':[10],
              'grade':[10],
              'waterfront':[1],
              'view':[4],
              'sqft_above':[37500],
              'sqft_basement':[12500],
              'yr_built':[1994],
              'yr_renovated':[2010],
              'lat':[47.627606],
              'long':[-122.242054],
              'sqft_living15':[5000],
              'sqft_lot15':[40000]}

print sqft_model.predict(graphlab.SFrame(bill_gates))
print my_features_model.predict(graphlab.SFrame(bill_gates))

## Assignment
# 1. Selection and summary statistics: In the notebook we covered in the module,
# we discovered which neighborhood (zip code) of Seattle had the highest average
# house sale price. Now, take the sales data, select only the houses with this
# zip code, and compute the average price.
sales.show(view='BoxWhisker Plot', x='zipcode', y='price')
zip_code_98039 = sales[sales['zipcode']=='98039']
zip_code_98039['price'].mean()
sales[sales['zipcode']=='98039']['price'].mean()

## Assignment
# 2. Filtering data: One of the key features we used in our model was the number
# of square feet of living space (‘sqft_living’) in the house. For this part, we
# are going to use the idea of filtering (selecting) data.

# In particular, we are going to use logical filters to select rows of an SFrame.
# You can find more info in the Logical Filter section of this documentation.
# Using such filters, first select the houses that have ‘sqft_living’ higher
# than 2000 sqft but no larger than 4000 sqft.What fraction of the all houses
# have ‘sqft_living’ in this range?

# https://turi.com/products/create/docs/generated/graphlab.SFrame.html

sqft_2000_to_4000 = sales[(sales['sqft_living'] > 2000) & (sales['sqft_living'] <= 4000)]
sqft_2000_to_4000.show()
sqft_2000_to_4000.num_columns()
sqft_rows = sqft_2000_to_4000.num_rows()
sqft_rows
sales_rows = sales.num_rows()
sales_rows
(float(sqft_rows) / float(sales_rows)) * 100
(9118.0 / 21613.0) * 100

## Assignment
# 3. Building a regression model with several more features: In the sample
# notebook, we built two regression models to predict house prices, one using
# just ‘sqft_living’ and the other one using a few more features, we called this
# set my_features.

# Now, going back to the original dataset, you will build a model using the
# following features:

# advanced_features =
# [
# 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode',
# 'condition', # condition of house
# 'grade', # measure of quality of construction
# 'waterfront', # waterfront property
# 'view', # type of view
# 'sqft_above', # square feet above ground
# 'sqft_basement', # square feet in basement
# 'yr_built', # the year built
# 'yr_renovated', # the year renovated
# 'lat', 'long', # the lat-long of the parcel
# 'sqft_living15', # average sq.ft. of 15 nearest neighbors
# 'sqft_lot15', # average lot size of 15 nearest neighbors
# ]

# 4. Compute the RMSE (root mean squared error) on the test_data for the model
# using just my_features, and for the one using advanced_features.

# - Compute the RMSE on the test_data for my_features_model
# - Compute the RMSE on the test_data for advanced_features_model

# Note 1: both models must be trained on the original sales dataset, not the
# filtered one.

# Note 2: when doing the train-test split, make sure you use seed=0, so you get
# the same training and test sets, and thus results, as we do.

# Note 3: in the module we discussed residual sum of squares (RSS) as an error
# metric for regression, but GraphLab Create uses root mean squared error (RMSE).
# These are two common measures of error regression, and RMSE is simply the
# square root of the mean RSS:

# RMSE = square_root(RSS / N)

# where N is the number of data points. RMSE can be more intuitive than RSS,
# since its units are the same as that of the target column in the data, in our
# case the unit is dollars ($), and doesn't grow with the number of data points,
# like the RSS does.

# (Important note: when answering the question below using GraphLab Create, when
# you call the linear_regression.create() function, make sure you use the
# parameter validation_set=None, as done above. When you use regression GraphLab
# Create, it sets aside a small random subset of the data to validate some
# parameters. This process can cause fluctuations in the final RMSE, so we will
# avoid it to make sure everyone gets the same answer.)

# 1. What is the difference in RMSE between the model trained with my_features
# and the one trained with advanced_features?

advanced_features = [ 'bedrooms', 'bathrooms', 'sqft_living',
                     'sqft_lot', 'floors', 'zipcode', 'condition',
                     'grade', 'waterfront', 'view', 'sqft_above',
                     'sqft_basement', 'yr_built', 'yr_renovated',
                     'lat', 'long', 'sqft_living15', 'sqft_lot15']

advanced_model = graphlab.linear_regression.create(train_data,target='price',features=advanced_features,validation_set=None)

print sqft_model.evaluate(test_data)
print my_features_model.evaluate(test_data)
print advanced_model.evaluate(test_data)

print 'Difference:'
print my_features_model.evaluate(test_data)['rmse'] - advanced_model.evaluate(test_data)['rmse']

# Just playing for here onwards:
print sqft_model.predict(house1)
print my_features_model.predict(house1)
print advanced_model.predict(house1)

print sqft_model.predict(house2)
print my_features_model.predict(house2)
print advanced_model.predict(house2)

print sqft_model.predict(graphlab.SFrame(bill_gates))
print my_features_model.predict(graphlab.SFrame(bill_gates))
print advanced_model.predict(graphlab.SFrame(bill_gates))

plt.plot(test_data['sqft_living'],test_data['price'],'.',
        test_data['sqft_living'],sqft_model.predict(test_data),'-')

plt.plot(test_data['sqft_living'],test_data['price'],'.',
        test_data['sqft_living'],my_features_model.predict(test_data),'-')

plt.plot(test_data['sqft_living'],test_data['price'],'.',
        test_data['sqft_living'],advanced_model.predict(test_data),'-')
