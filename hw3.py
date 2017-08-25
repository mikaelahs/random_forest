import sys
from pandas import read_csv, DataFrame, melt, get_dummies, concat
import numpy as np
from sklearn.ensemble import RandomForestRegressor

args = sys.argv

if len(args) != 3:
    print "ERROR: incorrect number of parameters. To run this program you must provide two parameters (csv file containing input data" \
          "and txt file to output predictions, respectively)."

# Read in data provided
test_x = DataFrame(read_csv(args[1], header=None))
test_x.columns = ['Country Name', 'Year', 'GDP']
del test_x['GDP']
train = DataFrame(read_csv('life expectancy by country and year.csv'))

# Format given data sets
train = melt(train, id_vars=['Country Name'], value_vars=['1961','1962','1963','1964','1965','1966','1967','1968','1969','1970','1971','1972',
                                                        '1973','1974','1975','1976','1977','1978','1979','1980','1981','1982','1983','1984',
                                                        '1985','1986','1987','1988','1989','1990','1991','1992','1993','1994','1995','1996',
                                                        '1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007','2008',
                                                        '2009','2010'], var_name='Year', value_name='Life Expectancy')

train = train.sort_values(by = ['Country Name','Year'])
train['Life Expectancy'] = train.groupby('Country Name')['Life Expectancy'].apply(lambda group: group.interpolate(method = 'linear', limit = 100, limit_direction = 'both'))

# Create dummy variables for countries
train_countries = get_dummies(train.iloc[:,0])
test_countries = get_dummies(test_x.iloc[:,0])
col_to_add = np.setdiff1d(train_countries.columns, test_countries.columns)
for c in col_to_add:
    test_countries[c] = 0
test_countries = test_countries[train_countries.columns]

# Construct final train and test sets
train_x = train.iloc[:,1:2]
train_x = concat([train_x, train_countries], axis=1)
test_x = test_x.iloc[:,1:2]
test_x = concat([test_x, test_countries], axis=1)
train_y = train.iloc[:,2]

# Fit model and create predictions
regressor = RandomForestRegressor(n_estimators=12)
regressor.fit(train_x, train_y)
predicted_y = regressor.predict(test_x)

# Write predictions to file
f = open(args[2], 'w')
for prediction in predicted_y:
    f.write(str(prediction)+'\n')
f.close()