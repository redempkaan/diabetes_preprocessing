#We want to develop a ml model that checks whether a person has diabetes or not.
#To achieve high accuracy rate, please process the data first.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection
from sklearn import preprocessing as sk
from sklearn.ensemble import RandomForestClassifier


def desc_w_null(dataframe, cols):
    for col in cols:
        raw = dataframe[col].describe()
        raw['null_count'] = dataframe[col].isnull().sum()
        print(raw, end= '\n##############\n')
    return ''

def classify_cols(dataframe, catnum_th=20, car_th=100 ):
    num_cols = [col for col in dataframe.columns if dataframe[col].dtype in ['int64', 'float64']]
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype == 'object']
    num_but_cat = [col for col in num_cols if dataframe[col].nunique() < catnum_th]
    cat_but_car = [col for col in cat_cols if dataframe[col].nunique() > car_th]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    cat_cols = [col for col in cat_cols if col not in cat_but_car] + num_but_cat
    return num_cols, cat_cols, cat_but_car

def cat_summary(df, cols):
    for col in cols:
        print(pd.DataFrame({col : df[col].value_counts(), 'Ratio' : 100 * df[col].value_counts() / df[col].shape[0]}), end= '\n############################\n')
    return

def num_summary(dataframe, cols, plot=False, quantiles=[0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]):
    for col in cols:
        print(dataframe[col].describe(quantiles).T, end= '\n###############\n')
        if plot:
            dataframe[col].hist()
            plt.xlabel(col)
            plt.title(col)
            plt.show(block=True)
    return

def target_summary(dataframe, target_variable, cat_cols, num_cols):
    for col in cat_cols:
        print(dataframe.groupby(col).agg({target_variable : 'mean'}).reset_index(drop=True), end='\n##################\n')
    for col in num_cols:
        print(dataframe.groupby(target_variable).agg({col : 'mean'}).reset_index(drop=True), end='\n##################\n')

def check_nulls(dataframe):
    if dataframe.isnull().values.any():
        null_cols = [col for col in dataframe.columns if dataframe[col].isnull().values.any()]
        print(pd.DataFrame({'Null_values' : [dataframe[col].isnull().sum() for col in null_cols], 'Ratio' : [100 * dataframe[col].isnull().sum() / len(dataframe[col]) for col in null_cols]}, index=[null_cols]))
        return null_cols, {col : dataframe[dataframe[col].isnull()].index for col in null_cols}
    else:
        print('No null values for this dataframe.')
        return


def check_outliers(dataframe, num_cols, q1=0.25, q2=0.75, multiplier=1.5):
    outlier_indexes = {}
    for col in num_cols:
        IQR = (dataframe[col].quantile(q2) - dataframe[col].quantile(q1)) * multiplier
        max_value = dataframe[col].quantile(q2) + IQR
        min_value = dataframe[col].quantile(q1) - IQR
        print(pd.DataFrame({'Out_max' : [len(dataframe[dataframe[col] > max_value])], 'Out_min' : [len(dataframe[dataframe[col] < min_value])]}, index= [col]), end= '\n######################\n')
        outlier_indexes[col] = [dataframe[dataframe[col] > max_value].index, dataframe[dataframe[col] < min_value].index]
    return outlier_indexes

def cor_table(dataframe):
    corr = dataframe.corr().abs()
    up_tri_matrix = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool_))
    high_cor_columns = [col for col in up_tri_matrix.columns if any(up_tri_matrix[col] > 0.90)]
    return high_cor_columns


def check_zeros(dataframe, cols):
    zero_cols = [col for col in cols if (dataframe[col] == 0).values.any()]
    indexes = {col : (dataframe[dataframe[col] == 0]).index for col in zero_cols}
    return indexes, zero_cols

def binary_encoder(dataframe):
    bin_cols = [col for col in dataframe.columns if (dataframe[col].nunique() == 2) & (dataframe[col].dtype not in ['int64', 'float64'])]
    if len(bin_cols) != 0:
        print(f'{len(bin_cols)} bin_col exist in dataframe.')
        for col in bin_cols:
            labelencoder = sk.LabelEncoder()
            dataframe[col] = labelencoder.fit_transform(dataframe[col])
        print('Binary Encoding was succesful.')
        return dataframe
    else:
        print('No binary columns exist in the dataframe.')
        return 0

def ohe_encoder(dataframe, drop_first=True):
    ohe_cols = [col for col in dataframe.columns if (dataframe[col].nunique() <= 20) & (dataframe[col].nunique() > 2)]
    print(f'{len(ohe_cols)} ohe_cols exist in the dataframe.')
    dataframe = pd.get_dummies(dataframe, columns=ohe_cols, drop_first=drop_first)
    print('One Hot Encoding was succesful.')
    return dataframe

def rob_scaler(dataframe, num_cols):
    for col in num_cols:
        rs = sk.RobustScaler()
        dataframe[col + '_RS'] = rs.fit_transform(dataframe[[col]])
    print('Robust Scaling was succesful.')
    return



df = pd.read_csv(r'C:\Users\kaanm\PycharmProjects\pythonProject\ml_bootcamp\diabetes.csv') # Target variable is 'outcome'.




#1- Take a look at the big picture.
print(desc_w_null(df, df.columns))

#2- Catch numeric and categorical variables.
num_cols, cat_cols, car_cols = classify_cols(df, catnum_th= 30)
print(num_cols, cat_cols, car_cols)

#3- Analyze numeric and categorical variables.
cat_summary(df, cat_cols)
num_summary(df, num_cols, plot=True)

#4- Analyze the target variable.
target_summary(df, 'Outcome', cat_cols, num_cols)

#5- Analyze the outliers.
check_outliers(df, num_cols)


#6- Analyze the null values.
check_nulls(df)

#7- Analyze the correlation of variables.
cor_table(df)

#8- Handle the null values and outliers.
check_nulls(df)
#At first sight looks like there is no null values, but there are '0' values in 'Glucose', 'Insulin' etc. which doesn't make any sense.
#Nulls
zero_indexes, zero_columns = check_zeros(df, num_cols)

for col in zero_columns:
    df.loc[zero_indexes[col], col] = float('NaN') #Flagging meaningless zeros with NaN's

null_cols, null_indexes = check_nulls(df)

for col in null_cols:
    df[col].fillna(df.groupby('Pregnancies')[col].transform('mean'), inplace=True)

#Outliers
outlier_indexes = check_outliers(df, num_cols)

for col in list(outlier_indexes.keys()):
    IQR = (df[col].quantile(0.95) - df[col].quantile(0.05)) * 1.5
    max_value = df[col].quantile(0.95) + IQR
    min_value = df[col].quantile(0.05) - IQR
    if len(outlier_indexes[col][0]) != 0:
        df.loc[outlier_indexes[col][0], col] = max_value
    if len(outlier_indexes[col][1]) != 0:
        df.loc[outlier_indexes[col][1], col] = min_value

#9- Create new variables.
#BMI
df.loc[df['BMI'] <= 18.5, 'Status'] = 'Underweight'
df.loc[(df['BMI'] > 18.5) & (df['BMI'] < 24.9), 'Status'] = 'Healthy'
df.loc[(df['BMI'] >= 24.9) & (df['BMI'] < 30), 'Status'] = 'Overweight'
df.loc[(df['BMI'] >= 30) & (df['BMI'] < 34.9), 'Status'] = 'Obese'
df.loc[(df['BMI'] >= 34.9) & (df['BMI'] < 39.9), 'Status'] = 'Severly_Obese'
df.loc[df['BMI'] >= 40, 'Status'] = 'Morbidly_Obese'
#Checking mean values of 'Outcome' in fracture of 'Status'.
df.groupby('Status').agg({'Outcome' : 'mean'})

#AGE
df['Age_Cat'] = pd.cut(x=df['Age'], bins=[20, 29, 40, 60, df['Age'].max()], labels=['Adult', 'Early_Middle', 'Middle_Aged', 'Old'])

#10- Encode the columns.
if binary_encoder(df) != 0:
    df = binary_encoder(df)

df = ohe_encoder(df)
rob_scaler(df, num_cols)

#11- Create a model.

y = df['Outcome']
X = df.drop(['Outcome'], axis=1)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.30, random_state= 17)
rf_model = sklearn.ensemble.RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
sklearn.metrics.accuracy_score(y_pred, y_test)




