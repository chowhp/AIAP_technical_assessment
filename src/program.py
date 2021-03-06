import pandas as pd
import sys
import sqlite3
from matplotlib import pyplot as plt
from sklearn import datasets, preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression	
import numpy as np


def model(X_train, X_test, y_train, y_test, val):
	"""
	Models to be trained with 20-fold CV
	param X_train, y_train: train samples
	param X_test, y_test: test samples
	return: print precision, recall and f1 
	"""
	if (val == '1'):
		algo = LogisticRegression()
		name = 'LogisticRegression'	
	elif (val == '2'):
		algo = GaussianNB()
		name = 'GaussianNB'	
	elif (val=='3'):
		algo = DecisionTreeClassifier(criterion = 'entropy')
		name = 'DecisionTreeClassifier'	
	elif (val == '4'):
		algo = RandomForestClassifier(n_estimators=10, criterion='entropy')
		name = 'RandomForestClassifier'	
	
	score = ['precision','recall','f1']
	algo_val = cross_validate(algo, X_train, y_train.values.ravel(), cv = 20, scoring = score)
	print('{} model with 20-fold cv has precision: {}, recall: {} and f1: {}'.format(name, algo_val['test_precision'].mean(), algo_val['test_recall'].mean(), algo_val['test_f1'].mean()))


def sample_cleanup(df):
	"""
	Prepare data - remove missing values, duplicates, redundant features & consistency check
	param df: input samples 
	return df: samples after cleaning
	"""
	# dropna
	df.dropna(inplace = True)
	
	# drop duplicates
	df.drop_duplicates(subset = 'ID', keep = 'first', inplace = True)

	# drop features 
	drop_col = ['Favorite color', 'Height', 'ID']
	df.drop(drop_col, axis=1, inplace=True)

	# check consistency & numeric
	df.loc[df['Diabetes'] == 'Normal', 'Diabetes'] = '0'
	df.loc[df['Diabetes'] == 'Pre-diabetes', 'Diabetes'] = '1'
	df.loc[df['Diabetes'] == 'Diabetes', 'Diabetes'] = '2'
	df['Diabetes']= df['Diabetes'].astype('int')
	#df['Diabetes'].describe()

	df.loc[df['Survive'] == 'No', 'Survive'] = '0'
	df.loc[df['Survive'] == 'Yes', 'Survive'] = '1'
	df['Survive']= df['Survive'].astype('int')

	df.loc[df['Gender'] == 'Male', 'Gender'] = '1'
	df.loc[df['Gender'] == 'Female', 'Gender'] = '0'
	df['Gender']= df['Gender'].astype('int')

	df.loc[df['Smoke'] == 'Yes', 'Smoke'] = '1'
	df.loc[df['Smoke'] == 'YES', 'Smoke'] = '1'
	df.loc[df['Smoke'] == 'No', 'Smoke'] = '0'
	df.loc[df['Smoke'] == 'NO', 'Smoke'] = '0'
	df['Smoke']= df['Smoke'].astype('int')

	df.loc[df['Ejection Fraction'] == 'L', 'Ejection Fraction'] = '1'
	df.loc[df['Ejection Fraction'] == 'Low', 'Ejection Fraction'] = '1'
	df.loc[df['Ejection Fraction'] == 'N', 'Ejection Fraction'] = '2'
	df.loc[df['Ejection Fraction'] == 'Normal', 'Ejection Fraction'] = '2'
	df.loc[df['Ejection Fraction'] == 'High', 'Ejection Fraction'] = '3'
	df['Ejection Fraction']= df['Ejection Fraction'].astype('int')

	df['Age'] = df['Age'].abs()
	df.reset_index(drop=True, inplace=True)

	return df


def create_traintest_sample(df, tsize):
	"""
	Split data into train and test sample base on test set ratio
	param df: data to be splited
	param tsize: test set ratio
	return X_train, X_test, y_train, y_test: train samples, test samples  
	"""
	# get columns name
	variables = df.columns[1:]
	target = ['Survive']
	
	# scale data
	names = df[variables].columns
	scaler = preprocessing.StandardScaler()
	scaled_df = scaler.fit_transform(df[variables])
	scaled_df = pd.DataFrame(scaled_df, columns=names)
    
	# split data to train & test 
	y = df[target]
	X_train, X_test, y_train, y_test = train_test_split(scaled_df, y, test_size=tsize, stratify = y)

	return X_train, X_test, y_train, y_test


def select_traintest_ratio():
	"""
	Train and test sample ratio selection
	Valid keypress input:"1", "2", "3", "4", "5","6"
	"1" => train_test ratio 40%_60%
	"2" => train_test ratio 50%_50%
	"3" => train_test ratio 60%_40%
	"4" => train_test ratio 70%_30%
	"5" => train_test ratio 80%_20%
	"6" => train_test ratio 90%_10%
	Default selection => train_test ratio 80%_20%
	"""
	#split data to training and test set
	print()
	print('Please select train sample to test sample ratio:')
	for i in range(6):
		print(i+1,'-',i*10+40,'% to',60-i*10, '%')
	value=0 
	valid_input = ['1', '2', '3', '4', '5','6']
	value = input()
	tsize =round((((7-int(value))*0.1) if value in valid_input else 0.2),1)
	print(100-int(tsize*100),'% to',(int(tsize*100)), '% is selected')
	print('Please proceed to select algorithm')
	
	return tsize


def print_page(ratio):
	"""
	Print menu for user to select model for training and testing 
	pram ratio: test sample ratio  
	"""
	print()
	print('Train sample to test sample ratio is:',int(100-ratio*100),'% to',int(ratio*100),'%')
	print("Please select the algorithm:")
	print("1. Logistic Regression")
	print("2. GaussianNB")
	print("3. Decision Tree Classifier")
	print("4. Random Forest Classifier")
	print("5. Train & Test ratio selection")
	print("6. END program")


def main():
	"""
	Retrieve and prepare data from /data/survive.db
	Selection of model for training and testing
	Valid keypress input:"1", "2", "3", "4", "5","6"
	"1" => Logistic Regression model selected
	"2" => Gaussian Naive Bayes model selected
	"3" => Decision Tree model selected
	"4" => Random Forest model selected
	"5" => Select train and test ratio
	"6" => Exit program
	"""
 	# query survive.db
	con = sqlite3.connect("./data/survive.db")
	df = pd.read_sql_query("SELECT * FROM survive", con)
	
	# clean up data
	df = sample_cleanup(df)
	
	value = 0
	tsize = 0.2
	while(value != '6'):
		print_page(tsize)
		value = input()
		X_train, X_test, y_train, y_test = create_traintest_sample(df,tsize)
		print()
		if ((value=='1') or (value =='2') or (value =='3') or (value == '4')):
			model(X_train, X_test, y_train, y_test, value)
		elif (value == '5'):
			tsize = select_traintest_ratio()
			X_train, X_test, y_train, y_test = create_traintest_sample(df,tsize)
		elif (value == '6'):
			print("Program ends.")
			break;
		else:
			print("Invalid selection")


if __name__ == '__main__':
	main()
