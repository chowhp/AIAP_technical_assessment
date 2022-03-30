import pandas as pd
import sys
import sqlite3
from matplotlib import pyplot as plt
from sklearn import datasets, linear_model, preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

def logistic_regression(X_train, X_test, y_train, y_test):
	log = linear_model.LogisticRegression()
	log_model = log.fit(X_train, y_train.values.ravel())
	log_predictions = log.predict(X_test)
	print('Logistic Regression slected and prediction score is %f\n' % log.score(X_test, y_test))

def decision_tree_classifier(X_train, X_test, y_train, y_test):
	dtree = DecisionTreeClassifier(criterion = 'entropy')
	dtree_model = dtree.fit(X_train,y_train)
	dtree_predictions = dtree.predict(X_test)	
	print('Decision Tree Classifier selected and prediction score is %f\n' % dtree.score(X_test, y_test))

def random_forest_classifier(X_train, X_test, y_train, y_test):
	forest = RandomForestClassifier(n_estimators=10, criterion='entropy')
	forest_model = forest.fit(X_train, y_train.values.ravel())
	forest_predictions = forest.predict(X_test)
	print('Random Forest Classifier selected and prediction score is %f\n' % forest.score(X_test, y_test))

def gaussian_nb(X_train, X_test, y_train, y_test):
	gaus = GaussianNB()
	gaus_model = gaus.fit(X_train, y_train.values.ravel())
	gaus_predictions = gaus.predict(X_test)
	print('GaussianNB selected and prediction score is %f\n' % gaus.score(X_test, y_test))



def sample_cleanup(df):
	# dropna
	df.dropna(inplace = True)
	
	# drop duplicates
	df.drop_duplicates(subset = 'ID', keep = 'first', inplace = True)

	# drop features 
	#df.drop('Diabetes', axis=1, inplace=True)
	df.drop('Favorite color', axis=1, inplace=True)
	df.drop('Height', axis=1, inplace=True)
	df.drop('ID', axis=1, inplace=True)

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


def sample_split(df, tsize):
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
	X_train, X_test, y_train, y_test = train_test_split(scaled_df, y, test_size=tsize)

	return X_train, X_test, y_train, y_test

def split_selection():
	#split data to training and test set
	print()
	print('Please select train set to test set ratio:')
	for i in range(6):
		print(i+1,'-',i*10+40,'% to',60-i*10, '%')
	value=0 
	check = 1
	if(check):
		value = input()
		if (value =='1'):
			tsize = 0.60
		elif (value =='2'): 
			tsize = 0.50
		elif (value =='3'):
			tsize = 0.40
		elif (value =='4'): 
			tsize = 0.30
		elif (value =='5'): 
			tsize = 0.20
		elif (value =='6'): 
			tsize = 0.10
		else: 
			tsize = 0.20
			check = 0
    
		if(check):
			print(int(100-tsize*100),'% to',int(tsize*100), '% is selected')
			print('please proceed to select algorithm')
		else:
			print('Default train set(80%) to test set(20%) is selected')
	
	return tsize


def main():
  # query survive.db
	con = sqlite3.connect("./data/survive.db")
	df = pd.read_sql_query("SELECT * FROM survive", con)
	
	# clean up data
	df = sample_cleanup(df)
	
	print()
	print("Please select the algorithm (default train/test is 80% to 20%):")
	print("1. Logistic Regression")
	print("2. GaussianNB")
	print("3. Decision Tree Classifier")
	print("4. Random Forest Classifier")
	print("5. Train set selction")
	print("6. END program")
	value = 0
	tsize = 0.2
	while(value != '6'):
		value = input()
		X_train, X_test, y_train, y_test = sample_split(df,tsize)
		if (value=='1'):
			logistic_regression(X_train, X_test, y_train, y_test)
		elif (value == '2'):
			gaussian_nb(X_train, X_test, y_train, y_test)
		elif (value == '3'):
			decision_tree_classifier(X_train, X_test, y_train, y_test)
		elif (value == '4'):
			random_forest_classifier(X_train, X_test, y_train, y_test)
		elif (value == '5'):
			tsize = split_selection()
			X_train, X_test, y_train, y_test = sample_split(df,tsize)
		elif (value == '6'):
			print("Program ends.")
		else:
			print("Invalid selection")


if __name__ == '__main__':
	main()
