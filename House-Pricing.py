import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_boston
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression

#Load Boston data
boston = load_boston()
boston_project1 = pd.DataFrame(boston.data, columns=boston.feature_names)
boston_project1['House Price per Squar fit']= boston.target
x = boston.data
y = boston.target
#Test error rate
reg = LinearRegression()
CV_scores = cross_val_score(reg,x,y,cv=25)
cross_validation = np.mean(CV_scores)

print('******************************************************************************')
print(' The rate of Error is :  ', format(cross_validation))
print('******************************************************************************')

lasso = Lasso(alpha=0.1, normalize=True)
lasso.fit(x,y)
lasso_coeff = lasso.coef_
plt.plot(range(13),lasso_coeff)
plt.xticks(range(13),boston.feature_names)
plt.ylabel('Coefficients')
plt.show()

print('******************************************************************************')
print('Note : The Graph shows that the most important feature is the number of rooms:')
print('******************************************************************************')
print('                                                                              ')
print('*** TO GET THE PRICE OF THE NEW HOUSE, PLEASE ENTER INFORMATON ***')

j=0
print("Do you want to enter the Average rates instead of entering the new rates? (Yes & No)")

G=input()
print(type(G))
G=G.upper()
print("You have chosen", G)

    
    



infor = np.array([3.61,11.36,11.13,0.06,0.55,6.28,68.57,3.79,9.54,408.23,18.45,356.67,12.65])
Name = np.array(['CRIME rate',
               'proportion of residential land zoned for lots over 25000 sq.ft.',
               'proportion of non-retail business acres per town',
               'harles River dummy variable (= 1 if tract bounds river; 0 otherwise',
               'nitric oxides concentration (parts per 10 million)',
               'average number of rooms per dwelling',
               'proportion of owner-occupied units built prior to 1940',
               'weighted distances to five Boston employment centres',
               'index of accessibility to radial highways',
               'full-value property-tax rate per $10,000',
               'pupil-teacher ratio by town',
               '1000(Bk - 0.63)^2 where Bk is the proportion of blacks',
               '% lower status of the population'])

if G =="NO":   
    for i,k in enumerate(Name) :
        print( "The average of ",k, "is < ", infor[i],"> please enter the new rate: ")
        INF=input()
        infor[i]=INF
        i=i+11
else:
    print("Note : * The total Price for the New House predicted based on the Avrage rates * ")       

X_New=infor[:]
#SQ=0
X_New=X_New.reshape(1,-1)
reg.fit(x,y)

Y_Predict= reg.predict(X_New)
print("Please enter the SQUARE FOOT of the new House : ")

S = input()
S=float(S)
Y=Y_Predict[0]
Total_Price = S*Y*10
Total_Price =int(Total_Price)
print("The total Price for the New House is around : $",Total_Price)

sleep(1)
 #print("The New house information is : " , infor[:])





