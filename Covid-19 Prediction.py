import numpy as np
import pandas as pd

#Visualization libraries
import matplotlib.pyplot as plt

#machine learning library
import sklearn
from sklearn.linear_model import LinearRegression

#importing preprocessing methods from sklearn
from sklearn.model_selection import train_test_split


#Reading coronavirus confirmed cases across the world

confirmed_case = pd.read_csv('time_series_covid19_confirmed_global.csv')


confirmed_case.head()


#Reading coronavirus recovered cases across the world

recovered_case = pd.read_csv('time_series_covid19_recovered_global.csv')
recovered_case.head()


cols = confirmed_case.keys()
print(cols)


confirmed = confirmed_case.loc[:,cols[4]:cols[-1]]
confirmed



recovered = recovered_case.loc[:,cols[4]:cols[-1]]
recovered



#Reading coronavirus deaths reported across the world 

deaths_reported=pd.read_csv("time_series_covid19_deaths_global.csv")

deaths_reported.head()


deaths = deaths_reported.loc[:,cols[4]:cols[-1]]
deaths


dates= confirmed.keys()

dates



world_cases=[]

total_deaths=[]

mortality_rate=[]

total_recovered=[]

india_cases=[]

for i in dates:
  confirmed_sum = confirmed[i].sum()
  # print(confirmed[i])
  death_sum=deaths[i].sum()
  recovered_sum=recovered[i].sum()

  world_cases.append(confirmed_sum)
  total_deaths.append(death_sum)
  mortality_rate.append(death_sum/confirmed_sum)
  total_recovered.append(recovered_sum)
  # india_cases.append(confirmed_case[confirmed_cases['Country/Region']=='India'][i].sum())




print(confirmed_sum)
print(death_sum)
print(recovered_sum)
print(world_cases)



#changing dates into days
v = 1
day_date = []
for i in range(len(dates)):
  v = i * 1
  day_date.append(v)

# day_date
  



#plotting the cases (confirmed, recovered and deaths) with time
plt.figure(figsize=(20,12))
plt.plot(day_date,world_cases)
plt.plot(day_date,total_recovered, color='green')
plt.plot(day_date,total_deaths,color='red')
plt.title("Coronavirus Cases, deaths and recovered with time", size=30)
plt.xlabel("Days",size=30)
plt.ylabel("Count of Cases",size=30)
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()





#check each day cases in world
def each_day_increase(records):
  d=[]
  for i in range(len(records)):
    if i==0:
      d.append(records[0])
    else:
      d.append(records[i]-records[i-1])
  return d



world_daily_increase = each_day_increase(world_cases)
world_daily_increase




print(type(world_cases))



#linear regression and unsupervised learning
#converting the list to array
world_cases = np.array(world_cases).reshape(-1,1)
total_deaths=np.array(total_deaths).reshape(-1,1)
total_recovered=np.array(total_recovered).reshape(-1,1)

days= np.array(day_date).reshape(-1,1)

print(days.shape)
Execution output from Jun 21, 2020 8:09 AM
	Stream
		(150, 1)



days_in_future = 10
future_forecast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1,1)

adjusted_dates = future_forecast[:-10]



# future_forecast
# adjusted_dates



x_train_confirmed,x_test_confirmed,y_train_confirmed,y_test_confirmed = train_test_split(days,world_cases,test_size=0.25,shuffle=False)




from sklearn.preprocessing import PolynomialFeatures
poly= PolynomialFeatures(degree=3)
poly_x_train_confirmed = poly.fit_transform(x_train_confirmed)
poly_x_test_confirmed = poly.fit_transform(x_test_confirmed)
poly_future_forcast=poly.fit_transform(future_forecast)




model = LinearRegression()
model.fit(poly_x_train_confirmed,y_train_confirmed)
Execution output from Jun 21, 2020 8:29 AM
	text/plain
		LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




test_pred = model.predict(poly_x_test_confirmed)   #y_test_confirmed



predictions = model.predict(poly_future_forcast)


from sklearn.metrics import mean_squared_error,mean_absolute_error




#mean square error
mean_squared_error(test_pred,y_test_confirmed)


mean_absolute_error(test_pred,y_test_confirmed)



plt.plot(y_test_confirmed,color='red')
plt.plot(test_pred)




