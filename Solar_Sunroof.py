import pandas as pd
import numpy as np
#from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv('Solar Sunroof angle.csv')

X_train =data.iloc[:,1:3]
y_train =data.iloc[:,3]
X_train.head()

rdf=RandomForestRegressor(n_estimators=10,random_state=43,max_depth=3)
rdf.fit(X_train,y_train)
print(rdf.feature_importances_)
data2 =pd.read_csv('Solar test.csv')
X_test =data2.iloc[:,1:3]
prediction_1 = rdf.predict(X_test)
prediction_1
prediction_1=prediction_1.round()
data2['angle']=prediction_1
data2.to_csv('Solar_energy_test.csv')

data_1=pd.read_csv('Solar Sunroof energy.csv')

X_train1=data_1.iloc[:,1:4]
y_train1=data_1.iloc[:,4]
y_train1.head()

rdf1=RandomForestRegressor(n_estimators=10,random_state=43,max_depth=5)
rdf1.fit(X_train1,y_train1)
print(rdf1.feature_importances_)

data3 =pd.read_csv('Solar_energy_test.csv')
X_test =data3.iloc[:1,2:5]
X_test

prediction2 = rdf1.predict(X_test)
prediction2

def installation_cost_and_yeartorecover(area,cost_per_msquare,energy_output,cost_per_unit):
    installation_cost=0.6*area*cost_per_msquare
    profit_per_year=0.6*area*energy_output*cost_per_unit
    years_to_recover=cost_per_msquare/(energy_output*cost_per_unit)
    return installation_cost,years_to_recover[0],profit_per_year[0]

area=float(input())
cost_per_msquare=15000.0                              #for 200 W solar panel
energy_output=prediction2
cost_per_unit=float(input())

installation_cost_and_yeartorecover(area,cost_per_msquare,energy_output,cost_per_unit)    
#The conventional wisdom (in the Northern Hemisphere) is that the best direction to face solar panels is south, since that is generally where they’d receive the most sunlight. However, the electricity system is not as simple as it sometimes seems, and the best direction to face solar panels may actually be west!
#Because of lower silicon purity, polycrystalline solar panels are not quite as efficient as monocrystalline solar panels.