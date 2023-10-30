import joblib

randomForest = joblib.load('random-forest-regression')

def prediction(DAILY_YIELD, TOTAL_YIELD, AMBIENT_TEMPERATURE, MODULE_TEMPERATURE,IRRADIATION, DC_POWER):
    AC_POWER = randomForest.predict([[DAILY_YIELD, TOTAL_YIELD, AMBIENT_TEMPERATURE, MODULE_TEMPERATURE,IRRADIATION, DC_POWER]])
    return AC_POWER

yourPrediction = prediction(1516.6,	1.6609,	27.863,	36.305,	0.442,	699.373)

print(yourPrediction)