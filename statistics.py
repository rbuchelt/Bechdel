import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('bechdel_code.csv')
data.dropna(inplace=True)

data2 = data['binary']

data3 = data2.groupby(data2).count()

data3[0], data3[1]

PASS = (data3[1]/((data3[0]+data3[1])))*100
PASS = int(PASS)
print(f'The percentage of films that passed the test is: {PASS}%')

filter_mask_fail = data['binary']=='FAIL'
df_fail = data[filter_mask_fail]

filter_mask_pass = data['binary']=='PASS'
df_pass = data[filter_mask_pass]

df_fail = df_fail.drop(['imdb', 'code', 'budget', 'domgross', 'intgross',
                        'budget_2013', 'domgross_2013', 'intgross_2013'],1)

df_pass = df_pass.drop(['imdb', 'code', 'budget', 'domgross', 'intgross',
                        'budget_2013', 'domgross_2013', 'intgross_2013'],1)

df_fail = df_fail.sort_values(by='year')
df_pass = df_pass.sort_values(by='year')

df_fail = df_fail.groupby('year')['binary'].count()
df_pass = df_pass.groupby('year')['binary'].count()
df_fail = pd.DataFrame(df_fail)
df_pass = pd.DataFrame(df_pass)

plt.plot(df_fail)
plt.ylabel('FAIL')
plt.xlabel('YEAR')
plt.show()

plt.plot(df_pass)
plt.ylabel('PASS')
plt.xlabel('YEAR')
plt.show()