#%%
import matplotlib.dates as mdates
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas.plotting as pp
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, plot_predict
#%%
df2 = pd.read_csv("https://api.coronavirus.data.gov.uk/v2/data?areaType=nation&areaCode=E92000001&metric=newCasesBySpecimenDate&metric=newAdmissions&format=csv", parse_dates=[3])
df = df2.rename({"newCasesBySpecimenDate":"C","newAdmissions":"H"},axis='columns')
df.set_index('date',inplace=True)
df.sort_index(inplace=True)
df = df['2020-10-05':'2021-07-12']
df.H.plot()
#%%
dr = df.resample('W-Mon').sum()
#%%
dl = np.log(dr)
# plt.plot(dl.index,dl.H)
# plt.plot(dl.index,dl.C.shift(1))

# plt.plot(dl.H,dl.C)
#%%
plt.figure(constrained_layout=False, figsize=(5.5, 3.2))
np.exp(-dl.H+dl.C.shift(1))[2:].plot()
plt.xlabel('')
plt.ylabel('Fälle pro Krankenhausaufnahme')
plt.title('Fälle pro Krankenhausaufnahme in England')
# plt.text(0.9,0.55,'Cornelius Roemer')
plt.text(0.74,-0.20,'Daten: UK Govt, Graph: @CorneliusRoemer',horizontalalignment='center',
     verticalalignment='center', transform = plt.gca().transAxes,fontdict={'size':8})
plt.ylim(bottom=0, top=50)
plt.tight_layout(pad=2)
plt.savefig('cases_per_admission.png',dpi=600,transparent=False)
#%%
# d1 = np.log10(dr.H).diff(1).dropna()
# #%%
# d1.plot()
# #%%
# model = ARIMA(dr.H, order=(1,0,0),exog=dr.C)
# model_fit = model.fit()
# print(model_fit.summary())
# #%%
# # Plot residual errors
# residuals = pd.DataFrame(model_fit.resid)
# fig, ax = plt.subplots(1,2)
# residuals.plot(title="Residuals", ax=ax[0])
# residuals.plot(kind='kde', title='Density', ax=ax[1])
# plt.show()
# #%%
# # Actual vs Fitted
# fig, ax = plt.subplots()
# ax = dr.H.plot(ax=ax)
# plot_predict(model_fit,'2021-05-16', '2021-07-11',exog=dr.C,dynamic=True,ax=ax)
# #%%
# d1
# #%%
# # pp.autocorrelation_plot(df.H)
# pp.autocorrelation_plot(df.H)
# model = ARIMA(df.C, order=(5,7,1))
# model_fit = model.fit()
# # summary of fit model
# print(model_fit.summary())
# # line plot of residuals
# residuals = pd.DataFrame(model_fit.resid)
# residuals.plot()
# plt.show()
# # density plot of residuals
# residuals.plot(kind='kde')
# plt.show()
# # summary stats of residuals
# print(residuals.describe())
# # %%
# # # df.plot(x="newCasesBySpecimenDate",y="newAdmissions") 

# # df.newAdmissions[-70:-40]
# # # %%
# # # %%
# # # Data for plotting
# # df.set_index('date',inplace=True)
# # #%%
# # df.sort_index(inplace=True)
# # d = df[-200:-5]
# # #%%
# # d
# # #%%

# # fig, ax1 = plt.subplots()

# # ax2 = ax1.twinx()
# # ax1.plot(d.index, d.newCasesBySpecimenDate, 'g-')
# # ax2.plot(d.index, d.newAdmissions, 'b-')
# # fig.autofmt_xdate()
# # locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
# # formatter = mdates.ConciseDateFormatter(locator)
# # ax1.xaxis.set_major_locator(locator)
# # ax1.xaxis.set_major_formatter(formatter)
# # plt.show()


# # # %%

# # %%
# df

# # %%
