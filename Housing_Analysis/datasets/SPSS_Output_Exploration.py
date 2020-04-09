### Imports
import numpy as np # Used for dataframe manipulation & linear algebra
import pandas as pd # Used for creating and modifying dataframes
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()



File_Name = "C:/Users/JAKEFREDRICH/Desktop/Backup/Chart Exploration/SPSS_Totals2.csv"
df = pd.read_csv(File_Name) # Read File


# Key Fields for total DF
total_df = df[['Country','Year','Month','Revenue','Gross Profit', 'USD', 'SLF', 'Expense', 'Royalty', '5XX', 'DE', 'Reported PTI', 'Earnings','Revenue_CC','Gross Profit_CC', 'USD_CC', 'SLF_CC', 'Expense_CC', 'Royalty_CC', '5XX_CC', 'DE_CC', 'Reported PTI_CC', 'Earnings_CC', 'SALES(Rev)','SERVICES(Rev)','SOFTWARE(Rev)','MAINTENANCE(Rev)','RENTAL & FINANCE (Rev)','Revenue_CC_ACC']]
total_df = total_df.dropna()
total_df = total_df.astype({'Revenue': 'int', 'Gross Profit': 'int', 'USD': 'int', 'SLF' : 'int', 'Expense' : 'int', 'Royalty' : 'int', '5XX' : 'int', 'DE' : 'int' , 'Reported PTI' : 'int', 'Earnings' : 'int','Revenue_CC': 'int', 'Gross Profit_CC': 'int', 'USD_CC': 'int', 'SLF_CC' : 'int', 'Expense_CC' : 'int', 'Royalty_CC' : 'int', '5XX_CC' : 'int', 'DE_CC' : 'int' , 'Reported PTI_CC' : 'int', 'Earnings_CC' : 'int'})
total_df = total_df[~total_df.Country.str.contains("LA PRICING/FAS8 ADJ", na=False)] #Remove LA PRICING/FAS8 ADJ
total_df = total_df[~total_df.Country.str.contains("ISRAEL PRICING ADJ", na=False)] #Remove LA PRICING/FAS8 ADJ

''' Clean up UCC Data and break into BAU and FAS8 dfs '''
ucc_country_df = total_df[['Country','Year','Month','Revenue','Gross Profit', 'USD', 'SLF', 'Expense', 'Royalty', '5XX', 'DE', 'Reported PTI', 'Earnings']]
ucc_country_df = ucc_country_df[~ucc_country_df.Country.str.contains("TOTAL", na=False)] #Remove Total
ucc_country_df = ucc_country_df[~ucc_country_df.Country.str.contains("OTHER", na=False)] #Remove Other categories
ucc_country_df = ucc_country_df[~ucc_country_df.Country.str.contains("IBM", na=False)] #Remove Countries with IBM in name
ucc_country_df = ucc_country_df[~ucc_country_df.Country.str.contains("TPG", na=False)] #Remove TPG categories
ucc_country_df = ucc_country_df[~ucc_country_df.Country.str.contains("DVU", na=False)] #Remove DVU categories


indexNames = ucc_country_df[ucc_country_df['Revenue'] == 0].index  # Get names of indexes for which Revenue = 0
ucc_country_df.drop(indexNames, inplace = True)  # Delete these row indexes from dataFrame


FAS8_And_Exceptions = ['HUNGARY', 'ROMANIA', 'RUSSIA', 'ARGENTINA', 'CHILE', 'COLOMBIA', 'MEXICO', 'PERU', 'URUGUAY', 'VENEZUELA', 'BRAZIL', 'IBM BRAZIL LEASING', 'ISRAEL (SEMEA SUB)']
FAS8_UCC_df = ucc_country_df[ucc_country_df['Country'].isin(FAS8_And_Exceptions)]  # FAS8 and Exception UCC Data
BAU_UCC_df = ucc_country_df[~ucc_country_df['Country'].isin(FAS8_And_Exceptions)]  # BAU UCC Data


print(BAU_UCC_df.head())