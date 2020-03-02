import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from csa import csa

count = 0
mip_data_df = pd.read_csv(r'test_data\mip_data.csv')
pops_data_df = pd.read_csv(r'test_data\pops_data.csv')
cdp_data_df = pd.read_csv(r'test_data\cdp_data.csv')

foo = np.zeros((len(mip_data_df),9))

while count < len(mip_data_df):
    mip_data = mip_data_df.iloc[count, :].values.tolist()
    pops_data = pops_data_df.iloc[count, :].values.tolist()
    cdp_data = cdp_data_df.iloc[count, :].values.tolist()

    bar = csa(mip_data, pops_data, cdp_data)
    # foo[count,:] = bar
    print(bar)
    count += 1

plt.plot(foo[:,0],foo[:,6])
plt.show()
print('')