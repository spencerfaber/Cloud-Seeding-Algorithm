import pandas as pd
from csa import csa

# counter for main loop below
count = 0

# Read MIP, POPS, and CDP CSVs in test_data folder
mip_data_df = pd.read_csv(r'test_data\mip_data.csv')
pops_data_df = pd.read_csv(r'test_data\pops_data.csv')
cdp_data_df = pd.read_csv(r'test_data\cdp_data.csv')

while count < len(mip_data_df):
    # Get one row of MIP, POPS, and CDP data. Convert row to python list (1D array)
    mip_data = mip_data_df.iloc[count, :].values.tolist()
    pops_data = pops_data_df.iloc[count, :].values.tolist()
    cdp_data = cdp_data_df.iloc[count, :].values.tolist()

    # Call the main cloud seeding algorithm function with MIP, POPS, and CDP data as inputs
    csa_out = csa(mip_data, pops_data, cdp_data)
    # Print what the cloud seeding algorithm returned
    print(csa_out)
    count += 1
