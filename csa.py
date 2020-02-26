# csa V1.0
# 11/10/2019
# Spencer Faber (sfaber@dropletmeasurement.com)
# Description: csa() accepts lists of MIP, POPS, and CDP data (mip_data, pops_data, cdp_data). Returns time data lists
# were passed, latitude, longitude, seed score (0 - 10), and seed switch (binary). csa() uses a lookup table with
# inputs of averaged POPS numb. conc. (pops_n_ave) and averaged MIP vertical wind (mip_w_ave) to produce a seedability
# score (seed_score). Multiple seed score lookup tables are used with different LCL temperatures. Seedability must
# exceed a seedability threshold (seed_score_thresh) for a set distance (seed_dist_thresh) before the seed command
# is passed (seed_switch = 1).
# Seed score tables are read in from the directory path set by css_table_path. Tables are named based on their LCL
# temperature (C).

import numpy as np
import pandas as pd
from datetime import datetime
import math, glob

# -------------------------------------------------------Settings-------------------------------------------------------
css_table_path = r'lookup_tables'  # Path to cloud seed score csv
cdp_sa = 0.253  # CDP sample area (mm^2)
cdp_n_cloud_thresh = 5  # CDP conc required to declare in cloud (cm-3)
seed_score_thresh = 8 # Seed score threshold to declare area seedable
seed_dist_thresh = 40 # Length threshold needed to trigger seeding event (m)
ave_int = 3 # Number of vertical wind/POPS N measurements to average
lcl_t = 0 # LCL temperature (C)
# -----------------------------------------------------End Settings-----------------------------------------------------


# -------------------------------------------------Import Lookup Tables-------------------------------------------------
# Assigns .css lookup tables in directory located at css_table_path to tables dictionary
table_path = glob.glob(css_table_path+'\*')
tables = {}
for item in table_path:                                  # Loop through table_path list and assign imported .css
    df_name = float(item.split('.csv')[0].split('\\')[-1])  # tables to pandas dataframes in tables dictionary.
    tables[df_name] = pd.read_csv(item, index_col=0)     # Keys are set from .css table names (lcl temp).

tables_lcl_t_float = np.empty(0)
for key in tables.keys():                                    # Assign tables keys to numpy array. Useful for selecting
    tables_lcl_t_float = np.append(tables_lcl_t_float, key)  # which table to index in function csa.
# -----------------------------------------------END Import Lookup Tables-----------------------------------------------


# --------------------------------------------csa (Cloud Seeding Algorithm)---------------------------------------------
# In: Lists of parameters output by MIP, POPS, CDP (mip_data, pops_data, cdp_data)
# Out:
# ----------------------------------------------------------------------------------------------------------------------
def csa(mip_data, pops_data, cdp_data):
    global cdp_sa  # CDP sample area (mm^2)
    global cdp_n_cloud_thresh  # CDP conc required to declare in cloud (cm-3)
    global tables  # Dictionary of lookup tables (as pandas dataframes) for different LCL temp
    global tables_lcl_t_float # Numpy array of tables keys. Used for selecting best table based on LCL temp
    tas = mip_data[27]  # True Air Speed (m/s)
    pitch = mip_data[11]  # UAV pitch (deg)
    wind_w = mip_data[39]  # Vertical Wind Component (m/s)
    lat = mip_data[5]  # Latitude (deg)
    long = mip_data[6]  # Longitude (deg)
    # time = mip_data[2]  # MIP GPS Time (ms)
    time = (datetime.now() - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
    pops_n = pops_data[3]  # POPS number concentration (cm^-3)

    # ---------------------------------------Calculate CDP Number Conc.---------------------------------------
    # Calculate CDP number conc. using CDP binned counts, CDP sample area (constant), MIP True Air Speed

    cdp_c = np.sum(cdp_data[15:45])  # Total CDP counts from bin counts
    cdp_sv = (cdp_sa / 100) * (tas * 100)  # CDP sample volume (cm^3)
    cdp_n = round(cdp_c / cdp_sv, 3)  # CDP number concentration rounded to 3 dec places (#/cm^3)
    # ---------------------------------------------------------------------------------------------------------

    # ------------------------------------------Evaluate Seedability-------------------------------------------
    # Evaluate seedability using lookup table, Ave MIP vertical wind, Ave POPS number conc.

    nearest_lcl_t = (np.abs(tables_lcl_t_float - lcl_t)).argmin() # Select lookup table to use based on LCL temp
    table_df = tables[tables_lcl_t_float[nearest_lcl_t]] # Assign best lookup table to table_df from tables dictionary

    # Call func ave_measurements to return average/standard dev of wind_w and pops_n
    wind_w_ave, wind_w_std, pops_n_ave, pops_n_std = ave_measurements(wind_w, pops_n)

    if np.isfinite(wind_w_ave): # Index lookup table to get seed score if wind_w_ave is finite
        x_ind = (np.abs(table_df.index - wind_w_ave)).argmin()  # Find index with min difference w/ wind_w_ave
        y_ind = (np.abs(table_df.columns.astype(float) - pops_n_ave)).argmin() # Find index with min diff. w/ pops_n_ave
        seed_score = table_df.iloc[x_ind, y_ind]
    else: # If wind_w_ave is not finite, assign 0 as seed score
        seed_score = 0

    seed_switch = seed_scale(seed_score, tas, pitch) # Get binary seed flag

    return [time, lat, long, seed_score, seed_switch]
# ------------------------------------------END csa (Cloud Seeding Algorithm)-------------------------------------------


# ------------------------------------------------------seed_scale------------------------------------------------------
# In: seed_score (0-10), true air speed, aircraft pitch
# Out: Binary seed switch

# Function calculates length scale of region with seed score > seed_score_thresh. seed_switch = 1 is returned if region
# length is > seed_dist_thresh.
# ----------------------------------------------------------------------------------------------------------------------
seedable_dist = 0
def seed_scale(seed_score, tas, pitch):
    global seed_score_thresh
    global seed_dist_thresh
    global seedable_dist

    if seed_score >= seed_score_thresh:
        x_dist = tas * math.cos(math.radians(pitch))
        seedable_dist = seedable_dist + x_dist
    else:
        seedable_dist = 0

    if seedable_dist >= seed_dist_thresh:
        seed_switch = 1
    else:
        seed_switch = 0

    return seed_switch
# ----------------------------------------------------END seed_scale----------------------------------------------------


# ---------------------------------------------------ave_measurements---------------------------------------------------
# In: MIP Vertical Wind (wind_w), POPS Num. Conc. (pops_n)
# Out: Average and Standard Deviation of MIP Vertical Wind/POPS Num. Conc. over interval defined by option ave_int

# Function will only consider updrafts (positive wind_w). It will return NaN for ave/standard dev of wind_w if
# fewer than 3 updraft measurements have been reported over ave_int
# ----------------------------------------------------------------------------------------------------------------------
wind_w_ave_arr = np.zeros(ave_int)  # Create empty numpy arrays for wind_w and pops_n measurements. Fill them
pops_n_ave_arr = np.zeros(ave_int)  # with NaN.
wind_w_ave_arr[:] = np.NaN          #
pops_n_ave_arr[:] = np.NaN          #

def ave_measurements(wind_w, pops_n):
    global wind_w_ave_arr
    global pops_n_ave_arr

    # If program just started we need to sequentially replace NaNs in ave_arr with measurements
    if len(np.argwhere(np.isnan(wind_w_ave_arr))) > 0:             # Test if wind_w_ave_arr contains NaN
        nan_min_i = np.min(np.argwhere(np.isnan(wind_w_ave_arr)))  # Find index where to assign new measurements
        wind_w_ave_arr[nan_min_i] = wind_w                         # Add latest wind_w to array at smallest i with NaN

        pops_n_ave_arr[nan_min_i] = pops_n                         # Add latest pops_n to array at smallest i with NaN

    # If ave_arr are filled with measurements, drop oldest measurement in ave_arr and add new values to last index
    # to do stats on latest n measurements
    else:
        wind_w_ave_arr[0] = np.NaN                   # Replace oldest measurement in wind_w_ave_arr with NaN
        wind_w_ave_arr = np.roll(wind_w_ave_arr,-1)  # Roll wind_w_ave_arr so that NaN is last index
        wind_w_ave_arr[-1] = wind_w                  # Replace NaN with latest measurement

        pops_n_ave_arr[0] = np.NaN
        pops_n_ave_arr = np.roll(pops_n_ave_arr, -1)
        pops_n_ave_arr[-1] = pops_n

    i_updraft = np.where(np.nan_to_num(wind_w_ave_arr) > 0) # Find indices where wind_w is updraft

    if len(i_updraft[0]) > 1:                               # If there are at least 2 updraft measurements,
        wind_w_ave = np.mean(wind_w_ave_arr[i_updraft])     # calc stats
        wind_w_std = np.std(wind_w_ave_arr[i_updraft])
    else:
        wind_w_ave = np.NaN                                 # If there are fewer than 2 updraft measurements,
        wind_w_std = np.NaN                                 # return NaN for wind_w average/standard deviation

    pops_n_ave = np.nanmean(pops_n_ave_arr)                 # Calc average pops_n
    pops_n_std = np.nanstd(pops_n_ave_arr)                  # Calc standard deviation pops_n

    return [wind_w_ave, wind_w_std, pops_n_ave, pops_n_std]
# -------------------------------------------------END ave_measurements-------------------------------------------------
