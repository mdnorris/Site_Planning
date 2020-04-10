# base file as of 3/9/20, could use cudf, dask, cupy, etc in future

import timeit
import numpy as np
import pandas as pd
import smtplib

init_time = timeit.default_timer()
# SECTION 1: DATA INPUTS

sites = pd.read_csv('sites_12057_rv1.csv', delimiter=',')
hr_sites = pd.read_csv('usage_12057.csv', delimiter=',')
npv_values = pd.read_csv('npv_high_vz_prop.csv', delimiter=',')
lte_params = pd.read_csv('lte_assumptions.csv', delimiter=',')

npv_threshold = 0
# 37, 61, 91, 127, 169, 217, 271, 331, 397
bin_prox = 17
sinr_deg = 10

# sites = sites.sample(frac=.5, random_state=1)
# hr_sites = hr_sites.sample(frac=.5, random_state=1)


print('12057 seg 2 speed test')

# print(sites['segment'].value_counts())
# sites = sites[sites.segment == 1]
sites = sites[sites.segment == 2]
# sites = sites[sites.segment == 3]
# sites = sites[sites.segment == 4]
# sites = sites[sites.segment == 5]
# sites = sites[sites.segment == 6]
# sites = sites[sites.segment == 7]
# sites = sites[sites.segment == 8]

# 6    3559389
# 1    3462672
# 4    2716468
# 3    2705023
# 8    1295703
# 5     700095
# 7     695202
# 2     355279

# print((sites.isna().sum()) / len(sites) * 100)
sites = sites.dropna()

# print((hr_sites.isna().sum() / len(sites)) * 100)
hr_sites = hr_sites.dropna()

sites = sites.drop(['segment', 'sum_GBs_hourly_monthly', 'fips'], axis=1)

# comment out with WTG test case
hr_sites['hourly_usage_gb'] = hr_sites['GBs_hourly_monthly'] / 30.42
hr_sites = hr_sites.drop(['GBs_hourly_monthly'], axis=1)

# sites_roe = sites[sites.asset_type == 'ROE']
# sites_smb = sites[sites.morphology == 'SMB']
sites = sites[sites.asset_type == 'Pole']

initial_sites = pd.Series(len(sites['asset_id'].unique()))
# print('initial sites', end='  ')
# print(len(sites['asset_id'].unique()))

# sites = sites.drop(['segment', 'sum_GBs_hourly_monthly', 'fips'], axis=1)

sites = sites.rename(columns={'bin_10': 'GridName', 'asset_id': 'fict_site', 'asset_type': 'Type',
                              'path_loss_db': 'path_loss_umi_db', 'morphology': 'Morphology'})
hr_sites = hr_sites.rename(columns={'bin_10': 'GridName', 'hourly_usage_gb': 'Hour_GBs'})

# comment out with WTG test case
# sites_roe = sites_roe.rename(columns={'bin_10': 'GridName', 'asset_id': 'fict_site', 'path_loss_db': 'path_loss_umi_db'})

# init_hr_sites = hr_sites

# rb_bins = pd.merge(sites, hr_sites, on='GridName')
# rb_bins['bin_req_hr_mbps'] = (rb_bins['Hour_GBs'] * 8 * (2 ** 10)) / 3600
# sum_req = rb_bins.sort_values(by=['GridName', 'bin_req_hr_mbps'], ascending=[True, False])
# bin_req = sum_req.groupby('GridName')['bin_req_hr_mbps'].max().reset_index()

day_usage = hr_sites.groupby('GridName')['Hour_GBs'].sum().reset_index()
sites = pd.merge(sites, day_usage, on='GridName')

# NPV DATA

npv_values.columns = ['non', 'pole_du', 'pole_u', 'pole_s', 'pole_r', 'off_du', 'off_u',
                      'off_s', 'off_r', 'roe_du', 'roe_u', 'roe_s', 'roe_r',
                      'smb_du', 'smb_u', 'smb_s', 'smb_r']

# This creates time periods for xnpv function (for more information
st_date20 = pd.date_range(start='2020-06-30', periods=11, freq='12M')
st_date21 = pd.date_range(start='2021-06-30', periods=10, freq='12M')
st_date22 = pd.date_range(start='2022-06-30', periods=9, freq='12M')
st_date23 = pd.date_range(start='2023-06-30', periods=8, freq='12M')
st_date24 = pd.date_range(start='2024-06-30', periods=7, freq='12M')
st_date25 = pd.date_range(start='2025-06-30', periods=6, freq='12M')
st_date26 = pd.date_range(start='2026-06-30', periods=5, freq='12M')
st_date27 = pd.date_range(start='2027-06-30', periods=4, freq='12M')
st_date28 = pd.date_range(start='2028-06-30', periods=3, freq='12M')
st_date29 = pd.date_range(start='2029-06-30', periods=2, freq='12M')
st_date30 = pd.date_range(start='2030-06-30', periods=1, freq='12M')

disc_rt = .15

# LTE INPUTS

rx_sensitivity_db = -89.4


# SECTION 2: FUNCTIONS

def morph(in_type, in_morph):
    morph_code = 0
    if in_type == 'Pole' and in_morph == 'Dense Urban':
        morph_code = 1
    elif in_type == 'Pole' and in_morph == 'Urban':
        morph_code = 2
    elif in_type == 'Pole' and in_morph == 'Suburban':
        morph_code = 3
    elif in_type == 'Pole' and in_morph == 'Rural':
        morph_code = 4
    elif in_type == 'Off' and in_morph == 'Dense Urban':
        morph_code = 5
    elif in_type == 'Off' and in_morph == 'Urban':
        morph_code = 6
    elif in_type == 'Off' and in_morph == 'Suburban':
        morph_code = 7
    elif in_type == 'Off' and in_morph == 'Rural':
        morph_code = 8
    elif in_type == 'ROE' and in_morph == 'Dense Urban':
        morph_code = 9
    elif in_type == 'ROE' and in_morph == 'Urban':
        morph_code = 10
    elif in_type == 'ROE' and in_morph == 'Suburban':
        morph_code = 11
    elif in_type == 'ROE' and in_morph == 'Rural':
        morph_code = 12
    elif in_type == 'SMB' and in_morph == 'Dense Urban':
        morph_code = 13
    elif in_type == 'SMB' and in_morph == 'Urban':
        morph_code = 14
    elif in_type == 'SMB' and in_morph == 'Suburban':
        morph_code = 15
    elif in_type == 'SMB' and in_morph == 'Rural':
        morph_code = 16
    # else:
    #     print('error')
    return morph_code


def fin_arrays(code):
    cpx = np.empty([11, 1])
    opx = np.empty([11, 1])
    growth = np.empty([11, 1])
    mvno = np.empty([11, 1])
    npv_array = np.zeros([len(npv_values), 1])
    if code == 1:
        npv_array = npv_values.loc[:, 'pole_du']
    elif code == 2:
        npv_array = npv_values.loc[:, 'pole_u']
    elif code == 3:
        npv_array = npv_values.loc[:, 'pole_s']
    elif code == 4:
        npv_array = npv_values.loc[:, 'pole_r']
    elif code == 5:
        npv_array = npv_values.loc[:, 'off_du']
    elif code == 6:
        npv_array = npv_values.loc[:, 'off_u']
    elif code == 7:
        npv_array = npv_values.loc[:, 'off_s']
    elif code == 8:
        npv_array = npv_values.loc[:, 'off_r']
    elif code == 9:
        npv_array = npv_values.loc[:, 'roe_du']
    elif code == 10:
        npv_array = npv_values.loc[:, 'roe_u']
    elif code == 11:
        npv_array = npv_values.loc[:, 'roe_s']
    elif code == 12:
        npv_array = npv_values.loc[:, 'roe_r']
    elif code == 13:
        npv_array = npv_values.loc[:, 'smb_du']
    elif code == 14:
        npv_array = npv_values.loc[:, 'smb_u']
    elif code == 15:
        npv_array = npv_values.loc[:, 'smb_s']
    elif code == 16:
        npv_array = npv_values.loc[:, 'smb_r']
    # else:
    #     print('error')
    for m in range(11):
        cpx[m] = float(npv_array[m + 1])
    for m in range(11):
        opx[m] = float(npv_array[m + 12])
    for m in range(11):
        growth[m] = float(npv_array[m + 23])
    for m in range(11):
        mvno[m] = float(npv_array[m + 34])
    end_array = np.hstack((cpx, opx, growth, mvno))
    return end_array


def morph_array(code):
    if code == 1:
        return pole_du
    elif code == 2:
        return pole_u
    elif code == 3:
        return pole_s
    elif code == 4:
        return pole_r
    elif code == 5:
        return off_du
    elif code == 6:
        return off_u
    elif code == 7:
        return off_s
    elif code == 8:
        return off_r
    elif code == 9:
        return roe_du
    elif code == 10:
        return roe_u
    elif code == 11:
        return roe_s
    elif code == 12:
        return roe_r
    elif code == 13:
        return smb_du
    elif code == 14:
        return smb_u
    elif code == 15:
        return smb_s
    elif code == 16:
        return smb_r
    # else:
    #     print('error')


# these variables are used in case adjustments to xnpv functinos are made
day_diff = 0.0
cell_split = 2.0

# bld_npv functions are for build year, return 12 if negative
# so != 12 can be used to subset rows


def bld_npv21(gbs, code):
    array = morph_array(code)
    array = array[1:11, :]
    value = (gbs * 365 * array[0][3] * array[0][2] - array[0][1] - array[0][0]) / (1.15 ** 1)
    if value > 0:
        return 1
    elif value < 0:
        return 12
    else:
        return 11


def bld_npv22(gbs, code):
    array = morph_array(code)
    array = array[2:11, :]
    value = (gbs * 365 * array[0][3] * array[0][2] - array[0][1] - array[0][0]) / (1.15 ** 2)
    if value > 0:
        return 2
    elif value < 0:
        return 12
    else:
        return 11


def bld_npv23(gbs, code):
    array = morph_array(code)
    array = array[3:11, :]
    value = (gbs * 365 * array[0][3] * array[0][2] - array[0][1] - array[0][0]) / (1.15 ** 3)
    if value > 0:
        return 3
    elif value < 0:
        return 12
    else:
        return 11


def bld_npv24(gbs, code):
    array = morph_array(code)
    array = array[4:11, :]
    value = (gbs * 365 * array[0][3] * array[0][2] - array[0][1] - array[0][0]) / (1.15 ** 4)
    if value > 0:
        return 4
    elif value < 0:
        return 12
    else:
        return 11


def bld_npv25(gbs, code):
    array = morph_array(code)
    array = array[5:11, :]
    value = (gbs * 365 * array[0][3] * array[0][2] - array[0][1] - array[0][0]) / (1.15 ** 5)
    if value > 0:
        return 5
    elif value < 0:
        return 12
    else:
        return 11


def bld_npv26(gbs, code):
    array = morph_array(code)
    array = array[6:11, :]
    value = (gbs * 365 * array[0][3] * array[0][2] - array[0][1] - array[0][0]) / (1.15 ** 6)
    if value > 0:
        return 6
    elif value < 0:
        return 12
    else:
        return 11


def bld_npv27(gbs, code):
    array = morph_array(code)
    array = array[7:11, :]
    value = (gbs * 365 * array[0][3] * array[0][2] - array[0][1] - array[0][0]) / (1.15 ** 7)
    if value > 0:
        return 7
    elif value < 0:
        return 12
    else:
        return 11


def bld_npv28(gbs, code):
    array = morph_array(code)
    array = array[8:11, :]
    value = (gbs * 365 * array[0][3] * array[0][2] - array[0][1] - array[0][0]) / (1.15 ** 8)
    if value > 0:
        return 8
    elif value < 0:
        return 12
    else:
        return 11


def bld_npv29(gbs, code):
    array = morph_array(code)
    array = array[9:11, :]
    value = (gbs * 365 * array[0][3] * array[0][2] - array[0][1] - array[0][0]) / (1.15 ** 9)
    if value > 0:
        return 9
    elif value < 0:
        return 12
    else:
        return 11


def bld_npv30(gbs, code):
    array = morph_array(code)
    array = array[10:11, :]
    value = (gbs * 365 * array[0][3] * array[0][2] - array[0][1] - array[0][0]) / (1.15 ** 10)
    if value > 0:
        return 11
    elif value < 0:
        return 12
    else:
        return 11


# npv functions used in initial npv calculation before loop,
# used numpy array to make year


def npv21(gbs, code):
    value = np.empty([10, 1])
    array = morph_array(code)
    array = array[1:11, :]
    for m in range(10):
        value[m] = gbs * 365 * array[m][3] * array[m][2] - array[m][1]
    year = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    year_0 = 0
    value[0] = value[0] - array[0][0]
    return int(sum([value_i / ((1.0 + disc_rt) ** (year_i - year_0))
                    for value_i, year_i in zip(value, year)]))


def npv22(gbs, code):
    value = np.empty([9, 1])
    array = morph_array(code)
    array = array[2:11, :]
    for m in range(9):
        value[m] = gbs * 365 * array[m][3] * array[m][2] - array[m][1]
    year = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
    year_0 = 0
    value[0] = value[0] - array[0][0]
    return int(sum([value_i / ((1.0 + disc_rt) ** (year_i - year_0))
                    for value_i, year_i in zip(value, year)]))


def npv23(gbs, code):
    value = np.empty([8, 1])
    array = morph_array(code)
    array = array[3:11, :]
    for m in range(8):
        value[m] = gbs * 365 * array[m][3] * array[m][2] - array[m][1]
    year = np.array([3, 4, 5, 6, 7, 8, 9, 10])
    year_0 = 0
    value[0] = value[0] - array[0][0]
    return int(sum([value_i / ((1.0 + disc_rt) ** (year_i - year_0))
                    for value_i, year_i in zip(value, year)]))


def npv24(gbs, code):
    value = np.empty([7, 1])
    array = morph_array(code)
    array = array[4:11, :]
    for m in range(7):
        value[m] = gbs * 365 * array[m][3] * array[m][2] - array[m][1]
    year = np.array([4, 5, 6, 7, 8, 9, 10])
    year_0 = 0
    value[0] = value[0] - array[0][0]
    return int(sum([value_i / ((1.0 + disc_rt) ** (year_i - year_0))
                    for value_i, year_i in zip(value, year)]))


def npv25(gbs, code):
    value = np.empty([6, 1])
    array = morph_array(code)
    array = array[5:11, :]
    for m in range(6):
        value[m] = gbs * 365 * array[m][3] * array[m][2] - array[m][1]
    year = np.array([5, 6, 7, 8, 9, 10])
    year_0 = 0
    value[0] = value[0] - array[0][0]
    return int(sum([value_i / ((1.0 + disc_rt) ** (year_i - year_0))
                    for value_i, year_i in zip(value, year)]))


def npv26(gbs, code):
    value = np.empty([5, 1])
    array = morph_array(code)
    array = array[6:11, :]
    for m in range(5):
        value[m] = gbs * 365 * array[m][3] * array[m][2] - array[m][1]
    year = np.array([6, 7, 8, 9, 10])
    year_0 = 0
    value[0] = value[0] - array[0][0]
    return int(sum([value_i / ((1.0 + disc_rt) ** (year_i - year_0))
                    for value_i, year_i in zip(value, year)]))


def npv27(gbs, code):
    value = np.empty([4, 1])
    array = morph_array(code)
    array = array[7:11, :]
    for m in range(4):
        value[m] = gbs * 365 * array[m][3] * array[m][2] - array[m][1]
    year = np.array([7, 8, 9, 10])
    year_0 = 0
    value[0] = value[0] - array[0][0]
    return int(sum([value_i / ((1.0 + disc_rt) ** (year_i - year_0))
                    for value_i, year_i in zip(value, year)]))


def npv28(gbs, code):
    value = np.empty([3, 1])
    array = morph_array(code)
    array = array[8:11, :]
    for m in range(3):
        value[m] = gbs * 365 * array[m][3] * array[m][2] - array[m][1]
    year = np.array([8, 9, 10])
    year_0 = 0
    value[0] = value[0] - array[0][0]
    return int(sum([value_i / ((1.0 + disc_rt) ** (year_i - year_0))
                    for value_i, year_i in zip(value, year)]))


def npv29(gbs, code):
    value = np.empty([2, 1])
    array = morph_array(code)
    array = array[9:11, :]
    for m in range(2):
        value[m] = gbs * 365 * array[m][3] * array[m][2] - array[m][1]
    year = np.array([9, 10])
    year_0 = 0
    value[0] = value[0] - array[0][0]
    return int(sum([value_i / ((1.0 + disc_rt) ** (year_i - year_0))
                    for value_i, year_i in zip(value, year)]))


def npv30(gbs, code):
    value = np.empty([1, 1])
    array = morph_array(code)
    array = array[10:11, :]
    for m in range(1):
        value[m] = gbs * 365 * array[m][3] * array[m][2] - array[m][1]
    year = np.array([10])
    year_0 = 0
    value[0] = value[0] - array[0][0]
    return int(sum([value_i / ((1.0 + disc_rt) ** (year_i - year_0))
                    for value_i, year_i in zip(value, year)]))


def loop_npv21(gbs, code):
    value = np.empty([10, 1])
    year = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    array = morph_array(code)
    for m in range(10):
        value[m] = gbs * 365 * array[m][3] * array[m][2] - array[m][1]
    year_0 = 0
    value[0] = value[0] - array[0][0]
    return int(sum([value_i / ((1.0 + disc_rt) ** (year_i - year_0))
                    for value_i, year_i in zip(value, year)]))


def loop_npv22(gbs, code):
    value = np.empty([9, 1])
    year = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
    array = morph_array(code)
    for m in range(9):
        value[m] = gbs * 365 * array[m][3] * array[m][2] - array[m][1]
    year_0 = 0
    value[0] = value[0] - array[0][0]
    return int(sum([value_i / ((1.0 + disc_rt) ** (year_i - year_0))
                    for value_i, year_i in zip(value, year)]))


def loop_npv23(gbs, code):
    value = np.empty([8, 1])
    year = np.array([3, 4, 5, 6, 7, 8, 9, 10])
    array = morph_array(code)
    for m in range(8):
        value[m] = gbs * 365 * array[m][3] * array[m][2] - array[m][1]
    year_0 = 0
    value[0] = value[0] - array[0][0]
    return int(sum([value_i / ((1.0 + disc_rt) ** (year_i - year_0))
                    for value_i, year_i in zip(value, year)]))


def loop_npv24(gbs, code):
    value = np.empty([7, 1])
    year = np.array([4, 5, 6, 7, 8, 9, 10])
    array = morph_array(code)
    for m in range(7):
        value[m] = gbs * 365 * array[m][3] * array[m][2] - array[m][1]
    year_0 = 0
    value[0] = value[0] - array[0][0]
    return int(sum([value_i / ((1.0 + disc_rt) ** (year_i - year_0))
                    for value_i, year_i in zip(value, year)]))


def loop_npv25(gbs, code):
    value = np.empty([6, 1])
    year = np.array([5, 6, 7, 8, 9, 10])
    array = morph_array(code)
    for m in range(6):
        value[m] = gbs * 365 * array[m][3] * array[m][2] - array[m][1]
    year_0 = 0
    value[0] = value[0] - array[0][0]
    npv_loop = int(sum([value_i / ((1.0 + disc_rt) ** (year_i - year_0))
                        for value_i, year_i in zip(value, year)]))
    return npv_loop


def loop_npv26(gbs, code):
    value = np.empty([5, 1])
    year = np.array([6, 7, 8, 9, 10])
    array = morph_array(code)
    for m in range(5):
        value[m] = gbs * 365 * array[m][3] * array[m][2] - array[m][1]
    year_0 = 0
    value[0] = value[0] - array[0][0]
    npv_loop = int(sum([value_i / ((1.0 + disc_rt) ** (year_i - year_0))
                        for value_i, year_i in zip(value, year)]))
    return npv_loop


def loop_npv27(gbs, code):
    value = np.empty([4, 1])
    year = np.array([7, 8, 9, 10])
    array = morph_array(code)
    for m in range(4):
        value[m] = gbs * 365 * array[m][3] * array[m][2] - array[m][1]
    year_0 = 0
    value[0] = value[0] - array[0][0]
    return int(sum([value_i / ((1.0 + disc_rt) ** (year_i - year_0))
                    for value_i, year_i in zip(value, year)]))


def loop_npv28(gbs, code):
    value = np.empty([3, 1])
    year = np.array([8, 9, 10])
    array = morph_array(code)
    for m in range(3):
        value[m] = gbs * 365 * array[m][3] * array[m][2] - array[m][1]
    year_0 = 0
    value[0] = value[0] - array[0][0]
    return int(sum([value_i / ((1.0 + disc_rt) ** (year_i - year_0))
                    for value_i, year_i in zip(value, year)]))


def loop_npv29(gbs, code):
    value = np.empty([2, 1])
    year = np.array([9, 10])
    array = morph_array(code)
    for m in range(2):
        value[m] = gbs * 365 * array[m][3] * array[m][2] - array[m][1]
    year_0 = 0
    value[0] = value[0] - array[0][0]
    return int(sum([value_i / ((1.0 + disc_rt) ** (year_i - year_0))
                    for value_i, year_i in zip(value, year)]))


def loop_npv30(gbs, code):
    value = np.empty([1, 1])
    year = np.array([10])
    array = morph_array(code)
    for m in range(1):
        value[m] = gbs * 365 * array[m][3] * array[m][2] - array[m][1]
    year_0 = 0
    value[0] = value[0] - array[0][0]
    return int(sum([value_i / ((1.0 + disc_rt) ** (year_i - year_0))
                    for value_i, year_i in zip(value, year)]))


def rb_thru_put(code_rate, symbols, mimo, subframe, retrans, high_layer_over, over_tp_kbps):
    rb_thru = (((code_rate * symbols * 4800000 * mimo * subframe) / 1000000) * ((1 - retrans) * (1 - high_layer_over)) - (over_tp_kbps / 1000)) / 400
    return rb_thru


def rx_calc(path_loss_umi_db):
    rx_signal_strength_db = 37.0 - path_loss_umi_db
    sinr = (rx_signal_strength_db - rx_sensitivity_db).round(0).astype(int)
    rx_signal_strength_mw = (10 ** ((37.0 - path_loss_umi_db) / 10) / 1000.000000000)
    return rx_signal_strength_db, sinr, rx_signal_strength_mw


def sinr_new(sum_rx_signal, rx_signal_strength_db):
    # note that -89.4 here is rx_sensitivity_db
    sinr_nouveau = np.round((sum_rx_signal.where(sum_rx_signal.isnull(),
                                                 rx_signal_strength_db -
                                                 (np.log10((sum_rx_signal + (10 ** (-89.4 / 10))
                                                            / 1000) * 1000) * 10))), 0)
    return sinr_nouveau


# SECTION 3: FIRST ITERATION TO GENERATE INHERENT NPV

pole_du = fin_arrays(1)
pole_u = fin_arrays(2)
pole_s = fin_arrays(3)
pole_r = fin_arrays(4)
off_du = fin_arrays(5)
off_u = fin_arrays(6)
off_s = fin_arrays(7)
off_r = fin_arrays(8)
roe_du = fin_arrays(9)
roe_u = fin_arrays(10)
roe_s = fin_arrays(11)
roe_r = fin_arrays(12)
smb_du = fin_arrays(13)
smb_u = fin_arrays(14)
smb_s = fin_arrays(15)
smb_r = fin_arrays(16)

# SECTION 3: FIRST ITERATION TO GENERATE INHERENT NPV

sites['morph_code'] = sites.apply(lambda x: morph(x['Type'], x['Morphology']), axis=1)
sites = sites.drop(['Type', 'Morphology'], axis=1)
site_t_m = sites.groupby('fict_site')['morph_code'].mean().reset_index()
init_sites = sites

# calculation of build years from inherent npv
site_bin_gbs = sites.groupby('fict_site')['Hour_GBs'].sum().reset_index()
bld_yr_sites = pd.merge(site_bin_gbs, site_t_m, on='fict_site')

bld_yr_sites['1'] = bld_yr_sites.apply(lambda x: bld_npv21(x['Hour_GBs'], x['morph_code']), axis=1)
bld_yr_sites['2'] = bld_yr_sites.apply(lambda x: bld_npv22(x['Hour_GBs'], x['morph_code']), axis=1)
bld_yr_sites['3'] = bld_yr_sites.apply(lambda x: bld_npv23(x['Hour_GBs'], x['morph_code']), axis=1)
bld_yr_sites['4'] = bld_yr_sites.apply(lambda x: bld_npv24(x['Hour_GBs'], x['morph_code']), axis=1)
bld_yr_sites['5'] = bld_yr_sites.apply(lambda x: bld_npv25(x['Hour_GBs'], x['morph_code']), axis=1)
bld_yr_sites['6'] = bld_yr_sites.apply(lambda x: bld_npv26(x['Hour_GBs'], x['morph_code']), axis=1)
bld_yr_sites['7'] = bld_yr_sites.apply(lambda x: bld_npv27(x['Hour_GBs'], x['morph_code']), axis=1)
bld_yr_sites['8'] = bld_yr_sites.apply(lambda x: bld_npv28(x['Hour_GBs'], x['morph_code']), axis=1)
bld_yr_sites['9'] = bld_yr_sites.apply(lambda x: bld_npv29(x['Hour_GBs'], x['morph_code']), axis=1)
bld_yr_sites['10'] = bld_yr_sites.apply(lambda x: bld_npv30(x['Hour_GBs'], x['morph_code']), axis=1)

inh_bld_yr = bld_yr_sites[['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']]

bld_yr_sites['build_yr'] = inh_bld_yr.idxmin(axis=1).astype(int)
bld_yr_sites = bld_yr_sites[bld_yr_sites['10'] != 12]
print(bld_yr_sites['build_yr'].value_counts())
init_build_yr_sites = bld_yr_sites[['fict_site', 'build_yr']]

# min_bld_yr_sites = pd.merge(init_sites, init_build_yr_sites, on='fict_site')
# min_bld_yr_sites.to_csv('wtg_min_bld_yr.csv')

sites_yr1 = bld_yr_sites.loc[bld_yr_sites['build_yr'] == 1]
sites_yr2 = bld_yr_sites.loc[bld_yr_sites['build_yr'] == 2]
sites_yr3 = bld_yr_sites.loc[bld_yr_sites['build_yr'] == 3]
sites_yr4 = bld_yr_sites.loc[bld_yr_sites['build_yr'] == 4]
sites_yr5 = bld_yr_sites.loc[bld_yr_sites['build_yr'] == 5]
sites_yr6 = bld_yr_sites.loc[bld_yr_sites['build_yr'] == 6]
sites_yr7 = bld_yr_sites.loc[bld_yr_sites['build_yr'] == 7]
sites_yr8 = bld_yr_sites.loc[bld_yr_sites['build_yr'] == 8]
sites_yr9 = bld_yr_sites.loc[bld_yr_sites['build_yr'] == 9]
sites_yr10 = bld_yr_sites.loc[bld_yr_sites['build_yr'] == 10]

site_number = len(sites_yr1) + len(sites_yr2) + len(sites_yr3) + len(sites_yr4) + len(sites_yr5) + len(sites_yr6) + \
              len(sites_yr7) + len(sites_yr8) + len(sites_yr9) + len(sites_yr10)
print(site_number)
selected = pd.DataFrame([])

print('Year 1', end=' ')

pole_du = pole_du[1:11, :]
pole_u = pole_u[1:11, :]
pole_s = pole_s[1:11, :]
pole_r = pole_r[1:11, :]
off_du = off_du[1:11, :]
off_u = off_u[1:11, :]
off_s = off_s[1:11, :]
off_r = off_r[1:11, :]
roe_du = roe_du[1:11, :]
roe_u = roe_u[1:11, :]
roe_s = roe_s[1:11, :]
roe_r = roe_r[1:11, :]
smb_du = smb_du[1:11, :]
smb_u = smb_u[1:11, :]
smb_s = smb_s[1:11, :]
smb_r = smb_r[1:11, :]


init_sites['rx_signal_strength_db'], init_sites['sinr'], init_sites['rx_signal_strength_mw'] = rx_calc(init_sites['path_loss_umi_db'])
init_sites.loc[init_sites['sinr'] > 50, 'sinr'] = 50.0
calc_sites = init_sites[['fict_site', 'GridName', 'Hour_GBs', 'sinr', 'rx_signal_strength_db',
                         'rx_signal_strength_mw', 'morph_code']]

sites_yr1 = sites_yr1[['fict_site']]
sites_1 = pd.merge(calc_sites, sites_yr1, on='fict_site')

start_yr1 = timeit.default_timer()
candidates = pd.DataFrame([])
init_ranking = pd.DataFrame([])
bad_candidate = 0.0
full = 1

if len(sites_yr1) > 0:
    sites = sites_1
    sites = sites.groupby('fict_site', as_index=False).agg({'morph_code': 'mean', 'Hour_GBs': 'sum'})
    sites['npv'] = sites.apply(lambda x: loop_npv21(x['Hour_GBs'], x['morph_code']), axis=1).astype(int)
    ranking = sites.loc[sites['npv'] > 0].copy()
    ranking['rank'] = ranking['npv'].rank(ascending=False)
    ranking = ranking.sort_values(by='rank')
    temp_ranking = ranking[['fict_site']]
    init_ranking_bins = pd.merge(temp_ranking, sites_1, on='fict_site')
    init_ranking_counts = init_ranking_bins['GridName'].value_counts().reset_index()
    init_ranking_counts = init_ranking_counts.rename(columns={'index': 'GridName', 'GridName': 'bin_count'})
    init_ranking = pd.merge(init_ranking_bins, init_ranking_counts, on='GridName')
    if len(selected) > 0 and len(ranking) > 0:
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[['fict_site', 'npv']]
        selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected.append(selected, sort=True)
        selected = selected[['fict_site', 'npv']]
    elif len(selected) == 0 and len(ranking) > 0:
        selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected[['fict_site', 'npv']]
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[['fict_site', 'npv']]
    else:
        candidates = pd.DataFrame([])

for j in range(len(candidates)):
    # start_outer = timeit.default_timer()
    temp_nb = pd.DataFrame([])
    for i in range(len(candidates)):
        next_best = (pd.DataFrame(candidates.iloc[i, :])).T
        sinr_sites = selected.append(next_best, sort=True)
        sinr_bins = pd.merge(init_ranking, sinr_sites, on='fict_site')

        sinr_bins_unique = sinr_bins[sinr_bins['bin_count'] == 1].copy()
        sinr_bins_dups = sinr_bins[sinr_bins['bin_count'] > 1].copy()
        temp_sinr_bins_dups = sinr_bins_dups.groupby('GridName')['rx_signal_strength_mw'].sum().reset_index()
        sinr_bins_dups = sinr_bins_dups.drop(['rx_signal_strength_mw'], axis=1)
        sinr_bins_dups = pd.merge(sinr_bins_dups, temp_sinr_bins_dups, on='GridName')
        sinr_bins_unique['sinr_new'] = sinr_bins_unique['sinr']
        sinr_bins_dups['sinr_new'] = sinr_new(sinr_bins_dups['rx_signal_strength_mw'],
                                              sinr_bins_dups['rx_signal_strength_db'])
        sinr_bins = sinr_bins_unique.append(sinr_bins_dups, sort=True)
        temp_sinr_bins = sinr_bins
        sinr_bins = sinr_bins[sinr_bins.sinr_new >= -7.0]

        sinr_bins_agg = sinr_bins.groupby('fict_site', as_index=False).agg({'morph_code': 'mean', 'Hour_GBs': 'sum'})
        sinr_bins_agg['xnpv'] = sinr_bins_agg.apply(lambda x: loop_npv21(x['Hour_GBs'], x['morph_code']),
                                                    axis=1).astype(int)
        next_best['adj_npv'] = sinr_bins_agg['xnpv'].min()
        next_best['sum_npv'] = sinr_bins_agg['xnpv'].sum()
        next_best = next_best[['fict_site', 'adj_npv', 'sum_npv']]
        if next_best['adj_npv'].values <= npv_threshold:
            bad_candidate = next_best['fict_site'].values
            full = 0
            break
        init_cand_bins = pd.merge(next_best, init_ranking, on='fict_site')
        init_cand_bins = init_cand_bins.nlargest(bin_prox, 'sinr')
        init_cand_bins = init_cand_bins[['sinr', 'fict_site']].mean().astype('int64')
        cand_bins = pd.merge(next_best, temp_sinr_bins, on='fict_site')
        cand_bins = cand_bins.nlargest(bin_prox, 'sinr_new')
        cand_bins = cand_bins[['sinr_new']].mean()
        if (abs(init_cand_bins['sinr'] - cand_bins['sinr_new'])) > sinr_deg:
            bad_candidate = init_cand_bins['fict_site'].astype('int64')
            full = 0
            break
        temp_nb = temp_nb.append(next_best, sort=True)
        full = 1
    if full == 1:
        if len(candidates) == 0:
            break
        new_ranking = temp_nb
        new_ranking['rank'] = new_ranking['sum_npv'].rank(ascending=True)
        new_ranking = new_ranking.sort_values(by='rank').reset_index(drop='True')
        temp_selected = new_ranking[new_ranking['rank'] == 1]
        temp_selected = temp_selected[['fict_site']]
        selected = selected.append(temp_selected, sort=True)
        new_ranking = new_ranking[['fict_site']]
        candidates = pd.DataFrame(new_ranking.iloc[1:, :]).reset_index(drop=True)
        # print(len(candidates), end='  ')
        # print(timeit.default_timer() - start_outer)
    elif full == 0:
        if len(candidates) == 0:
            break
        candidates = candidates.to_numpy()
        candidates = np.delete(candidates, np.where(candidates == bad_candidate)[0], axis=0)
        candidates = pd.DataFrame(candidates).reset_index(drop=True)
        candidates = candidates.rename(columns={0: 'fict_site'})
        # print(len(candidates), end='  ')
        # print(timeit.default_timer() - start_outer)

end_yr1 = timeit.default_timer()
print(end_yr1 - start_yr1)

print('Year 2', end=' ')
# print(len(selected['fict_site'].unique()))

pole_du = pole_du[1:11, :]
pole_u = pole_u[1:11, :]
pole_s = pole_s[1:11, :]
pole_r = pole_r[1:11, :]
off_du = off_du[1:11, :]
off_u = off_u[1:11, :]
off_s = off_s[1:11, :]
off_r = off_r[1:11, :]
roe_du = roe_du[1:11, :]
roe_u = roe_u[1:11, :]
roe_s = roe_s[1:11, :]
roe_r = roe_r[1:11, :]
smb_du = smb_du[1:11, :]
smb_u = smb_u[1:11, :]
smb_s = smb_s[1:11, :]
smb_r = smb_r[1:11, :]

start_yr2 = timeit.default_timer()
sites_yr2 = sites_yr2[['fict_site']]
sites_2 = pd.merge(calc_sites, sites_yr2, on='fict_site')

if len(sites_yr2) > 0:
    sites = sites_2
    sites = sites.groupby('fict_site', as_index=False).agg({'morph_code': 'mean', 'Hour_GBs': 'sum'})
    sites['npv'] = sites.apply(lambda x: loop_npv22(x['Hour_GBs'], x['morph_code']), axis=1).astype(int)
    ranking = sites.loc[sites['npv'] > 0].copy()
    ranking['rank'] = ranking['npv'].rank(ascending=False)
    ranking = ranking.sort_values(by='rank')
    temp_ranking = ranking[['fict_site']]
    init_ranking_bins = pd.merge(temp_ranking, sites_2, on='fict_site')
    init_ranking_counts = init_ranking_bins['GridName'].value_counts().reset_index()
    init_ranking_counts = init_ranking_counts.rename(columns={'index': 'GridName', 'GridName': 'bin_count'})
    init_ranking = pd.merge(init_ranking_bins, init_ranking_counts, on='GridName')
    if len(selected) > 0 and len(ranking) > 0:
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[['fict_site', 'npv']]
        selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected.append(selected, sort=True)
        selected = selected[['fict_site', 'npv']]
    elif len(selected) == 0 and len(ranking) > 0:
        selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected[['fict_site', 'npv']]
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[['fict_site', 'npv']]
    else:
        candidates = pd.DataFrame([])

for j in range(len(candidates)):
    # start_outer = timeit.default_timer()
    temp_nb = pd.DataFrame([])
    for i in range(len(candidates)):
        next_best = (pd.DataFrame(candidates.iloc[i, :])).T
        sinr_sites = selected.append(next_best, sort=True)
        sinr_bins = pd.merge(init_ranking, sinr_sites, on='fict_site')

        sinr_bins_unique = sinr_bins[sinr_bins['bin_count'] == 1].copy()
        sinr_bins_dups = sinr_bins[sinr_bins['bin_count'] > 1].copy()
        temp_sinr_bins_dups = sinr_bins_dups.groupby('GridName')['rx_signal_strength_mw'].sum().reset_index()
        sinr_bins_dups = sinr_bins_dups.drop(['rx_signal_strength_mw'], axis=1)
        sinr_bins_dups = pd.merge(sinr_bins_dups, temp_sinr_bins_dups, on='GridName')
        sinr_bins_unique['sinr_new'] = sinr_bins_unique['sinr']
        sinr_bins_dups['sinr_new'] = sinr_new(sinr_bins_dups['rx_signal_strength_mw'],
                                              sinr_bins_dups['rx_signal_strength_db'])
        sinr_bins = sinr_bins_unique.append(sinr_bins_dups, sort=True)
        temp_sinr_bins = sinr_bins
        sinr_bins = sinr_bins[sinr_bins.sinr_new >= -7.0]

        sinr_bins_agg = sinr_bins.groupby('fict_site', as_index=False).agg({'morph_code': 'mean', 'Hour_GBs': 'sum'})
        sinr_bins_agg['xnpv'] = sinr_bins_agg.apply(lambda x: loop_npv22(x['Hour_GBs'], x['morph_code']),
                                                    axis=1).astype(int)
        next_best['adj_npv'] = sinr_bins_agg['xnpv'].min()
        next_best['sum_npv'] = sinr_bins_agg['xnpv'].sum()
        next_best = next_best[['fict_site', 'adj_npv', 'sum_npv']]
        if next_best['adj_npv'].values <= npv_threshold:
            bad_candidate = next_best['fict_site'].values
            full = 0
            break
        init_cand_bins = pd.merge(next_best, init_ranking, on='fict_site')
        init_cand_bins = init_cand_bins.nlargest(bin_prox, 'sinr')
        init_cand_bins = init_cand_bins[['sinr', 'fict_site']].mean().astype('int64')
        cand_bins = pd.merge(next_best, temp_sinr_bins, on='fict_site')
        cand_bins = cand_bins.nlargest(bin_prox, 'sinr_new')
        cand_bins = cand_bins[['sinr_new']].mean()
        if (abs(init_cand_bins['sinr'] - cand_bins['sinr_new'])) > sinr_deg:
            bad_candidate = init_cand_bins['fict_site'].astype('int64')
            full = 0
            break
        temp_nb = temp_nb.append(next_best, sort=True)
        full = 1
    if full == 1:
        if len(candidates) == 0:
            break
        new_ranking = temp_nb
        new_ranking['rank'] = new_ranking['sum_npv'].rank(ascending=True)
        new_ranking = new_ranking.sort_values(by='rank').reset_index(drop='True')
        temp_selected = new_ranking[new_ranking['rank'] == 1]
        temp_selected = temp_selected[['fict_site']]
        selected = selected.append(temp_selected, sort=True)
        new_ranking = new_ranking[['fict_site']]
        candidates = pd.DataFrame(new_ranking.iloc[1:, :]).reset_index(drop=True)
        # print(len(candidates), end='  ')
        # print(timeit.default_timer() - start_outer)
    elif full == 0:
        if len(candidates) == 0:
            break
        candidates = candidates.to_numpy()
        candidates = np.delete(candidates, np.where(candidates == bad_candidate)[0], axis=0)
        candidates = pd.DataFrame(candidates).reset_index(drop=True)
        candidates = candidates.rename(columns={0: 'fict_site'})
        # print(len(candidates), end='  ')
        # print(timeit.default_timer() - start_outer)
    elif len(candidates) == 0:
        break

selected = selected.reset_index(drop=True)

end_yr2 = timeit.default_timer()
print(end_yr2 - start_yr2)

print('Year 3', end=' ')
# print(len(selected['fict_site'].unique()))

pole_du = pole_du[1:11, :]
pole_u = pole_u[1:11, :]
pole_s = pole_s[1:11, :]
pole_r = pole_r[1:11, :]
off_du = off_du[1:11, :]
off_u = off_u[1:11, :]
off_s = off_s[1:11, :]
off_r = off_r[1:11, :]
roe_du = roe_du[1:11, :]
roe_u = roe_u[1:11, :]
roe_s = roe_s[1:11, :]
roe_r = roe_r[1:11, :]
smb_du = smb_du[1:11, :]
smb_u = smb_u[1:11, :]
smb_s = smb_s[1:11, :]
smb_r = smb_r[1:11, :]

start_yr3 = timeit.default_timer()
sites_yr3 = sites_yr3[['fict_site']]
sites_3 = pd.merge(calc_sites, sites_yr3, on='fict_site')

if len(sites_yr3) > 0:
    sites = sites_3
    sites = sites.groupby('fict_site', as_index=False).agg({'morph_code': 'mean', 'Hour_GBs': 'sum'})
    sites['npv'] = sites.apply(lambda x: loop_npv23(x['Hour_GBs'], x['morph_code']), axis=1).astype(int)
    ranking = sites.loc[sites['npv'] > 0].copy()
    ranking['rank'] = ranking['npv'].rank(ascending=False)
    ranking = ranking.sort_values(by='rank')
    temp_ranking = ranking[['fict_site']]
    init_ranking_bins = pd.merge(temp_ranking, sites_3, on='fict_site')
    init_ranking_counts = init_ranking_bins['GridName'].value_counts().reset_index()
    init_ranking_counts = init_ranking_counts.rename(columns={'index': 'GridName', 'GridName': 'bin_count'})
    init_ranking = pd.merge(init_ranking_bins, init_ranking_counts, on='GridName')
    if len(selected) > 0 and len(ranking) > 0:
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[['fict_site', 'npv']]
        new_selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected.append(new_selected, sort=True)
        selected = selected[['fict_site', 'npv']]
    elif len(selected) == 0 and len(ranking) > 0:
        selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected[['fict_site', 'npv']]
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[['fict_site', 'npv']]
    else:
        candidates = pd.DataFrame([])

for j in range(len(candidates)):
    # start_outer = timeit.default_timer()
    temp_nb = pd.DataFrame([])
    for i in range(len(candidates)):
        next_best = (pd.DataFrame(candidates.iloc[i, :])).T
        sinr_sites = selected.append(next_best, sort=True)
        sinr_bins = pd.merge(init_ranking, sinr_sites, on='fict_site')

        sinr_bins_unique = sinr_bins[sinr_bins['bin_count'] == 1].copy()
        sinr_bins_dups = sinr_bins[sinr_bins['bin_count'] > 1].copy()
        temp_sinr_bins_dups = sinr_bins_dups.groupby('GridName')['rx_signal_strength_mw'].sum().reset_index()
        sinr_bins_dups = sinr_bins_dups.drop(['rx_signal_strength_mw'], axis=1)
        sinr_bins_dups = pd.merge(sinr_bins_dups, temp_sinr_bins_dups, on='GridName')
        sinr_bins_unique['sinr_new'] = sinr_bins_unique['sinr']
        sinr_bins_dups['sinr_new'] = sinr_new(sinr_bins_dups['rx_signal_strength_mw'],
                                              sinr_bins_dups['rx_signal_strength_db'])
        sinr_bins = sinr_bins_unique.append(sinr_bins_dups, sort=True)
        temp_sinr_bins = sinr_bins
        sinr_bins = sinr_bins[sinr_bins.sinr_new >= -7.0]

        sinr_bins_agg = sinr_bins.groupby('fict_site', as_index=False).agg({'morph_code': 'mean', 'Hour_GBs': 'sum'})
        sinr_bins_agg['xnpv'] = sinr_bins_agg.apply(lambda x: loop_npv23(x['Hour_GBs'], x['morph_code']),
                                                    axis=1).astype(int)
        next_best['adj_npv'] = sinr_bins_agg['xnpv'].min()
        next_best['sum_npv'] = sinr_bins_agg['xnpv'].sum()
        next_best = next_best[['fict_site', 'adj_npv', 'sum_npv']]
        if next_best['adj_npv'].values <= npv_threshold:
            bad_candidate = next_best['fict_site'].values
            full = 0
            break
        init_cand_bins = pd.merge(next_best, init_ranking, on='fict_site')
        init_cand_bins = init_cand_bins.nlargest(bin_prox, 'sinr')
        init_cand_bins = init_cand_bins[['sinr', 'fict_site']].mean().astype('int64')
        cand_bins = pd.merge(next_best, temp_sinr_bins, on='fict_site')
        cand_bins = cand_bins.nlargest(bin_prox, 'sinr_new')
        cand_bins = cand_bins[['sinr_new']].mean()
        if (abs(init_cand_bins['sinr'] - cand_bins['sinr_new'])) > sinr_deg:
            bad_candidate = init_cand_bins['fict_site'].astype('int64')
            full = 0
            break
        temp_nb = temp_nb.append(next_best, sort=True)
        full = 1
    if full == 1:
        if len(candidates) == 0:
            break
        new_ranking = temp_nb
        new_ranking['rank'] = new_ranking['sum_npv'].rank(ascending=True)
        new_ranking = new_ranking.sort_values(by='rank').reset_index(drop='True')
        temp_selected = new_ranking[new_ranking['rank'] == 1]
        temp_selected = temp_selected[['fict_site']]
        selected = selected.append(temp_selected, sort=True)
        new_ranking = new_ranking[['fict_site']]
        candidates = pd.DataFrame(new_ranking.iloc[1:, :]).reset_index(drop=True)
        # print(len(candidates), end='  ')
        # print(timeit.default_timer() - start_outer)
    elif full == 0:
        if len(candidates) == 0:
            break
        candidates = candidates.to_numpy()
        candidates = np.delete(candidates, np.where(candidates == bad_candidate)[0], axis=0)
        candidates = pd.DataFrame(candidates).reset_index(drop=True)
        candidates = candidates.rename(columns={0: 'fict_site'})
        # print(len(candidates), end='  ')
        # print(timeit.default_timer() - start_outer)
    elif len(candidates) == 0:
        break

selected = selected.reset_index(drop=True)

end_yr3 = timeit.default_timer()
print(end_yr3 - start_yr3)

print('Year 4', end=' ')
# print(len(selected['fict_site'].unique()))

pole_du = pole_du[1:11, :]
pole_u = pole_u[1:11, :]
pole_s = pole_s[1:11, :]
pole_r = pole_r[1:11, :]
off_du = off_du[1:11, :]
off_u = off_u[1:11, :]
off_s = off_s[1:11, :]
off_r = off_r[1:11, :]
roe_du = roe_du[1:11, :]
roe_u = roe_u[1:11, :]
roe_s = roe_s[1:11, :]
roe_r = roe_r[1:11, :]
smb_du = smb_du[1:11, :]
smb_u = smb_u[1:11, :]
smb_s = smb_s[1:11, :]
smb_r = smb_r[1:11, :]

start_yr4 = timeit.default_timer()
sites_yr4 = sites_yr4[['fict_site']]
sites_4 = pd.merge(calc_sites, sites_yr4, on='fict_site')

if len(sites_yr4) > 0:
    sites = sites_4
    sites = sites.groupby('fict_site', as_index=False).agg({'morph_code': 'mean', 'Hour_GBs': 'sum'})
    sites['npv'] = sites.apply(lambda x: loop_npv24(x['Hour_GBs'], x['morph_code']), axis=1).astype(int)
    ranking = sites.loc[sites['npv'] > 0].copy()
    ranking['rank'] = ranking['npv'].rank(ascending=False)
    ranking = ranking.sort_values(by='rank')
    temp_ranking = ranking[['fict_site']]
    init_ranking_bins = pd.merge(temp_ranking, sites_4, on='fict_site')
    init_ranking_counts = init_ranking_bins['GridName'].value_counts().reset_index()
    init_ranking_counts = init_ranking_counts.rename(columns={'index': 'GridName', 'GridName': 'bin_count'})
    init_ranking = pd.merge(init_ranking_bins, init_ranking_counts, on='GridName')
    if len(selected) > 0 and len(ranking) > 0:
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[['fict_site', 'npv']]
        new_selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected.append(new_selected, sort=True)
        selected = selected[['fict_site', 'npv']]
    elif len(selected) == 0 and len(ranking) > 0:
        selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected[['fict_site', 'npv']]
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[['fict_site', 'npv']]
    else:
        candidates = pd.DataFrame([])

for j in range(len(candidates)):
    # start_outer = timeit.default_timer()
    temp_nb = pd.DataFrame([])
    for i in range(len(candidates)):
        next_best = (pd.DataFrame(candidates.iloc[i, :])).T
        sinr_sites = selected.append(next_best, sort=True)
        sinr_bins = pd.merge(init_ranking, sinr_sites, on='fict_site')

        sinr_bins_unique = sinr_bins[sinr_bins['bin_count'] == 1].copy()
        sinr_bins_dups = sinr_bins[sinr_bins['bin_count'] > 1].copy()
        temp_sinr_bins_dups = sinr_bins_dups.groupby('GridName')['rx_signal_strength_mw'].sum().reset_index()
        sinr_bins_dups = sinr_bins_dups.drop(['rx_signal_strength_mw'], axis=1)
        sinr_bins_dups = pd.merge(sinr_bins_dups, temp_sinr_bins_dups, on='GridName')
        sinr_bins_unique['sinr_new'] = sinr_bins_unique['sinr']
        sinr_bins_dups['sinr_new'] = sinr_new(sinr_bins_dups['rx_signal_strength_mw'],
                                              sinr_bins_dups['rx_signal_strength_db'])
        sinr_bins = sinr_bins_unique.append(sinr_bins_dups, sort=True)
        temp_sinr_bins = sinr_bins
        sinr_bins = sinr_bins[sinr_bins.sinr_new >= -7.0]

        sinr_bins_agg = sinr_bins.groupby('fict_site', as_index=False).agg({'morph_code': 'mean', 'Hour_GBs': 'sum'})
        sinr_bins_agg['xnpv'] = sinr_bins_agg.apply(lambda x: loop_npv24(x['Hour_GBs'], x['morph_code']),
                                                    axis=1).astype(int)
        next_best['adj_npv'] = sinr_bins_agg['xnpv'].min()
        next_best['sum_npv'] = sinr_bins_agg['xnpv'].sum()
        next_best = next_best[['fict_site', 'adj_npv', 'sum_npv']]
        if next_best['adj_npv'].values <= npv_threshold:
            bad_candidate = next_best['fict_site'].values
            full = 0
            break
        init_cand_bins = pd.merge(next_best, init_ranking, on='fict_site')
        init_cand_bins = init_cand_bins.nlargest(bin_prox, 'sinr')
        init_cand_bins = init_cand_bins[['sinr', 'fict_site']].mean().astype('int64')
        cand_bins = pd.merge(next_best, temp_sinr_bins, on='fict_site')
        cand_bins = cand_bins.nlargest(bin_prox, 'sinr_new')
        cand_bins = cand_bins[['sinr_new']].mean()
        if (abs(init_cand_bins['sinr'] - cand_bins['sinr_new'])) > sinr_deg:
            bad_candidate = init_cand_bins['fict_site'].astype('int64')
            full = 0
            break
        temp_nb = temp_nb.append(next_best, sort=True)
        full = 1
    if full == 1:
        if len(candidates) == 0:
            break
        new_ranking = temp_nb
        new_ranking['rank'] = new_ranking['sum_npv'].rank(ascending=True)
        new_ranking = new_ranking.sort_values(by='rank').reset_index(drop='True')
        temp_selected = new_ranking[new_ranking['rank'] == 1]
        temp_selected = temp_selected[['fict_site']]
        selected = selected.append(temp_selected, sort=True)
        new_ranking = new_ranking[['fict_site']]
        candidates = pd.DataFrame(new_ranking.iloc[1:, :]).reset_index(drop=True)
        # print(len(candidates), end='  ')
        # print(timeit.default_timer() - start_outer)
    elif full == 0:
        if len(candidates) == 0:
            break
        candidates = candidates.to_numpy()
        candidates = np.delete(candidates, np.where(candidates == bad_candidate)[0], axis=0)
        candidates = pd.DataFrame(candidates).reset_index(drop=True)
        candidates = candidates.rename(columns={0: 'fict_site'})
        # print(len(candidates), end='  ')
        # print(timeit.default_timer() - start_outer)

selected = selected.reset_index(drop=True)


end_yr4 = timeit.default_timer()
print(end_yr4 - start_yr4)

print('Year 5', end=' ')
print(len(selected['fict_site'].unique()), end=' ')

pole_du = pole_du[1:11, :]
pole_u = pole_u[1:11, :]
pole_s = pole_s[1:11, :]
pole_r = pole_r[1:11, :]
off_du = off_du[1:11, :]
off_u = off_u[1:11, :]
off_s = off_s[1:11, :]
off_r = off_r[1:11, :]
roe_du = roe_du[1:11, :]
roe_u = roe_u[1:11, :]
roe_s = roe_s[1:11, :]
roe_r = roe_r[1:11, :]
smb_du = smb_du[1:11, :]
smb_u = smb_u[1:11, :]
smb_s = smb_s[1:11, :]
smb_r = smb_r[1:11, :]

start_yr5 = timeit.default_timer()
sites_yr5 = sites_yr5[['fict_site']]
sites_5 = pd.merge(calc_sites, sites_yr5, on='fict_site')

if len(sites_yr5) > 0:
    sites = sites_5
    sites = sites.groupby('fict_site', as_index=False).agg({'morph_code': 'mean', 'Hour_GBs': 'sum'})
    sites['npv'] = sites.apply(lambda x: loop_npv25(x['Hour_GBs'], x['morph_code']), axis=1).astype(int)
    ranking = sites.loc[sites['npv'] > 0].copy()
    ranking['rank'] = ranking['npv'].rank(ascending=False)
    ranking = ranking.sort_values(by='rank')
    temp_ranking = ranking[['fict_site']]
    init_ranking_bins = pd.merge(temp_ranking, sites_5, on='fict_site')
    init_ranking_counts = init_ranking_bins['GridName'].value_counts().reset_index()
    init_ranking_counts = init_ranking_counts.rename(columns={'index': 'GridName', 'GridName': 'bin_count'})
    init_ranking = pd.merge(init_ranking_bins, init_ranking_counts, on='GridName')
    if len(selected) > 0 and len(ranking) > 0:
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[['fict_site', 'npv']]
        new_selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected.append(new_selected, sort=True)
        selected = selected[['fict_site', 'npv']]
    elif len(selected) == 0 and len(ranking) > 0:
        selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected[['fict_site', 'npv']]
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[['fict_site', 'npv']]
    else:
        candidates = pd.DataFrame([])

for j in range(len(candidates)):
    # start_outer = timeit.default_timer()
    temp_nb = pd.DataFrame([])
    for i in range(len(candidates)):
        next_best = (pd.DataFrame(candidates.iloc[i, :])).T
        sinr_sites = selected.append(next_best, sort=True)
        sinr_bins = pd.merge(init_ranking, sinr_sites, on='fict_site')

        sinr_bins_unique = sinr_bins[sinr_bins['bin_count'] == 1].copy()
        sinr_bins_dups = sinr_bins[sinr_bins['bin_count'] > 1].copy()
        temp_sinr_bins_dups = sinr_bins_dups.groupby('GridName')['rx_signal_strength_mw'].sum().reset_index()
        sinr_bins_dups = sinr_bins_dups.drop(['rx_signal_strength_mw'], axis=1)
        sinr_bins_dups = pd.merge(sinr_bins_dups, temp_sinr_bins_dups, on='GridName')
        sinr_bins_unique['sinr_new'] = sinr_bins_unique['sinr']
        sinr_bins_dups['sinr_new'] = sinr_new(sinr_bins_dups['rx_signal_strength_mw'],
                                              sinr_bins_dups['rx_signal_strength_db'])
        sinr_bins = sinr_bins_unique.append(sinr_bins_dups, sort=True)
        temp_sinr_bins = sinr_bins
        sinr_bins = sinr_bins[sinr_bins.sinr_new >= -7.0]

        sinr_bins_agg = sinr_bins.groupby('fict_site', as_index=False).agg({'morph_code': 'mean', 'Hour_GBs': 'sum'})
        sinr_bins_agg['xnpv'] = sinr_bins_agg.apply(lambda x: loop_npv25(x['Hour_GBs'], x['morph_code']),
                                                    axis=1).astype(int)
        next_best['adj_npv'] = sinr_bins_agg['xnpv'].min()
        next_best['sum_npv'] = sinr_bins_agg['xnpv'].sum()
        next_best = next_best[['fict_site', 'adj_npv', 'sum_npv']]
        if next_best['adj_npv'].values <= npv_threshold:
            bad_candidate = next_best['fict_site'].values
            full = 0
            break
        init_cand_bins = pd.merge(next_best, init_ranking, on='fict_site')
        init_cand_bins = init_cand_bins.nlargest(bin_prox, 'sinr')
        init_cand_bins = init_cand_bins[['sinr', 'fict_site']].mean().astype('int64')
        cand_bins = pd.merge(next_best, temp_sinr_bins, on='fict_site')
        cand_bins = cand_bins.nlargest(bin_prox, 'sinr_new')
        cand_bins = cand_bins[['sinr_new']].mean()
        if (abs(init_cand_bins['sinr'] - cand_bins['sinr_new'])) > sinr_deg:
            bad_candidate = init_cand_bins['fict_site'].astype('int64')
            full = 0
            break
        temp_nb = temp_nb.append(next_best, sort=True)
        full = 1
    if full == 1:
        if len(candidates) == 0:
            break
        new_ranking = temp_nb
        new_ranking['rank'] = new_ranking['sum_npv'].rank(ascending=True)
        new_ranking = new_ranking.sort_values(by='rank').reset_index(drop='True')
        temp_selected = new_ranking[new_ranking['rank'] == 1]
        temp_selected = temp_selected[['fict_site']]
        selected = selected.append(temp_selected, sort=True)
        new_ranking = new_ranking[['fict_site']]
        candidates = pd.DataFrame(new_ranking.iloc[1:, :]).reset_index(drop=True)
        # print(len(candidates), end='  ')
        # print(timeit.default_timer() - start_outer)
    elif full == 0:
        if len(candidates) == 0:
            break
        candidates = candidates.to_numpy()
        candidates = np.delete(candidates, np.where(candidates == bad_candidate)[0], axis=0)
        candidates = pd.DataFrame(candidates).reset_index(drop=True)
        candidates = candidates.rename(columns={0: 'fict_site'})
        # print(len(candidates), end='  ')
        # print(timeit.default_timer() - start_outer)

end_yr5 = timeit.default_timer()
print(end_yr5 - start_yr5)

print('Year 6', end=' ')
print(len(selected['fict_site'].unique()), end=' ')

pole_du = pole_du[1:11, :]
pole_u = pole_u[1:11, :]
pole_s = pole_s[1:11, :]
pole_r = pole_r[1:11, :]
off_du = off_du[1:11, :]
off_u = off_u[1:11, :]
off_s = off_s[1:11, :]
off_r = off_r[1:11, :]
roe_du = roe_du[1:11, :]
roe_u = roe_u[1:11, :]
roe_s = roe_s[1:11, :]
roe_r = roe_r[1:11, :]
smb_du = smb_du[1:11, :]
smb_u = smb_u[1:11, :]
smb_s = smb_s[1:11, :]
smb_r = smb_r[1:11, :]

start_yr6 = timeit.default_timer()
sites_yr6 = sites_yr6[['fict_site']]
sites_6 = pd.merge(calc_sites, sites_yr6, on='fict_site')

if len(sites_yr6) > 0:
    sites = sites_6
    sites = sites.groupby('fict_site', as_index=False).agg({'morph_code': 'mean', 'Hour_GBs': 'sum'})
    sites['npv'] = sites.apply(lambda x: loop_npv26(x['Hour_GBs'], x['morph_code']), axis=1).astype(int)
    ranking = sites.loc[sites['npv'] > 0].copy()
    ranking['rank'] = ranking['npv'].rank(ascending=False)
    ranking = ranking.sort_values(by='rank')
    temp_ranking = ranking[['fict_site']]
    init_ranking_bins = pd.merge(temp_ranking, sites_6, on='fict_site')
    init_ranking_counts = init_ranking_bins['GridName'].value_counts().reset_index()
    init_ranking_counts = init_ranking_counts.rename(columns={'index': 'GridName', 'GridName': 'bin_count'})
    init_ranking = pd.merge(init_ranking_bins, init_ranking_counts, on='GridName')
    if len(selected) > 0 and len(ranking) > 0:
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[['fict_site', 'npv']]
        new_selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected.append(new_selected, sort=True)
        selected = selected[['fict_site', 'npv']]
    elif len(selected) == 0 and len(ranking) > 0:
        selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected[['fict_site', 'npv']]
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[['fict_site', 'npv']]
    else:
        candidates = pd.DataFrame([])

for j in range(len(candidates)):
    # start_outer = timeit.default_timer()
    temp_nb = pd.DataFrame([])
    for i in range(len(candidates)):
        next_best = (pd.DataFrame(candidates.iloc[i, :])).T
        sinr_sites = selected.append(next_best, sort=True)
        sinr_bins = pd.merge(init_ranking, sinr_sites, on='fict_site')

        sinr_bins_unique = sinr_bins[sinr_bins['bin_count'] == 1].copy()
        sinr_bins_dups = sinr_bins[sinr_bins['bin_count'] > 1].copy()
        temp_sinr_bins_dups = sinr_bins_dups.groupby('GridName')['rx_signal_strength_mw'].sum().reset_index()
        sinr_bins_dups = sinr_bins_dups.drop(['rx_signal_strength_mw'], axis=1)
        sinr_bins_dups = pd.merge(sinr_bins_dups, temp_sinr_bins_dups, on='GridName')
        sinr_bins_unique['sinr_new'] = sinr_bins_unique['sinr']
        sinr_bins_dups['sinr_new'] = sinr_new(sinr_bins_dups['rx_signal_strength_mw'],
                                              sinr_bins_dups['rx_signal_strength_db'])
        sinr_bins = sinr_bins_unique.append(sinr_bins_dups, sort=True)
        temp_sinr_bins = sinr_bins
        sinr_bins = sinr_bins[sinr_bins.sinr_new >= -7.0]

        sinr_bins_agg = sinr_bins.groupby('fict_site', as_index=False).agg({'morph_code': 'mean', 'Hour_GBs': 'sum'})
        sinr_bins_agg['xnpv'] = sinr_bins_agg.apply(lambda x: loop_npv26(x['Hour_GBs'], x['morph_code']),
                                                    axis=1).astype(int)
        next_best['adj_npv'] = sinr_bins_agg['xnpv'].min()
        next_best['sum_npv'] = sinr_bins_agg['xnpv'].sum()
        next_best = next_best[['fict_site', 'adj_npv', 'sum_npv']]
        if next_best['adj_npv'].values <= npv_threshold:
            bad_candidate = next_best['fict_site'].values
            full = 0
            break
        init_cand_bins = pd.merge(next_best, init_ranking, on='fict_site')
        init_cand_bins = init_cand_bins.nlargest(bin_prox, 'sinr')
        init_cand_bins = init_cand_bins[['sinr', 'fict_site']].mean().astype('int64')
        cand_bins = pd.merge(next_best, temp_sinr_bins, on='fict_site')
        cand_bins = cand_bins.nlargest(bin_prox, 'sinr_new')
        cand_bins = cand_bins[['sinr_new']].mean()
        if (abs(init_cand_bins['sinr'] - cand_bins['sinr_new'])) > sinr_deg:
            bad_candidate = init_cand_bins['fict_site'].astype('int64')
            full = 0
            break
        temp_nb = temp_nb.append(next_best, sort=True)
        full = 1
    if full == 1:
        if len(candidates) == 0:
            break
        new_ranking = temp_nb
        new_ranking['rank'] = new_ranking['sum_npv'].rank(ascending=True)
        new_ranking = new_ranking.sort_values(by='rank').reset_index(drop='True')
        temp_selected = new_ranking[new_ranking['rank'] == 1]
        temp_selected = temp_selected[['fict_site']]
        selected = selected.append(temp_selected, sort=True)
        new_ranking = new_ranking[['fict_site']]
        candidates = pd.DataFrame(new_ranking.iloc[1:, :]).reset_index(drop=True)
        # print(len(candidates), end='  ')
        # print(timeit.default_timer() - start_outer)
    elif full == 0:
        if len(candidates) == 0:
            break
        candidates = candidates.to_numpy()
        candidates = np.delete(candidates, np.where(candidates == bad_candidate)[0], axis=0)
        candidates = pd.DataFrame(candidates).reset_index(drop=True)
        candidates = candidates.rename(columns={0: 'fict_site'})
        # print(len(candidates), end='  ')
        # print(timeit.default_timer() - start_outer)

end_yr6 = timeit.default_timer()
print(end_yr6 - start_yr6)

print('Year 7', end=' ')
print(len(selected['fict_site'].unique()), end=' ')

pole_du = pole_du[1:11, :]
pole_u = pole_u[1:11, :]
pole_s = pole_s[1:11, :]
pole_r = pole_r[1:11, :]
off_du = off_du[1:11, :]
off_u = off_u[1:11, :]
off_s = off_s[1:11, :]
off_r = off_r[1:11, :]
roe_du = roe_du[1:11, :]
roe_u = roe_u[1:11, :]
roe_s = roe_s[1:11, :]
roe_r = roe_r[1:11, :]
smb_du = smb_du[1:11, :]
smb_u = smb_u[1:11, :]
smb_s = smb_s[1:11, :]
smb_r = smb_r[1:11, :]

start_yr7 = timeit.default_timer()
sites_yr7 = sites_yr7[['fict_site']]
sites_7 = pd.merge(calc_sites, sites_yr7, on='fict_site')

if len(sites_yr7) > 0:
    sites = sites_7
    sites = sites.groupby('fict_site', as_index=False).agg({'morph_code': 'mean', 'Hour_GBs': 'sum'})
    sites['npv'] = sites.apply(lambda x: loop_npv27(x['Hour_GBs'], x['morph_code']), axis=1).astype(int)
    ranking = sites.loc[sites['npv'] > 0].copy()
    ranking['rank'] = ranking['npv'].rank(ascending=False)
    ranking = ranking.sort_values(by='rank')
    temp_ranking = ranking[['fict_site']]
    init_ranking_bins = pd.merge(temp_ranking, sites_7, on='fict_site')
    init_ranking_counts = init_ranking_bins['GridName'].value_counts().reset_index()
    init_ranking_counts = init_ranking_counts.rename(columns={'index': 'GridName', 'GridName': 'bin_count'})
    init_ranking = pd.merge(init_ranking_bins, init_ranking_counts, on='GridName')
    if len(selected) > 0 and len(ranking) > 0:
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[['fict_site', 'npv']]
        new_selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected.append(new_selected, sort=True)
        selected = selected[['fict_site', 'npv']]
    elif len(selected) == 0 and len(ranking) > 0:
        selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected[['fict_site', 'npv']]
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[['fict_site', 'npv']]
    else:
        candidates = pd.DataFrame([])

for j in range(len(candidates)):
    # start_outer = timeit.default_timer()
    temp_nb = pd.DataFrame([])
    for i in range(len(candidates)):
        next_best = (pd.DataFrame(candidates.iloc[i, :])).T
        sinr_sites = selected.append(next_best, sort=True)
        sinr_bins = pd.merge(init_ranking, sinr_sites, on='fict_site')

        sinr_bins_unique = sinr_bins[sinr_bins['bin_count'] == 1].copy()
        sinr_bins_dups = sinr_bins[sinr_bins['bin_count'] > 1].copy()
        temp_sinr_bins_dups = sinr_bins_dups.groupby('GridName')['rx_signal_strength_mw'].sum().reset_index()
        sinr_bins_dups = sinr_bins_dups.drop(['rx_signal_strength_mw'], axis=1)
        sinr_bins_dups = pd.merge(sinr_bins_dups, temp_sinr_bins_dups, on='GridName')
        sinr_bins_unique['sinr_new'] = sinr_bins_unique['sinr']
        sinr_bins_dups['sinr_new'] = sinr_new(sinr_bins_dups['rx_signal_strength_mw'],
                                              sinr_bins_dups['rx_signal_strength_db'])
        sinr_bins = sinr_bins_unique.append(sinr_bins_dups, sort=True)
        temp_sinr_bins = sinr_bins
        sinr_bins = sinr_bins[sinr_bins.sinr_new >= -7.0]

        sinr_bins_agg = sinr_bins.groupby('fict_site', as_index=False).agg({'morph_code': 'mean', 'Hour_GBs': 'sum'})
        sinr_bins_agg['xnpv'] = sinr_bins_agg.apply(lambda x: loop_npv27(x['Hour_GBs'], x['morph_code']),
                                                    axis=1).astype(int)
        next_best['adj_npv'] = sinr_bins_agg['xnpv'].min()
        next_best['sum_npv'] = sinr_bins_agg['xnpv'].sum()
        next_best = next_best[['fict_site', 'adj_npv', 'sum_npv']]
        if next_best['adj_npv'].values <= npv_threshold:
            bad_candidate = next_best['fict_site'].values
            full = 0
            break
        init_cand_bins = pd.merge(next_best, init_ranking, on='fict_site')
        init_cand_bins = init_cand_bins.nlargest(bin_prox, 'sinr')
        init_cand_bins = init_cand_bins[['sinr', 'fict_site']].mean().astype('int64')
        cand_bins = pd.merge(next_best, temp_sinr_bins, on='fict_site')
        cand_bins = cand_bins.nlargest(bin_prox, 'sinr_new')
        cand_bins = cand_bins[['sinr_new']].mean()
        if (abs(init_cand_bins['sinr'] - cand_bins['sinr_new'])) > sinr_deg:
            bad_candidate = init_cand_bins['fict_site'].astype('int64')
            full = 0
            break
        temp_nb = temp_nb.append(next_best, sort=True)
        full = 1
    if full == 1:
        if len(candidates) == 0:
            break
        new_ranking = temp_nb
        new_ranking['rank'] = new_ranking['sum_npv'].rank(ascending=True)
        new_ranking = new_ranking.sort_values(by='rank').reset_index(drop='True')
        temp_selected = new_ranking[new_ranking['rank'] == 1]
        temp_selected = temp_selected[['fict_site']]
        selected = selected.append(temp_selected, sort=True)
        new_ranking = new_ranking[['fict_site']]
        candidates = pd.DataFrame(new_ranking.iloc[1:, :]).reset_index(drop=True)
        # print(len(candidates), end='  ')
        # print(timeit.default_timer() - start_outer)
    elif full == 0:
        if len(candidates) == 0:
            break
        candidates = candidates.to_numpy()
        candidates = np.delete(candidates, np.where(candidates == bad_candidate)[0], axis=0)
        candidates = pd.DataFrame(candidates).reset_index(drop=True)
        candidates = candidates.rename(columns={0: 'fict_site'})
        # print(len(candidates), end='  ')
        # print(timeit.default_timer() - start_outer)

end_yr7 = timeit.default_timer()
print(end_yr7 - start_yr7)

print('Year 8', end=' ')
print(len(selected['fict_site'].unique()), end=' ')

pole_du = pole_du[1:11, :]
pole_u = pole_u[1:11, :]
pole_s = pole_s[1:11, :]
pole_r = pole_r[1:11, :]
off_du = off_du[1:11, :]
off_u = off_u[1:11, :]
off_s = off_s[1:11, :]
off_r = off_r[1:11, :]
roe_du = roe_du[1:11, :]
roe_u = roe_u[1:11, :]
roe_s = roe_s[1:11, :]
roe_r = roe_r[1:11, :]
smb_du = smb_du[1:11, :]
smb_u = smb_u[1:11, :]
smb_s = smb_s[1:11, :]
smb_r = smb_r[1:11, :]

start_yr8 = timeit.default_timer()
sites_yr8 = sites_yr8[['fict_site']]
sites_8 = pd.merge(calc_sites, sites_yr8, on='fict_site')

if len(sites_yr8) > 0:
    sites = sites_8
    sites = sites.groupby('fict_site', as_index=False).agg({'morph_code': 'mean', 'Hour_GBs': 'sum'})
    sites['npv'] = sites.apply(lambda x: loop_npv28(x['Hour_GBs'], x['morph_code']), axis=1).astype(int)
    ranking = sites.loc[sites['npv'] > 0].copy()
    ranking['rank'] = ranking['npv'].rank(ascending=False)
    ranking = ranking.sort_values(by='rank')
    temp_ranking = ranking[['fict_site']]
    init_ranking_bins = pd.merge(temp_ranking, sites_8, on='fict_site')
    init_ranking_counts = init_ranking_bins['GridName'].value_counts().reset_index()
    init_ranking_counts = init_ranking_counts.rename(columns={'index': 'GridName', 'GridName': 'bin_count'})
    init_ranking = pd.merge(init_ranking_bins, init_ranking_counts, on='GridName')
    if len(selected) > 0 and len(ranking) > 0:
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[['fict_site', 'npv']]
        new_selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected.append(new_selected, sort=True)
        selected = selected[['fict_site', 'npv']]
    elif len(selected) == 0 and len(ranking) > 0:
        selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected[['fict_site', 'npv']]
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[['fict_site', 'npv']]
    else:
        candidates = pd.DataFrame([])

for j in range(len(candidates)):
    # start_outer = timeit.default_timer()
    temp_nb = pd.DataFrame([])
    for i in range(len(candidates)):
        next_best = (pd.DataFrame(candidates.iloc[i, :])).T
        sinr_sites = selected.append(next_best, sort=True)
        sinr_bins = pd.merge(init_ranking, sinr_sites, on='fict_site')

        sinr_bins_unique = sinr_bins[sinr_bins['bin_count'] == 1].copy()
        sinr_bins_dups = sinr_bins[sinr_bins['bin_count'] > 1].copy()
        temp_sinr_bins_dups = sinr_bins_dups.groupby('GridName')['rx_signal_strength_mw'].sum().reset_index()
        sinr_bins_dups = sinr_bins_dups.drop(['rx_signal_strength_mw'], axis=1)
        sinr_bins_dups = pd.merge(sinr_bins_dups, temp_sinr_bins_dups, on='GridName')
        sinr_bins_unique['sinr_new'] = sinr_bins_unique['sinr']
        sinr_bins_dups['sinr_new'] = sinr_new(sinr_bins_dups['rx_signal_strength_mw'],
                                              sinr_bins_dups['rx_signal_strength_db'])
        sinr_bins = sinr_bins_unique.append(sinr_bins_dups, sort=True)
        temp_sinr_bins = sinr_bins
        sinr_bins = sinr_bins[sinr_bins.sinr_new >= -7.0]

        sinr_bins_agg = sinr_bins.groupby('fict_site', as_index=False).agg({'morph_code': 'mean', 'Hour_GBs': 'sum'})
        sinr_bins_agg['xnpv'] = sinr_bins_agg.apply(lambda x: loop_npv28(x['Hour_GBs'], x['morph_code']),
                                                    axis=1).astype(int)
        next_best['adj_npv'] = sinr_bins_agg['xnpv'].min()
        next_best['sum_npv'] = sinr_bins_agg['xnpv'].sum()
        next_best = next_best[['fict_site', 'adj_npv', 'sum_npv']]
        if next_best['adj_npv'].values <= npv_threshold:
            bad_candidate = next_best['fict_site'].values
            full = 0
            break
        init_cand_bins = pd.merge(next_best, init_ranking, on='fict_site')
        init_cand_bins = init_cand_bins.nlargest(bin_prox, 'sinr')
        init_cand_bins = init_cand_bins[['sinr', 'fict_site']].mean().astype('int64')
        cand_bins = pd.merge(next_best, temp_sinr_bins, on='fict_site')
        cand_bins = cand_bins.nlargest(bin_prox, 'sinr_new')
        cand_bins = cand_bins[['sinr_new']].mean()
        if (abs(init_cand_bins['sinr'] - cand_bins['sinr_new'])) > sinr_deg:
            bad_candidate = init_cand_bins['fict_site'].astype('int64')
            full = 0
            break
        temp_nb = temp_nb.append(next_best, sort=True)
        full = 1
    if full == 1:
        if len(candidates) == 0:
            break
        new_ranking = temp_nb
        new_ranking['rank'] = new_ranking['sum_npv'].rank(ascending=True)
        new_ranking = new_ranking.sort_values(by='rank').reset_index(drop='True')
        temp_selected = new_ranking[new_ranking['rank'] == 1]
        temp_selected = temp_selected[['fict_site']]
        selected = selected.append(temp_selected, sort=True)
        new_ranking = new_ranking[['fict_site']]
        candidates = pd.DataFrame(new_ranking.iloc[1:, :]).reset_index(drop=True)
        # print(len(candidates), end='  ')
        # print(timeit.default_timer() - start_outer)
    elif full == 0:
        if len(candidates) == 0:
            break
        candidates = candidates.to_numpy()
        candidates = np.delete(candidates, np.where(candidates == bad_candidate)[0], axis=0)
        candidates = pd.DataFrame(candidates).reset_index(drop=True)
        candidates = candidates.rename(columns={0: 'fict_site'})
        # print(len(candidates), end='  ')
        # print(timeit.default_timer() - start_outer)

end_yr8 = timeit.default_timer()
print(end_yr8 - start_yr8)

print('Year 9', end=' ')
print(len(selected['fict_site'].unique()), end=' ')

pole_du = pole_du[1:11, :]
pole_u = pole_u[1:11, :]
pole_s = pole_s[1:11, :]
pole_r = pole_r[1:11, :]
off_du = off_du[1:11, :]
off_u = off_u[1:11, :]
off_s = off_s[1:11, :]
off_r = off_r[1:11, :]
roe_du = roe_du[1:11, :]
roe_u = roe_u[1:11, :]
roe_s = roe_s[1:11, :]
roe_r = roe_r[1:11, :]
smb_du = smb_du[1:11, :]
smb_u = smb_u[1:11, :]
smb_s = smb_s[1:11, :]
smb_r = smb_r[1:11, :]

start_yr9 = timeit.default_timer()
sites_yr9 = sites_yr9[['fict_site']]
sites_9 = pd.merge(calc_sites, sites_yr9, on='fict_site')

if len(sites_yr9) > 0:
    sites = sites_9
    sites = sites.groupby('fict_site', as_index=False).agg({'morph_code': 'mean', 'Hour_GBs': 'sum'})
    sites['npv'] = sites.apply(lambda x: loop_npv29(x['Hour_GBs'], x['morph_code']), axis=1).astype(int)
    ranking = sites.loc[sites['npv'] > 0].copy()
    ranking['rank'] = ranking['npv'].rank(ascending=False)
    ranking = ranking.sort_values(by='rank')
    temp_ranking = ranking[['fict_site']]
    init_ranking_bins = pd.merge(temp_ranking, sites_9, on='fict_site')
    init_ranking_counts = init_ranking_bins['GridName'].value_counts().reset_index()
    init_ranking_counts = init_ranking_counts.rename(columns={'index': 'GridName', 'GridName': 'bin_count'})
    init_ranking = pd.merge(init_ranking_bins, init_ranking_counts, on='GridName')
    if len(selected) > 0 and len(ranking) > 0:
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[['fict_site', 'npv']]
        new_selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected.append(new_selected, sort=True)
        selected = selected[['fict_site', 'npv']]
    elif len(selected) == 0 and len(ranking) > 0:
        selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected[['fict_site', 'npv']]
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[['fict_site', 'npv']]
    else:
        candidates = pd.DataFrame([])

for j in range(len(candidates)):
    # start_outer = timeit.default_timer()
    temp_nb = pd.DataFrame([])
    for i in range(len(candidates)):
        next_best = (pd.DataFrame(candidates.iloc[i, :])).T
        sinr_sites = selected.append(next_best, sort=True)
        sinr_bins = pd.merge(init_ranking, sinr_sites, on='fict_site')

        sinr_bins_unique = sinr_bins[sinr_bins['bin_count'] == 1].copy()
        sinr_bins_dups = sinr_bins[sinr_bins['bin_count'] > 1].copy()
        temp_sinr_bins_dups = sinr_bins_dups.groupby('GridName')['rx_signal_strength_mw'].sum().reset_index()
        sinr_bins_dups = sinr_bins_dups.drop(['rx_signal_strength_mw'], axis=1)
        sinr_bins_dups = pd.merge(sinr_bins_dups, temp_sinr_bins_dups, on='GridName')
        sinr_bins_unique['sinr_new'] = sinr_bins_unique['sinr']
        sinr_bins_dups['sinr_new'] = sinr_new(sinr_bins_dups['rx_signal_strength_mw'],
                                              sinr_bins_dups['rx_signal_strength_db'])
        sinr_bins = sinr_bins_unique.append(sinr_bins_dups, sort=True)
        temp_sinr_bins = sinr_bins
        sinr_bins = sinr_bins[sinr_bins.sinr_new >= -7.0]

        sinr_bins_agg = sinr_bins.groupby('fict_site', as_index=False).agg({'morph_code': 'mean', 'Hour_GBs': 'sum'})
        sinr_bins_agg['xnpv'] = sinr_bins_agg.apply(lambda x: loop_npv29(x['Hour_GBs'], x['morph_code']),
                                                    axis=1).astype(int)
        next_best['adj_npv'] = sinr_bins_agg['xnpv'].min()
        next_best['sum_npv'] = sinr_bins_agg['xnpv'].sum()
        next_best = next_best[['fict_site', 'adj_npv', 'sum_npv']]
        if next_best['adj_npv'].values <= npv_threshold:
            bad_candidate = next_best['fict_site'].values
            full = 0
            break
        init_cand_bins = pd.merge(next_best, init_ranking, on='fict_site')
        init_cand_bins = init_cand_bins.nlargest(bin_prox, 'sinr')
        init_cand_bins = init_cand_bins[['sinr', 'fict_site']].mean().astype('int64')
        cand_bins = pd.merge(next_best, temp_sinr_bins, on='fict_site')
        cand_bins = cand_bins.nlargest(bin_prox, 'sinr_new')
        cand_bins = cand_bins[['sinr_new']].mean()
        if (abs(init_cand_bins['sinr'] - cand_bins['sinr_new'])) > sinr_deg:
            bad_candidate = init_cand_bins['fict_site'].astype('int64')
            full = 0
            break
        temp_nb = temp_nb.append(next_best, sort=True)
        full = 1
    if full == 1:
        if len(candidates) == 0:
            break
        new_ranking = temp_nb
        new_ranking['rank'] = new_ranking['sum_npv'].rank(ascending=True)
        new_ranking = new_ranking.sort_values(by='rank').reset_index(drop='True')
        temp_selected = new_ranking[new_ranking['rank'] == 1]
        temp_selected = temp_selected[['fict_site']]
        selected = selected.append(temp_selected, sort=True)
        new_ranking = new_ranking[['fict_site']]
        candidates = pd.DataFrame(new_ranking.iloc[1:, :]).reset_index(drop=True)
        # print(len(candidates), end='  ')
        # print(timeit.default_timer() - start_outer)
    elif full == 0:
        if len(candidates) == 0:
            break
        candidates = candidates.to_numpy()
        candidates = np.delete(candidates, np.where(candidates == bad_candidate)[0], axis=0)
        candidates = pd.DataFrame(candidates).reset_index(drop=True)
        candidates = candidates.rename(columns={0: 'fict_site'})
        # print(len(candidates), end='  ')
        # print(timeit.default_timer() - start_outer)

end_yr9 = timeit.default_timer()
print(end_yr9 - start_yr9)

pole_du = pole_du[1:11, :]
pole_u = pole_u[1:11, :]
pole_s = pole_s[1:11, :]
pole_r = pole_r[1:11, :]
off_du = off_du[1:11, :]
off_u = off_u[1:11, :]
off_s = off_s[1:11, :]
off_r = off_r[1:11, :]
roe_du = roe_du[1:11, :]
roe_u = roe_u[1:11, :]
roe_s = roe_s[1:11, :]
roe_r = roe_r[1:11, :]
smb_du = smb_du[1:11, :]
smb_u = smb_u[1:11, :]
smb_s = smb_s[1:11, :]
smb_r = smb_r[1:11, :]

print('Year 10', end=' ')
print(len(selected['fict_site'].unique()), end=' ')

start_yr10 = timeit.default_timer()
sites_yr10 = sites_yr10[['fict_site']]
sites_10 = pd.merge(calc_sites, sites_yr10, on='fict_site')

if len(sites_yr10) > 0:
    sites = sites_10
    sites = sites.groupby('fict_site', as_index=False).agg({'morph_code': 'mean', 'Hour_GBs': 'sum'})
    sites['npv'] = sites.apply(lambda x: loop_npv30(x['Hour_GBs'], x['morph_code']), axis=1).astype(int)
    ranking = sites.loc[sites['npv'] > 0].copy()
    ranking['rank'] = ranking['npv'].rank(ascending=False)
    ranking = ranking.sort_values(by='rank')
    temp_ranking = ranking[['fict_site']]
    init_ranking_bins = pd.merge(temp_ranking, sites_10, on='fict_site')
    init_ranking_counts = init_ranking_bins['GridName'].value_counts().reset_index()
    init_ranking_counts = init_ranking_counts.rename(columns={'index': 'GridName', 'GridName': 'bin_count'})
    init_ranking = pd.merge(init_ranking_bins, init_ranking_counts, on='GridName')
    if len(selected) > 0 and len(ranking) > 0:
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[['fict_site', 'npv']]
        new_selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected.append(new_selected, sort=True)
        selected = selected[['fict_site', 'npv']]
    elif len(selected) == 0 and len(ranking) > 0:
        selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected[['fict_site', 'npv']]
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[['fict_site', 'npv']]
    else:
        candidates = pd.DataFrame([])

for j in range(len(candidates)):
    # start_outer = timeit.default_timer()
    temp_nb = pd.DataFrame([])
    for i in range(len(candidates)):
        next_best = (pd.DataFrame(candidates.iloc[i, :])).T
        sinr_sites = selected.append(next_best, sort=True)
        sinr_bins = pd.merge(init_ranking, sinr_sites, on='fict_site')

        sinr_bins_unique = sinr_bins[sinr_bins['bin_count'] == 1].copy()
        sinr_bins_dups = sinr_bins[sinr_bins['bin_count'] > 1].copy()
        temp_sinr_bins_dups = sinr_bins_dups.groupby('GridName')['rx_signal_strength_mw'].sum().reset_index()
        sinr_bins_dups = sinr_bins_dups.drop(['rx_signal_strength_mw'], axis=1)
        sinr_bins_dups = pd.merge(sinr_bins_dups, temp_sinr_bins_dups, on='GridName')
        sinr_bins_unique['sinr_new'] = sinr_bins_unique['sinr']
        sinr_bins_dups['sinr_new'] = sinr_new(sinr_bins_dups['rx_signal_strength_mw'],
                                              sinr_bins_dups['rx_signal_strength_db'])
        sinr_bins = sinr_bins_unique.append(sinr_bins_dups, sort=True)
        temp_sinr_bins = sinr_bins
        sinr_bins = sinr_bins[sinr_bins.sinr_new >= -7.0]

        sinr_bins_agg = sinr_bins.groupby('fict_site', as_index=False).agg({'morph_code': 'mean', 'Hour_GBs': 'sum'})
        sinr_bins_agg['xnpv'] = sinr_bins_agg.apply(lambda x: loop_npv30(x['Hour_GBs'], x['morph_code']),
                                                    axis=1).astype(int)
        next_best['adj_npv'] = sinr_bins_agg['xnpv'].min()
        next_best['sum_npv'] = sinr_bins_agg['xnpv'].sum()
        next_best = next_best[['fict_site', 'adj_npv', 'sum_npv']]
        if next_best['adj_npv'].values <= npv_threshold:
            bad_candidate = next_best['fict_site'].values
            full = 0
            break
        init_cand_bins = pd.merge(next_best, init_ranking, on='fict_site')
        init_cand_bins = init_cand_bins.nlargest(bin_prox, 'sinr')
        init_cand_bins = init_cand_bins[['sinr', 'fict_site']].mean().astype('int64')
        cand_bins = pd.merge(next_best, temp_sinr_bins, on='fict_site')
        cand_bins = cand_bins.nlargest(bin_prox, 'sinr_new')
        cand_bins = cand_bins[['sinr_new']].mean()
        if (abs(init_cand_bins['sinr'] - cand_bins['sinr_new'])) > sinr_deg:
            bad_candidate = init_cand_bins['fict_site'].astype('int64')
            full = 0
            break
        temp_nb = temp_nb.append(next_best, sort=True)
        full = 1
    if full == 1:
        if len(candidates) == 0:
            break
        new_ranking = temp_nb
        new_ranking['rank'] = new_ranking['sum_npv'].rank(ascending=True)
        new_ranking = new_ranking.sort_values(by='rank').reset_index(drop='True')
        temp_selected = new_ranking[new_ranking['rank'] == 1]
        temp_selected = temp_selected[['fict_site']]
        selected = selected.append(temp_selected, sort=True)
        new_ranking = new_ranking[['fict_site']]
        candidates = pd.DataFrame(new_ranking.iloc[1:, :]).reset_index(drop=True)
        # print(len(candidates), end='  ')
        # print(timeit.default_timer() - start_outer)
    elif full == 0:
        if len(candidates) == 0:
            break
        candidates = candidates.to_numpy()
        candidates = np.delete(candidates, np.where(candidates == bad_candidate)[0], axis=0)
        candidates = pd.DataFrame(candidates).reset_index(drop=True)
        candidates = candidates.rename(columns={0: 'fict_site'})
        # print(len(candidates), end='  ')
        # print(timeit.default_timer() - start_outer)

end_yr10 = timeit.default_timer()
print(end_yr10 - start_yr10)

pole_du = fin_arrays(1)
pole_u = fin_arrays(2)
pole_s = fin_arrays(3)
pole_r = fin_arrays(4)
off_du = fin_arrays(5)
off_u = fin_arrays(6)
off_s = fin_arrays(7)
off_r = fin_arrays(8)
roe_du = fin_arrays(9)
roe_u = fin_arrays(10)
roe_s = fin_arrays(11)
roe_r = fin_arrays(12)
smb_du = fin_arrays(13)
smb_u = fin_arrays(14)
smb_s = fin_arrays(15)
smb_r = fin_arrays(16)

selected = pd.merge(selected, init_build_yr_sites, on='fict_site')
selected = pd.merge(selected, init_sites, on='fict_site')

bld_yr_sites['build_yr'] = inh_bld_yr.idxmin(axis=1).astype(int)
print(bld_yr_sites['build_yr'].value_counts())
init_build_yr_sites = bld_yr_sites[['fict_site', 'build_yr']]

site_bin_gbs = selected.groupby('fict_site')['Hour_GBs'].sum().reset_index()
bld_yr_sites = pd.merge(site_bin_gbs, site_t_m, on='fict_site')

bld_yr_sites['1'] = bld_yr_sites.apply(lambda x: bld_npv21(x['Hour_GBs'], x['morph_code']), axis=1)
bld_yr_sites['2'] = bld_yr_sites.apply(lambda x: bld_npv22(x['Hour_GBs'], x['morph_code']), axis=1)
bld_yr_sites['3'] = bld_yr_sites.apply(lambda x: bld_npv23(x['Hour_GBs'], x['morph_code']), axis=1)
bld_yr_sites['4'] = bld_yr_sites.apply(lambda x: bld_npv24(x['Hour_GBs'], x['morph_code']), axis=1)
bld_yr_sites['5'] = bld_yr_sites.apply(lambda x: bld_npv25(x['Hour_GBs'], x['morph_code']), axis=1)
bld_yr_sites['6'] = bld_yr_sites.apply(lambda x: bld_npv26(x['Hour_GBs'], x['morph_code']), axis=1)
bld_yr_sites['7'] = bld_yr_sites.apply(lambda x: bld_npv27(x['Hour_GBs'], x['morph_code']), axis=1)
bld_yr_sites['8'] = bld_yr_sites.apply(lambda x: bld_npv28(x['Hour_GBs'], x['morph_code']), axis=1)
bld_yr_sites['9'] = bld_yr_sites.apply(lambda x: bld_npv29(x['Hour_GBs'], x['morph_code']), axis=1)
bld_yr_sites['10'] = bld_yr_sites.apply(lambda x: bld_npv30(x['Hour_GBs'], x['morph_code']), axis=1)

inh_bld_yr = bld_yr_sites[['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']]

bld_yr_sites['design_build_yr'] = inh_bld_yr.idxmin(axis=1).astype(int)
final_build_yr_sites = bld_yr_sites[['fict_site', 'design_build_yr']]
selected = pd.merge(selected, final_build_yr_sites, on='fict_site')
selected['opt_build_year'] = ((selected['design_build_yr'] + selected['build_yr']) / 2).round(0)

# !!! need to update this code and add in smbs

# ADDITION OF ROEs - comment out with WTG test case

# sans_roe = selected
#
# sans_roe = sans_roe.drop(['npv'], axis=1)
#
# sites_roe['morph_code'] = sites_roe.apply(lambda x: morph(x['asset_type'], x['morphology']), axis=1)
# temp_sans_roe = sans_roe[['GridName']]
# roe_merge = sites_roe.merge(temp_sans_roe.drop_duplicates(), on='GridName', how='left', indicator=True)
# # noinspection PyProtectedMember
# sites_roe = roe_merge[roe_merge._merge == 'left_only']
# sites_roe = sites_roe.drop(['asset_type', 'morphology', '_merge'], axis=1)
#
# sum_sites_roe = pd.merge(sites_roe, day_usage, on='GridName')
# site_t_m = sites_roe.groupby('fict_site')['morph_code'].mean().reset_index()
# site_bin_gbs = sum_sites_roe.groupby('fict_site')['Hour_GBs'].sum().reset_index()
# bld_yr_sites = pd.merge(site_bin_gbs, site_t_m, on='fict_site')
#
# bld_yr_sites['1'] = bld_yr_sites.apply(lambda x: npv21(x['Hour_GBs'], x['morph_code']), axis=1)
# bld_yr_sites['2'] = bld_yr_sites.apply(lambda x: npv22(x['Hour_GBs'], x['morph_code']), axis=1)
# bld_yr_sites['3'] = bld_yr_sites.apply(lambda x: npv23(x['Hour_GBs'], x['morph_code']), axis=1)
# bld_yr_sites['4'] = bld_yr_sites.apply(lambda x: npv24(x['Hour_GBs'], x['morph_code']), axis=1)
# bld_yr_sites['5'] = bld_yr_sites.apply(lambda x: npv25(x['Hour_GBs'], x['morph_code']), axis=1)
# bld_yr_sites['6'] = bld_yr_sites.apply(lambda x: npv26(x['Hour_GBs'], x['morph_code']), axis=1)
# bld_yr_sites['7'] = bld_yr_sites.apply(lambda x: npv27(x['Hour_GBs'], x['morph_code']), axis=1)
# bld_yr_sites['8'] = bld_yr_sites.apply(lambda x: npv28(x['Hour_GBs'], x['morph_code']), axis=1)
# bld_yr_sites['9'] = bld_yr_sites.apply(lambda x: npv29(x['Hour_GBs'], x['morph_code']), axis=1)
# bld_yr_sites['10'] = bld_yr_sites.apply(lambda x: npv30(x['Hour_GBs'], x['morph_code']), axis=1)
#
# inh_bld_yr = bld_yr_sites[['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']]
#
# bld_yr_sites['build_yr'] = inh_bld_yr.idxmax(axis=1).astype(int)
# roe_build_yr_sites = bld_yr_sites[['fict_site', 'build_yr']]
# sites_roe = pd.merge(roe_build_yr_sites, sites_roe, on='fict_site')
# sites_roe = pd.merge(sites_roe, day_usage, on='GridName')
# sites_roe['design_build_yr'] = sites_roe['build_yr']
# sites_roe['build_year'] = sites_roe['build_yr']
#
# selected = sans_roe.append(sites_roe, sort=True)

end_time = timeit.default_timer()

print(len(selected['fict_site'].unique()), end=' ')

print('site count', end=' ')
print(site_number)
print('total time seconds', end=' ')
total_time = round((end_time - init_time), 2)
print(total_time)
print('total time minutes', end=' ')
print(total_time / 60)
print('time per site', end=' ')
print(total_time / initial_sites.values)
selected.to_csv('12057_test.csv')

content = '12057_seg complete'
mail = smtplib.SMTP('smtp.gmail.com', 587)
mail.ehlo()
mail.starttls()
mail.login('notifications.norris@gmail.com', 'Keynes92')
mail.sendmail('notifications.norris@gmail.com', 'matthew@mdnorris.com', content)
mail.close()
print("Email Sent")
