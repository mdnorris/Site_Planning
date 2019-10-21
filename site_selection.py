# SITE SELECTION ALGORITHM V2
# MATTHEW NORRIS (C-MATTHEW.NORRIS@CHARTER.COM)
# LAST UPDATED 10/21/19

# this is a non-optimized algorithm that does the
# following: 1) take in inputs (Chris gave me available tables
# with fields and data type) 2) initially calculate an
# adjusted NPV value based on interference and drop all
# negative NPV sites 3) In the first iteration, select
# the highest asset NPV and compare individually
# with each other asset to adjust the individual site
# NPV by the interference from each pair 4) identify
# the second best asset, then "group" the first and second
# and compare the interference between those two sites
# and every other possible site, dropping negative NPV
# values through each iteration until an iteration is
# run where there are no more NPV positive sites remaining

# !!! an important note is the final output of this method
# is predicated on which sites are chosen to be selected
# first, as they adjust the SINR values for all subsequent
# sites, which adjusts NPV and gives the next iteration
# an NPV order to base its decision on

# DATA

# After looking over the tables that Chris sent us,
# Jason and I believe that the variables oi_lines.asset_id,
# aoi_lines.bin_30, aoi_lines.min_total_loss_db,
# npv_assign.asset_id, and npv_assign.tonnage_gb_all
# should be sufficient for this first test

import pandas as pd
from datetime import date
import numpy as np

# import .csv file as test data

# test_data = '/home/usr/Downloads/ss_v1_test_data.csv'
# df = pd.read_csv(test_data)

# NPV DATA #

disc_rate = .15
offload_mo_gb = 1
init_penetration_rate = .00753477
other = 60
max_cap_mo = 8020.89

# capex and opex currently have differing values based on morphology, adjust in v2
capex = [19055, 184585, 17929, 17391, 16869, 16363, 15872, 15396, 14486, 14051]
opex = [3918, 4036, 4157, 4281, 4410, 4542, 4678, 4819, 4963, 5112, 5266]
tot_growth = [1, 9.72, 34.43, 94.24, 241.16, 538.44, 1033.72, 1792.53, 2944.98, 4705.43, 7401.58]
chtr_mvno_rate = [2.78, 2.62, 2.47, 2.32, 2.19, 2.06, 1.94, 1.83, 1.72, 1.62, 1.53]
value = np.asarray([1, 9.72, 34.43, 94.24, 241.16, 538.44, 1033.72, 1792.53, 2944.98, 4705.43])
st_date = pd.date_range(start=pd.Timestamp('2020-01-01'), periods=10, freq='AS')

# testing variables
adj_fact = .123
offload_mo_gb_new = 1
sum_strength_mw = 10.0
sinr = 7
strength_db = 10
sens_db = 10


def xnpv(value):
    for i in range(10):
        if offload_mo_gb_new * init_penetration_rate * (tot_growth[i]) * 12 < max_cap_mo * 12:
            value[i] = offload_mo_gb_new * init_penetration_rate * 12 * (chtr_mvno_rate[i]) - \
                       (capex[i]) - (opex[i])
        else:
            value[i] = max_cap_mo * 12 * (chtr_mvno_rate[i]) - (capex[i]) - \
                       (opex[i]) - other
    date_0 = st_date[0]
    return sum([value_i / (1.0 + disc_rate) ** ((date_i - date_0).days / 365.0)
                for value_i, date_i in zip(value, st_date)])


def sinr_calc(sum_strength_mw, strength_db, sens_db):
    if pd.isnull(sum_strength_mw):
        return sinr
    else:
        return strength_db - (np.log10(sum_strength_mw) + (sens_db / 10) ** 10 / 1000)


def select_site(col_1=sum_strength_mw, col_2=strength_db, col_3=sens_db, col_4=offload_mo_gb):
    offload_mo_gb_new =  sinr_calc(col_1, col_2, col_3) * adj_fact * value
    npv_adj = xnpv(offload_mo_gb_new)
    # if adjusted npv is negative, drop
    print(npv_adj)


# ALGORITHM v1

# FIRST LOOP

select_site(1, 2, 3, 4)

# NEXT N ITERATIONS
# slowly create new array with "best" sites
# delete sites with negative NPV in original array

# STOPPING POINT

# after the original array has no more NPV positive
# values when compared with new array, exit loop
# sum adjusted NPV for SINR-adjusted county NPV