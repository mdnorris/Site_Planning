import smtplib
import timeit
import numpy as np
import pandas as pd

init_time = timeit.default_timer()
sites = pd.read_csv("locations.csv", delimiter=",")
npv_values = pd.read_csv("financials.csv", delimiter=",")
rev_min = 0.0
prox = 17
sig_loss = 10.0
density_pairs = {11: 1, 13: 2, 17: 3, 19: 4}


# SECTION 2: FUNCTIONS


def asset(asset_in):
    asset_type = 0
    if asset_in == "Pole":
        asset_type = 1
    return asset_type


def density(density_in):
    density_type = 0
    if density_in == "Dense Urban":
        density_type = 11
    elif density_in == "Urban":
        density_type = 13
    elif density_in == "Suburban":
        density_type = 17
    elif density_in == "Rural":
        density_type = 19
    return density_type


def asset_density(asset_num, density_num):
    a_m = asset_num * density_num
    return density_pairs[a_m]


def density_array(code):
    ma_pole = ""
    if code == 1:
        ma_pole = pole_du
    elif code == 2:
        ma_pole = pole_u
    elif code == 3:
        ma_pole = pole_s
    elif code == 4:
        ma_pole = pole_r
    return ma_pole


def financial_calcs(code):
    cpx_pole = np.empty([11, 1])
    opx_pole = np.empty([11, 1])
    growth_pole = np.empty([11, 1])
    mvno_pole = np.empty([11, 1])
    npv_array_pole = np.zeros([len(npv_values), 1])
    if code == 1:
        npv_array_pole = npv_values.loc[:, "pole_du"]
    elif code == 2:
        npv_array_pole = npv_values.loc[:, "pole_u"]
    elif code == 3:
        npv_array_pole = npv_values.loc[:, "pole_s"]
    elif code == 4:
        npv_array_pole = npv_values.loc[:, "pole_r"]
    for m in range(11):
        cpx_pole[m] = float(npv_array_pole[m + 1])
    for m in range(11):
        opx_pole[m] = float(npv_array_pole[m + 12])
    for m in range(11):
        growth_pole[m] = float(npv_array_pole[m + 23])
    for m in range(11):
        mvno_pole[m] = float(npv_array_pole[m + 34])
    end_array = np.hstack((cpx_pole, opx_pole, growth_pole, mvno_pole))
    return end_array


def bld_npv(yr_bld, gbs, code):
    array = density_array(code)
    array = array[yr_bld:11, :]
    value = (gbs * 365 * array[0][3] * array[0][2] - array[0][1] - array[0][0]) / (
        1.15 ** yr_bld
    )
    if value > 0:
        year = 1
    elif value < 0:
        year = 12
    else:
        year = 11
    return year


def loop_npv(yr, gbs, code):
    n = 11 - yr
    value = np.empty([n, 1])
    year = decade[-n:]
    array = density_array(code)
    for m in range(n):
        value[m] = gbs * 365 * array[m][3] * array[m][2] - array[m][1]
    value[0] = value[0] - array[0][0]
    return int(sum([value_i / (1.15 ** yr) for value_i, year_i in zip(value, year)]))


def rx_calc(pl_db):
    sig_db = 37.0 - pl_db
    noise = (sig_db - -89.4).round(0).astype(int)
    sig_mw = 10 ** ((37.0 - pl_db) / 10) / 1000.000000000
    return sig_db, noise, sig_mw


def noise_new(sum_sig, sig_db):
    new_noise = np.round(
        (
            sum_sig.where(sum_sig.isnull(), sig_db -
                          (np.log10((sum_sig * 10000 + 114.815),)), 0)
        )
    )
    return new_noise

# SECTION 3: Pre-calculations
pole_du = financial_calcs(1)
pole_u = financial_calcs(2)
pole_s = financial_calcs(3)
pole_r = financial_calcs(4)

sites["asset_id"] = sites.apply(lambda x: asset(x["Pole"]), axis=1)
sites["density_id"] = sites.apply(lambda x: density(x["density"]), axis=1)
sites["density_code"] = sites.apply(
    lambda x: asset_density(x["asset_id"], x["density_id"]), axis=1
)
sites = sites.drop(["Pole", "density", "asset_id", "density_id"], axis=1)
site_t_m = sites.groupby("fake_site")["density_code"].mean().reset_index()
init_sites = sites

# calculation of build years from inherent npv
site_bin_gbs = sites.groupby("fake_site")["Hour_GBs"].sum().reset_index()
bld_yr_sites = pd.merge(site_bin_gbs, site_t_m, on="fake_site")

for q in range(10):
    bld_yr_sites[q + 1] = bld_yr_sites.apply(
        lambda x: bld_npv(q + 1, x["Hour_GBs"], x["density_code"]), axis=1
    )

inh_bld_yr = bld_yr_sites[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
bld_yr_sites["build_yr"] = inh_bld_yr.idxmin(axis=1).astype(int)
bld_yr_sites = bld_yr_sites[bld_yr_sites[10] != 12]
print(bld_yr_sites["build_yr"].value_counts())
init_build_yr_sites = bld_yr_sites[["fake_site", "build_yr"]]

sites_yr1 = bld_yr_sites.loc[bld_yr_sites["build_yr"] == 1]
sites_yr2 = bld_yr_sites.loc[bld_yr_sites["build_yr"] == 2]
sites_yr3 = bld_yr_sites.loc[bld_yr_sites["build_yr"] == 3]
sites_yr4 = bld_yr_sites.loc[bld_yr_sites["build_yr"] == 4]
sites_yr5 = bld_yr_sites.loc[bld_yr_sites["build_yr"] == 5]
sites_yr6 = bld_yr_sites.loc[bld_yr_sites["build_yr"] == 6]
sites_yr7 = bld_yr_sites.loc[bld_yr_sites["build_yr"] == 7]
sites_yr8 = bld_yr_sites.loc[bld_yr_sites["build_yr"] == 8]
sites_yr9 = bld_yr_sites.loc[bld_yr_sites["build_yr"] == 9]
sites_yr10 = bld_yr_sites.loc[bld_yr_sites["build_yr"] == 10]

site_number = (
    len(sites_yr1)
    + len(sites_yr2)
    + len(sites_yr3)
    + len(sites_yr4)
    + len(sites_yr5)
    + len(sites_yr6)
    + len(sites_yr7)
    + len(sites_yr8)
    + len(sites_yr9)
    + len(sites_yr10)
)

selected = pd.DataFrame([])

print("Year_1", end=" ")

pole_du = pole_du[1:11, :]
pole_u = pole_u[1:11, :]
pole_s = pole_s[1:11, :]
pole_r = pole_r[1:11, :]

(init_sites["sig_db"], init_sites["noise"], init_sites["sig_mw"],) = rx_calc(
    init_sites["pl_db"]
)
init_sites.loc[init_sites["noise"] > 50, "noise"] = 50.0
calc_sites = init_sites[
    ["fake_site", "hex", "Hour_GBs", "noise", "sig_db", "sig_mw", "density_code",]
]

# note that declaring variables and following if statement avoid errors
decade = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
sites_yr1 = sites_yr1[["fake_site"]]
sites_1 = pd.merge(calc_sites, sites_yr1, on="fake_site")
start_yr1 = timeit.default_timer()
candidates = pd.DataFrame([])
init_ranking = pd.DataFrame([])
bad_candidate = 0.0
full = 1
year_1 = 1

if len(sites_yr1) > 0:
    sites = sites_1
    sites = sites.groupby("fake_site", as_index=False).agg(
        {"density_code": "mean", "Hour_GBs": "sum"}
    )
    sites["npv"] = sites.apply(
        lambda x: loop_npv(year_1, x["Hour_GBs"], x["density_code"]), axis=1
    ).astype(int)
    ranking = sites.loc[sites["npv"] > 0].copy()
    ranking["rank"] = ranking["npv"].rank(ascending=False)
    ranking = ranking.sort_values(by="rank")
    temp_ranking = ranking[["fake_site"]]
    init_ranking_bins = pd.merge(temp_ranking, sites_1, on="fake_site")
    init_ranking_counts = init_ranking_bins["hex"].value_counts().reset_index()
    init_ranking_counts = init_ranking_counts.rename(
        columns={"index": "hex", "hex": "bin_count"}
    )
    init_ranking = pd.merge(init_ranking_bins, init_ranking_counts, on="hex")
    if len(selected) > 0 and len(ranking) > 0:
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[["fake_site", "npv"]]
        selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected.append(selected, sort=True)
        selected = selected[["fake_site", "npv"]]
    elif len(selected) == 0 and len(ranking) > 0:
        selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected[["fake_site", "npv"]]
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[["fake_site", "npv"]]
    else:
        candidates = pd.DataFrame([])

for j in range(len(candidates)):
    temp_nb = pd.DataFrame([])
    for i in range(len(candidates)):
        next_best = (pd.DataFrame(candidates.iloc[i, :])).T
        noise_sites = selected.append(next_best, sort=True)
        noise_bins = pd.merge(init_ranking, noise_sites, on="fake_site")
        noise_bins_unique = noise_bins[noise_bins["bin_count"] == 1].copy()
        noise_bins_dups = noise_bins[noise_bins["bin_count"] > 1].copy()
        temp_noise_bins_dups = (
            noise_bins_dups.groupby("hex")["sig_mw"].sum().reset_index()
        )
        noise_bins_dups = noise_bins_dups.drop(["sig_mw"], axis=1)
        noise_bins_dups = pd.merge(noise_bins_dups, temp_noise_bins_dups, on="hex")
        noise_bins_unique["noise_new"] = noise_bins_unique["noise"]
        noise_bins_dups["noise_new"] = noise_new(
            noise_bins_dups["sig_mw"], noise_bins_dups["sig_db"],
        )
        noise_bins = noise_bins_unique.append(noise_bins_dups, sort=True)
        temp_noise_bins = noise_bins
        noise_bins = noise_bins[noise_bins.noise_new >= -7.0]

        noise_bins_agg = noise_bins.groupby("fake_site", as_index=False).agg(
            {"density_code": "mean", "Hour_GBs": "sum"}
        )
        noise_bins_agg["xnpv"] = noise_bins_agg.apply(
            lambda x: loop_npv(year_1, x["Hour_GBs"], x["density_code"]), axis=1
        ).astype(int)
        next_best["adj_npv"] = noise_bins_agg["xnpv"].min()
        next_best["sum_npv"] = noise_bins_agg["xnpv"].sum()
        next_best = next_best[["fake_site", "adj_npv", "sum_npv"]]
        if next_best["adj_npv"].values <= rev_min:
            bad_candidate = next_best["fake_site"].values
            full = 0
            break
        init_cand_bins = pd.merge(next_best, init_ranking, on="fake_site")
        init_cand_bins = init_cand_bins.nlargest(prox, "noise")
        init_cand_bins = init_cand_bins[["noise", "fake_site"]].mean().astype("int64")
        cand_bins = pd.merge(next_best, temp_noise_bins, on="fake_site")
        cand_bins = cand_bins.nlargest(prox, "noise_new")
        cand_bins = cand_bins[["noise_new"]].mean()
        if (abs(init_cand_bins["noise"] - cand_bins["noise_new"])) > sig_loss:
            bad_candidate = init_cand_bins["fake_site"].astype("int64")
            full = 0
            break
        temp_nb = temp_nb.append(next_best, sort=True)
        full = 1
    if full == 1:
        if len(candidates) == 0:
            break
        new_ranking = temp_nb
        new_ranking["rank"] = new_ranking["sum_npv"].rank(ascending=True)
        new_ranking = new_ranking.sort_values(by="rank").reset_index(drop="True")
        temp_selected = new_ranking[new_ranking["rank"] == 1]
        temp_selected = temp_selected[["fake_site"]]
        selected = selected.append(temp_selected, sort=True)
        new_ranking = new_ranking[["fake_site"]]
        candidates = pd.DataFrame(new_ranking.iloc[1:, :]).reset_index(drop=True)
    elif full == 0:
        if len(candidates) == 0:
            break
        candidates = candidates.to_numpy()
        candidates = np.delete(
            candidates, np.where(candidates == bad_candidate)[0], axis=0
        )
        candidates = pd.DataFrame(candidates).reset_index(drop=True)
        candidates = candidates.rename(columns={0: "fake_site"})

end_yr1 = timeit.default_timer()
print(end_yr1 - start_yr1)
print("Year_2", end=" ")

pole_du = pole_du[1:11, :]
pole_u = pole_u[1:11, :]
pole_s = pole_s[1:11, :]
pole_r = pole_r[1:11, :]

start_yr2 = timeit.default_timer()
sites_yr2 = sites_yr2[["fake_site"]]
sites_2 = pd.merge(calc_sites, sites_yr2, on="fake_site")
year_2 = 2

if len(sites_yr2) > 0:
    sites = sites_2
    sites = sites.groupby("fake_site", as_index=False).agg(
        {"density_code": "mean", "Hour_GBs": "sum"}
    )
    sites["npv"] = sites.apply(
        lambda x: loop_npv(year_2, x["Hour_GBs"], x["density_code"]), axis=1
    ).astype(int)
    ranking = sites.loc[sites["npv"] > 0].copy()
    ranking["rank"] = ranking["npv"].rank(ascending=False)
    ranking = ranking.sort_values(by="rank")
    temp_ranking = ranking[["fake_site"]]
    init_ranking_bins = pd.merge(temp_ranking, sites_2, on="fake_site")
    init_ranking_counts = init_ranking_bins["hex"].value_counts().reset_index()
    init_ranking_counts = init_ranking_counts.rename(
        columns={"index": "hex", "hex": "bin_count"}
    )
    init_ranking = pd.merge(init_ranking_bins, init_ranking_counts, on="hex")
    if len(selected) > 0 and len(ranking) > 0:
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[["fake_site", "npv"]]
        selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected.append(selected, sort=True)
        selected = selected[["fake_site", "npv"]]
    elif len(selected) == 0 and len(ranking) > 0:
        selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected[["fake_site", "npv"]]
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[["fake_site", "npv"]]
    else:
        candidates = pd.DataFrame([])

for j in range(len(candidates)):
    temp_nb = pd.DataFrame([])
    for i in range(len(candidates)):
        next_best = (pd.DataFrame(candidates.iloc[i, :])).T
        noise_sites = selected.append(next_best, sort=True)
        noise_bins = pd.merge(init_ranking, noise_sites, on="fake_site")
        noise_bins_unique = noise_bins[noise_bins["bin_count"] == 1].copy()
        noise_bins_dups = noise_bins[noise_bins["bin_count"] > 1].copy()
        temp_noise_bins_dups = (
            noise_bins_dups.groupby("hex")["sig_mw"].sum().reset_index()
        )
        noise_bins_dups = noise_bins_dups.drop(["sig_mw"], axis=1)
        noise_bins_dups = pd.merge(noise_bins_dups, temp_noise_bins_dups, on="hex")
        noise_bins_unique["noise_new"] = noise_bins_unique["noise"]
        noise_bins_dups["noise_new"] = noise_new(
            noise_bins_dups["sig_mw"], noise_bins_dups["sig_db"],
        )
        noise_bins = noise_bins_unique.append(noise_bins_dups, sort=True)
        temp_noise_bins = noise_bins
        noise_bins = noise_bins[noise_bins.noise_new >= -7.0]
        noise_bins_agg = noise_bins.groupby("fake_site", as_index=False).agg(
            {"density_code": "mean", "Hour_GBs": "sum"}
        )
        noise_bins_agg["xnpv"] = noise_bins_agg.apply(
            lambda x: loop_npv(year_2, x["Hour_GBs"], x["density_code"]), axis=1
        ).astype(int)
        next_best["adj_npv"] = noise_bins_agg["xnpv"].min()
        next_best["sum_npv"] = noise_bins_agg["xnpv"].sum()
        next_best = next_best[["fake_site", "adj_npv", "sum_npv"]]
        if next_best["adj_npv"].values <= rev_min:
            bad_candidate = next_best["fake_site"].values
            full = 0
            break
        init_cand_bins = pd.merge(next_best, init_ranking, on="fake_site")
        init_cand_bins = init_cand_bins.nlargest(prox, "noise")
        init_cand_bins = init_cand_bins[["noise", "fake_site"]].mean().astype("int64")
        cand_bins = pd.merge(next_best, temp_noise_bins, on="fake_site")
        cand_bins = cand_bins.nlargest(prox, "noise_new")
        cand_bins = cand_bins[["noise_new"]].mean()
        if (abs(init_cand_bins["noise"] - cand_bins["noise_new"])) > sig_loss:
            bad_candidate = init_cand_bins["fake_site"].astype("int64")
            full = 0
            break
        temp_nb = temp_nb.append(next_best, sort=True)
        full = 1
    if full == 1:
        if len(candidates) == 0:
            break
        new_ranking = temp_nb
        new_ranking["rank"] = new_ranking["sum_npv"].rank(ascending=True)
        new_ranking = new_ranking.sort_values(by="rank").reset_index(drop="True")
        temp_selected = new_ranking[new_ranking["rank"] == 1]
        temp_selected = temp_selected[["fake_site"]]
        selected = selected.append(temp_selected, sort=True)
        new_ranking = new_ranking[["fake_site"]]
        candidates = pd.DataFrame(new_ranking.iloc[1:, :]).reset_index(drop=True)
    elif full == 0:
        if len(candidates) == 0:
            break
        candidates = candidates.to_numpy()
        candidates = np.delete(
            candidates, np.where(candidates == bad_candidate)[0], axis=0
        )
        candidates = pd.DataFrame(candidates).reset_index(drop=True)
        candidates = candidates.rename(columns={0: "fake_site"})
    elif len(candidates) == 0:
        break

end_yr2 = timeit.default_timer()
print(end_yr2 - start_yr2)
print("Year_3", end=" ")

pole_du = pole_du[1:11, :]
pole_u = pole_u[1:11, :]
pole_s = pole_s[1:11, :]
pole_r = pole_r[1:11, :]

start_yr3 = timeit.default_timer()
sites_yr3 = sites_yr3[["fake_site"]]
sites_3 = pd.merge(calc_sites, sites_yr3, on="fake_site")
year_3 = 3

if len(sites_yr3) > 0:
    sites = sites_3
    sites = sites.groupby("fake_site", as_index=False).agg(
        {"density_code": "mean", "Hour_GBs": "sum"}
    )
    sites["npv"] = sites.apply(
        lambda x: loop_npv(year_3, x["Hour_GBs"], x["density_code"]), axis=1
    ).astype(int)
    ranking = sites.loc[sites["npv"] > 0].copy()
    ranking["rank"] = ranking["npv"].rank(ascending=False)
    ranking = ranking.sort_values(by="rank")
    temp_ranking = ranking[["fake_site"]]
    init_ranking_bins = pd.merge(temp_ranking, sites_3, on="fake_site")
    init_ranking_counts = init_ranking_bins["hex"].value_counts().reset_index()
    init_ranking_counts = init_ranking_counts.rename(
        columns={"index": "hex", "hex": "bin_count"}
    )
    init_ranking = pd.merge(init_ranking_bins, init_ranking_counts, on="hex")
    if len(selected) > 0 and len(ranking) > 0:
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[["fake_site", "npv"]]
        new_selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected.append(new_selected, sort=True)
        selected = selected[["fake_site", "npv"]]
    elif len(selected) == 0 and len(ranking) > 0:
        selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected[["fake_site", "npv"]]
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[["fake_site", "npv"]]
    else:
        candidates = pd.DataFrame([])

for j in range(len(candidates)):
    temp_nb = pd.DataFrame([])
    for i in range(len(candidates)):
        next_best = (pd.DataFrame(candidates.iloc[i, :])).T
        noise_sites = selected.append(next_best, sort=True)
        noise_bins = pd.merge(init_ranking, noise_sites, on="fake_site")

        noise_bins_unique = noise_bins[noise_bins["bin_count"] == 1].copy()
        noise_bins_dups = noise_bins[noise_bins["bin_count"] > 1].copy()
        temp_noise_bins_dups = (
            noise_bins_dups.groupby("hex")["sig_mw"].sum().reset_index()
        )
        noise_bins_dups = noise_bins_dups.drop(["sig_mw"], axis=1)
        noise_bins_dups = pd.merge(noise_bins_dups, temp_noise_bins_dups, on="hex")
        noise_bins_unique["noise_new"] = noise_bins_unique["noise"]
        noise_bins_dups["noise_new"] = noise_new(
            noise_bins_dups["sig_mw"], noise_bins_dups["sig_db"],
        )
        noise_bins = noise_bins_unique.append(noise_bins_dups, sort=True)
        temp_noise_bins = noise_bins
        noise_bins = noise_bins[noise_bins.noise_new >= -7.0]

        noise_bins_agg = noise_bins.groupby("fake_site", as_index=False).agg(
            {"density_code": "mean", "Hour_GBs": "sum"}
        )
        noise_bins_agg["xnpv"] = noise_bins_agg.apply(
            lambda x: loop_npv(year_3, x["Hour_GBs"], x["density_code"]), axis=1
        ).astype(int)
        next_best["adj_npv"] = noise_bins_agg["xnpv"].min()
        next_best["sum_npv"] = noise_bins_agg["xnpv"].sum()
        next_best = next_best[["fake_site", "adj_npv", "sum_npv"]]
        if next_best["adj_npv"].values <= rev_min:
            bad_candidate = next_best["fake_site"].values
            full = 0
            break
        init_cand_bins = pd.merge(next_best, init_ranking, on="fake_site")
        init_cand_bins = init_cand_bins.nlargest(prox, "noise")
        init_cand_bins = init_cand_bins[["noise", "fake_site"]].mean().astype("int64")
        cand_bins = pd.merge(next_best, temp_noise_bins, on="fake_site")
        cand_bins = cand_bins.nlargest(prox, "noise_new")
        cand_bins = cand_bins[["noise_new"]].mean()
        if (abs(init_cand_bins["noise"] - cand_bins["noise_new"])) > sig_loss:
            bad_candidate = init_cand_bins["fake_site"].astype("int64")
            full = 0
            break
        temp_nb = temp_nb.append(next_best, sort=True)
        full = 1
    if full == 1:
        if len(candidates) == 0:
            break
        new_ranking = temp_nb
        new_ranking["rank"] = new_ranking["sum_npv"].rank(ascending=True)
        new_ranking = new_ranking.sort_values(by="rank").reset_index(drop="True")
        temp_selected = new_ranking[new_ranking["rank"] == 1]
        temp_selected = temp_selected[["fake_site"]]
        selected = selected.append(temp_selected, sort=True)
        new_ranking = new_ranking[["fake_site"]]
        candidates = pd.DataFrame(new_ranking.iloc[1:, :]).reset_index(drop=True)
    elif full == 0:
        if len(candidates) == 0:
            break
        candidates = candidates.to_numpy()
        candidates = np.delete(
            candidates, np.where(candidates == bad_candidate)[0], axis=0
        )
        candidates = pd.DataFrame(candidates).reset_index(drop=True)
        candidates = candidates.rename(columns={0: "fake_site"})
    elif len(candidates) == 0:
        break

end_yr3 = timeit.default_timer()
print(end_yr3 - start_yr3)

print("Year_4", end=" ")
print(len(selected["fake_site"].unique()), end=" ")

pole_du = pole_du[1:11, :]
pole_u = pole_u[1:11, :]
pole_s = pole_s[1:11, :]
pole_r = pole_r[1:11, :]

start_yr4 = timeit.default_timer()
sites_yr4 = sites_yr4[["fake_site"]]
sites_4 = pd.merge(calc_sites, sites_yr4, on="fake_site")
year_4 = 4

if len(sites_yr4) > 0:
    sites = sites_4
    sites = sites.groupby("fake_site", as_index=False).agg(
        {"density_code": "mean", "Hour_GBs": "sum"}
    )
    sites["npv"] = sites.apply(
        lambda x: loop_npv(year_4, x["Hour_GBs"], x["density_code"]), axis=1
    ).astype(int)
    ranking = sites.loc[sites["npv"] > 0].copy()
    ranking["rank"] = ranking["npv"].rank(ascending=False)
    ranking = ranking.sort_values(by="rank")
    temp_ranking = ranking[["fake_site"]]
    init_ranking_bins = pd.merge(temp_ranking, sites_4, on="fake_site")
    init_ranking_counts = init_ranking_bins["hex"].value_counts().reset_index()
    init_ranking_counts = init_ranking_counts.rename(
        columns={"index": "hex", "hex": "bin_count"}
    )
    init_ranking = pd.merge(init_ranking_bins, init_ranking_counts, on="hex")
    if len(selected) > 0 and len(ranking) > 0:
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[["fake_site", "npv"]]
        new_selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected.append(new_selected, sort=True)
        selected = selected[["fake_site", "npv"]]
    elif len(selected) == 0 and len(ranking) > 0:
        selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected[["fake_site", "npv"]]
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[["fake_site", "npv"]]
    else:
        candidates = pd.DataFrame([])

for j in range(len(candidates)):
    temp_nb = pd.DataFrame([])
    for i in range(len(candidates)):
        next_best = (pd.DataFrame(candidates.iloc[i, :])).T
        noise_sites = selected.append(next_best, sort=True)
        noise_bins = pd.merge(init_ranking, noise_sites, on="fake_site")
        noise_bins_unique = noise_bins[noise_bins["bin_count"] == 1].copy()
        noise_bins_dups = noise_bins[noise_bins["bin_count"] > 1].copy()
        temp_noise_bins_dups = (
            noise_bins_dups.groupby("hex")["sig_mw"].sum().reset_index()
        )
        noise_bins_dups = noise_bins_dups.drop(["sig_mw"], axis=1)
        noise_bins_dups = pd.merge(noise_bins_dups, temp_noise_bins_dups, on="hex")
        noise_bins_unique["noise_new"] = noise_bins_unique["noise"]
        noise_bins_dups["noise_new"] = noise_new(
            noise_bins_dups["sig_mw"], noise_bins_dups["sig_db"],
        )
        noise_bins = noise_bins_unique.append(noise_bins_dups, sort=True)
        temp_noise_bins = noise_bins
        noise_bins = noise_bins[noise_bins.noise_new >= -7.0]
        noise_bins_agg = noise_bins.groupby("fake_site", as_index=False).agg(
            {"density_code": "mean", "Hour_GBs": "sum"}
        )
        noise_bins_agg["xnpv"] = noise_bins_agg.apply(
            lambda x: loop_npv(year_4, x["Hour_GBs"], x["density_code"]), axis=1
        ).astype(int)
        next_best["adj_npv"] = noise_bins_agg["xnpv"].min()
        next_best["sum_npv"] = noise_bins_agg["xnpv"].sum()
        next_best = next_best[["fake_site", "adj_npv", "sum_npv"]]
        if next_best["adj_npv"].values <= rev_min:
            bad_candidate = next_best["fake_site"].values
            full = 0
            break
        init_cand_bins = pd.merge(next_best, init_ranking, on="fake_site")
        init_cand_bins = init_cand_bins.nlargest(prox, "noise")
        init_cand_bins = init_cand_bins[["noise", "fake_site"]].mean().astype("int64")
        cand_bins = pd.merge(next_best, temp_noise_bins, on="fake_site")
        cand_bins = cand_bins.nlargest(prox, "noise_new")
        cand_bins = cand_bins[["noise_new"]].mean()
        if (abs(init_cand_bins["noise"] - cand_bins["noise_new"])) > sig_loss:
            bad_candidate = init_cand_bins["fake_site"].astype("int64")
            full = 0
            break
        temp_nb = temp_nb.append(next_best, sort=True)
        full = 1
    if full == 1:
        if len(candidates) == 0:
            break
        new_ranking = temp_nb
        new_ranking["rank"] = new_ranking["sum_npv"].rank(ascending=True)
        new_ranking = new_ranking.sort_values(by="rank").reset_index(drop="True")
        temp_selected = new_ranking[new_ranking["rank"] == 1]
        temp_selected = temp_selected[["fake_site"]]
        selected = selected.append(temp_selected, sort=True)
        new_ranking = new_ranking[["fake_site"]]
        candidates = pd.DataFrame(new_ranking.iloc[1:, :]).reset_index(drop=True)
    elif full == 0:
        if len(candidates) == 0:
            break
        candidates = candidates.to_numpy()
        candidates = np.delete(
            candidates, np.where(candidates == bad_candidate)[0], axis=0
        )
        candidates = pd.DataFrame(candidates).reset_index(drop=True)
        candidates = candidates.rename(columns={0: "fake_site"})

end_yr4 = timeit.default_timer()
print(end_yr4 - start_yr4)
print("Year_5", end=" ")
print(len(selected["fake_site"].unique()), end=" ")

pole_du = pole_du[1:11, :]
pole_u = pole_u[1:11, :]
pole_s = pole_s[1:11, :]
pole_r = pole_r[1:11, :]

start_yr5 = timeit.default_timer()
sites_yr5 = sites_yr5[["fake_site"]]
sites_5 = pd.merge(calc_sites, sites_yr5, on="fake_site")
year_5 = 5

if len(sites_yr5) > 0:
    sites = sites_5
    sites = sites.groupby("fake_site", as_index=False).agg(
        {"density_code": "mean", "Hour_GBs": "sum"}
    )
    sites["npv"] = sites.apply(
        lambda x: loop_npv(year_5, x["Hour_GBs"], x["density_code"]), axis=1
    ).astype(int)
    ranking = sites.loc[sites["npv"] > 0].copy()
    ranking["rank"] = ranking["npv"].rank(ascending=False)
    ranking = ranking.sort_values(by="rank")
    temp_ranking = ranking[["fake_site"]]
    init_ranking_bins = pd.merge(temp_ranking, sites_5, on="fake_site")
    init_ranking_counts = init_ranking_bins["hex"].value_counts().reset_index()
    init_ranking_counts = init_ranking_counts.rename(
        columns={"index": "hex", "hex": "bin_count"}
    )
    init_ranking = pd.merge(init_ranking_bins, init_ranking_counts, on="hex")
    if len(selected) > 0 and len(ranking) > 0:
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[["fake_site", "npv"]]
        new_selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected.append(new_selected, sort=True)
        selected = selected[["fake_site", "npv"]]
    elif len(selected) == 0 and len(ranking) > 0:
        selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected[["fake_site", "npv"]]
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[["fake_site", "npv"]]
    else:
        candidates = pd.DataFrame([])

for j in range(len(candidates)):
    temp_nb = pd.DataFrame([])
    for i in range(len(candidates)):
        next_best = (pd.DataFrame(candidates.iloc[i, :])).T
        noise_sites = selected.append(next_best, sort=True)
        noise_bins = pd.merge(init_ranking, noise_sites, on="fake_site")
        noise_bins_unique = noise_bins[noise_bins["bin_count"] == 1].copy()
        noise_bins_dups = noise_bins[noise_bins["bin_count"] > 1].copy()
        temp_noise_bins_dups = (
            noise_bins_dups.groupby("hex")["sig_mw"].sum().reset_index()
        )
        noise_bins_dups = noise_bins_dups.drop(["sig_mw"], axis=1)
        noise_bins_dups = pd.merge(noise_bins_dups, temp_noise_bins_dups, on="hex")
        noise_bins_unique["noise_new"] = noise_bins_unique["noise"]
        noise_bins_dups["noise_new"] = noise_new(
            noise_bins_dups["sig_mw"], noise_bins_dups["sig_db"],
        )
        noise_bins = noise_bins_unique.append(noise_bins_dups, sort=True)
        temp_noise_bins = noise_bins
        noise_bins = noise_bins[noise_bins.noise_new >= -7.0]

        noise_bins_agg = noise_bins.groupby("fake_site", as_index=False).agg(
            {"density_code": "mean", "Hour_GBs": "sum"}
        )
        noise_bins_agg["xnpv"] = noise_bins_agg.apply(
            lambda x: loop_npv(year_5, x["Hour_GBs"], x["density_code"]), axis=1
        ).astype(int)
        next_best["adj_npv"] = noise_bins_agg["xnpv"].min()
        next_best["sum_npv"] = noise_bins_agg["xnpv"].sum()
        next_best = next_best[["fake_site", "adj_npv", "sum_npv"]]
        if next_best["adj_npv"].values <= rev_min:
            bad_candidate = next_best["fake_site"].values
            full = 0
            break
        init_cand_bins = pd.merge(next_best, init_ranking, on="fake_site")
        init_cand_bins = init_cand_bins.nlargest(prox, "noise")
        init_cand_bins = init_cand_bins[["noise", "fake_site"]].mean().astype("int64")
        cand_bins = pd.merge(next_best, temp_noise_bins, on="fake_site")
        cand_bins = cand_bins.nlargest(prox, "noise_new")
        cand_bins = cand_bins[["noise_new"]].mean()
        if (abs(init_cand_bins["noise"] - cand_bins["noise_new"])) > sig_loss:
            bad_candidate = init_cand_bins["fake_site"].astype("int64")
            full = 0
            break
        temp_nb = temp_nb.append(next_best, sort=True)
        full = 1
    if full == 1:
        if len(candidates) == 0:
            break
        new_ranking = temp_nb
        new_ranking["rank"] = new_ranking["sum_npv"].rank(ascending=True)
        new_ranking = new_ranking.sort_values(by="rank").reset_index(drop="True")
        temp_selected = new_ranking[new_ranking["rank"] == 1]
        temp_selected = temp_selected[["fake_site"]]
        selected = selected.append(temp_selected, sort=True)
        new_ranking = new_ranking[["fake_site"]]
        candidates = pd.DataFrame(new_ranking.iloc[1:, :]).reset_index(drop=True)
    elif full == 0:
        if len(candidates) == 0:
            break
        candidates = candidates.to_numpy()
        candidates = np.delete(
            candidates, np.where(candidates == bad_candidate)[0], axis=0
        )
        candidates = pd.DataFrame(candidates).reset_index(drop=True)
        candidates = candidates.rename(columns={0: "fake_site"})

end_yr5 = timeit.default_timer()
print(end_yr5 - start_yr5)
print("Year_6", end=" ")
print(len(selected["fake_site"].unique()), end=" ")

pole_du = pole_du[1:11, :]
pole_u = pole_u[1:11, :]
pole_s = pole_s[1:11, :]
pole_r = pole_r[1:11, :]

start_yr6 = timeit.default_timer()
sites_yr6 = sites_yr6[["fake_site"]]
sites_6 = pd.merge(calc_sites, sites_yr6, on="fake_site")
year_6 = 6

if len(sites_yr6) > 0:
    sites = sites_6
    sites = sites.groupby("fake_site", as_index=False).agg(
        {"density_code": "mean", "Hour_GBs": "sum"}
    )
    sites["npv"] = sites.apply(
        lambda x: loop_npv(year_6, x["Hour_GBs"], x["density_code"]), axis=1
    ).astype(int)
    ranking = sites.loc[sites["npv"] > 0].copy()
    ranking["rank"] = ranking["npv"].rank(ascending=False)
    ranking = ranking.sort_values(by="rank")
    temp_ranking = ranking[["fake_site"]]
    init_ranking_bins = pd.merge(temp_ranking, sites_6, on="fake_site")
    init_ranking_counts = init_ranking_bins["hex"].value_counts().reset_index()
    init_ranking_counts = init_ranking_counts.rename(
        columns={"index": "hex", "hex": "bin_count"}
    )
    init_ranking = pd.merge(init_ranking_bins, init_ranking_counts, on="hex")
    if len(selected) > 0 and len(ranking) > 0:
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[["fake_site", "npv"]]
        new_selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected.append(new_selected, sort=True)
        selected = selected[["fake_site", "npv"]]
    elif len(selected) == 0 and len(ranking) > 0:
        selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected[["fake_site", "npv"]]
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[["fake_site", "npv"]]
    else:
        candidates = pd.DataFrame([])

for j in range(len(candidates)):
    temp_nb = pd.DataFrame([])
    for i in range(len(candidates)):
        next_best = (pd.DataFrame(candidates.iloc[i, :])).T
        noise_sites = selected.append(next_best, sort=True)
        noise_bins = pd.merge(init_ranking, noise_sites, on="fake_site")
        noise_bins_unique = noise_bins[noise_bins["bin_count"] == 1].copy()
        noise_bins_dups = noise_bins[noise_bins["bin_count"] > 1].copy()
        temp_noise_bins_dups = (
            noise_bins_dups.groupby("hex")["sig_mw"].sum().reset_index()
        )
        noise_bins_dups = noise_bins_dups.drop(["sig_mw"], axis=1)
        noise_bins_dups = pd.merge(noise_bins_dups, temp_noise_bins_dups, on="hex")
        noise_bins_unique["noise_new"] = noise_bins_unique["noise"]
        noise_bins_dups["noise_new"] = noise_new(
            noise_bins_dups["sig_mw"], noise_bins_dups["sig_db"],
        )
        noise_bins = noise_bins_unique.append(noise_bins_dups, sort=True)
        temp_noise_bins = noise_bins
        noise_bins = noise_bins[noise_bins.noise_new >= -7.0]

        noise_bins_agg = noise_bins.groupby("fake_site", as_index=False).agg(
            {"density_code": "mean", "Hour_GBs": "sum"}
        )
        noise_bins_agg["xnpv"] = noise_bins_agg.apply(
            lambda x: loop_npv(year_6, x["Hour_GBs"], x["density_code"]), axis=1
        ).astype(int)
        next_best["adj_npv"] = noise_bins_agg["xnpv"].min()
        next_best["sum_npv"] = noise_bins_agg["xnpv"].sum()
        next_best = next_best[["fake_site", "adj_npv", "sum_npv"]]
        if next_best["adj_npv"].values <= rev_min:
            bad_candidate = next_best["fake_site"].values
            full = 0
            break
        init_cand_bins = pd.merge(next_best, init_ranking, on="fake_site")
        init_cand_bins = init_cand_bins.nlargest(prox, "noise")
        init_cand_bins = init_cand_bins[["noise", "fake_site"]].mean().astype("int64")
        cand_bins = pd.merge(next_best, temp_noise_bins, on="fake_site")
        cand_bins = cand_bins.nlargest(prox, "noise_new")
        cand_bins = cand_bins[["noise_new"]].mean()
        if (abs(init_cand_bins["noise"] - cand_bins["noise_new"])) > sig_loss:
            bad_candidate = init_cand_bins["fake_site"].astype("int64")
            full = 0
            break
        temp_nb = temp_nb.append(next_best, sort=True)
        full = 1
    if full == 1:
        if len(candidates) == 0:
            break
        new_ranking = temp_nb
        new_ranking["rank"] = new_ranking["sum_npv"].rank(ascending=True)
        new_ranking = new_ranking.sort_values(by="rank").reset_index(drop="True")
        temp_selected = new_ranking[new_ranking["rank"] == 1]
        temp_selected = temp_selected[["fake_site"]]
        selected = selected.append(temp_selected, sort=True)
        new_ranking = new_ranking[["fake_site"]]
        candidates = pd.DataFrame(new_ranking.iloc[1:, :]).reset_index(drop=True)
    elif full == 0:
        if len(candidates) == 0:
            break
        candidates = candidates.to_numpy()
        candidates = np.delete(
            candidates, np.where(candidates == bad_candidate)[0], axis=0
        )
        candidates = pd.DataFrame(candidates).reset_index(drop=True)
        candidates = candidates.rename(columns={0: "fake_site"})

end_yr6 = timeit.default_timer()
print(end_yr6 - start_yr6)
print("Year_7", end=" ")
print(len(selected["fake_site"].unique()), end=" ")

pole_du = pole_du[1:11, :]
pole_u = pole_u[1:11, :]
pole_s = pole_s[1:11, :]
pole_r = pole_r[1:11, :]

start_yr7 = timeit.default_timer()
sites_yr7 = sites_yr7[["fake_site"]]
sites_7 = pd.merge(calc_sites, sites_yr7, on="fake_site")
year_7 = 7

if len(sites_yr7) > 0:
    sites = sites_7
    sites = sites.groupby("fake_site", as_index=False).agg(
        {"density_code": "mean", "Hour_GBs": "sum"}
    )
    sites["npv"] = sites.apply(
        lambda x: loop_npv(year_7, x["Hour_GBs"], x["density_code"]), axis=1
    ).astype(int)
    ranking = sites.loc[sites["npv"] > 0].copy()
    ranking["rank"] = ranking["npv"].rank(ascending=False)
    ranking = ranking.sort_values(by="rank")
    temp_ranking = ranking[["fake_site"]]
    init_ranking_bins = pd.merge(temp_ranking, sites_7, on="fake_site")
    init_ranking_counts = init_ranking_bins["hex"].value_counts().reset_index()
    init_ranking_counts = init_ranking_counts.rename(
        columns={"index": "hex", "hex": "bin_count"}
    )
    init_ranking = pd.merge(init_ranking_bins, init_ranking_counts, on="hex")
    if len(selected) > 0 and len(ranking) > 0:
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[["fake_site", "npv"]]
        new_selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected.append(new_selected, sort=True)
        selected = selected[["fake_site", "npv"]]
    elif len(selected) == 0 and len(ranking) > 0:
        selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected[["fake_site", "npv"]]
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[["fake_site", "npv"]]
    else:
        candidates = pd.DataFrame([])

for j in range(len(candidates)):
    temp_nb = pd.DataFrame([])
    for i in range(len(candidates)):
        next_best = (pd.DataFrame(candidates.iloc[i, :])).T
        noise_sites = selected.append(next_best, sort=True)
        noise_bins = pd.merge(init_ranking, noise_sites, on="fake_site")
        noise_bins_unique = noise_bins[noise_bins["bin_count"] == 1].copy()
        noise_bins_dups = noise_bins[noise_bins["bin_count"] > 1].copy()
        temp_noise_bins_dups = (
            noise_bins_dups.groupby("hex")["sig_mw"].sum().reset_index()
        )
        noise_bins_dups = noise_bins_dups.drop(["sig_mw"], axis=1)
        noise_bins_dups = pd.merge(noise_bins_dups, temp_noise_bins_dups, on="hex")
        noise_bins_unique["noise_new"] = noise_bins_unique["noise"]
        noise_bins_dups["noise_new"] = noise_new(
            noise_bins_dups["sig_mw"], noise_bins_dups["sig_db"],
        )
        noise_bins = noise_bins_unique.append(noise_bins_dups, sort=True)
        temp_noise_bins = noise_bins
        noise_bins = noise_bins[noise_bins.noise_new >= -7.0]

        noise_bins_agg = noise_bins.groupby("fake_site", as_index=False).agg(
            {"density_code": "mean", "Hour_GBs": "sum"}
        )
        noise_bins_agg["xnpv"] = noise_bins_agg.apply(
            lambda x: loop_npv(year_7, x["Hour_GBs"], x["density_code"]), axis=1
        ).astype(int)
        next_best["adj_npv"] = noise_bins_agg["xnpv"].min()
        next_best["sum_npv"] = noise_bins_agg["xnpv"].sum()
        next_best = next_best[["fake_site", "adj_npv", "sum_npv"]]
        if next_best["adj_npv"].values <= rev_min:
            bad_candidate = next_best["fake_site"].values
            full = 0
            break
        init_cand_bins = pd.merge(next_best, init_ranking, on="fake_site")
        init_cand_bins = init_cand_bins.nlargest(prox, "noise")
        init_cand_bins = init_cand_bins[["noise", "fake_site"]].mean().astype("int64")
        cand_bins = pd.merge(next_best, temp_noise_bins, on="fake_site")
        cand_bins = cand_bins.nlargest(prox, "noise_new")
        cand_bins = cand_bins[["noise_new"]].mean()
        if (abs(init_cand_bins["noise"] - cand_bins["noise_new"])) > sig_loss:
            bad_candidate = init_cand_bins["fake_site"].astype("int64")
            full = 0
            break
        temp_nb = temp_nb.append(next_best, sort=True)
        full = 1
    if full == 1:
        if len(candidates) == 0:
            break
        new_ranking = temp_nb
        new_ranking["rank"] = new_ranking["sum_npv"].rank(ascending=True)
        new_ranking = new_ranking.sort_values(by="rank").reset_index(drop="True")
        temp_selected = new_ranking[new_ranking["rank"] == 1]
        temp_selected = temp_selected[["fake_site"]]
        selected = selected.append(temp_selected, sort=True)
        new_ranking = new_ranking[["fake_site"]]
        candidates = pd.DataFrame(new_ranking.iloc[1:, :]).reset_index(drop=True)
    elif full == 0:
        if len(candidates) == 0:
            break
        candidates = candidates.to_numpy()
        candidates = np.delete(
            candidates, np.where(candidates == bad_candidate)[0], axis=0
        )
        candidates = pd.DataFrame(candidates).reset_index(drop=True)
        candidates = candidates.rename(columns={0: "fake_site"})

end_yr7 = timeit.default_timer()
print(end_yr7 - start_yr7)
print("Year_8", end=" ")
print(len(selected["fake_site"].unique()), end=" ")

pole_du = pole_du[1:11, :]
pole_u = pole_u[1:11, :]
pole_s = pole_s[1:11, :]
pole_r = pole_r[1:11, :]

start_yr8 = timeit.default_timer()
sites_yr8 = sites_yr8[["fake_site"]]
sites_8 = pd.merge(calc_sites, sites_yr8, on="fake_site")
year_8 = 8

if len(sites_yr8) > 0:
    sites = sites_8
    sites = sites.groupby("fake_site", as_index=False).agg(
        {"density_code": "mean", "Hour_GBs": "sum"}
    )
    sites["npv"] = sites.apply(
        lambda x: loop_npv(year_8, x["Hour_GBs"], x["density_code"]), axis=1
    ).astype(int)
    ranking = sites.loc[sites["npv"] > 0].copy()
    ranking["rank"] = ranking["npv"].rank(ascending=False)
    ranking = ranking.sort_values(by="rank")
    temp_ranking = ranking[["fake_site"]]
    init_ranking_bins = pd.merge(temp_ranking, sites_8, on="fake_site")
    init_ranking_counts = init_ranking_bins["hex"].value_counts().reset_index()
    init_ranking_counts = init_ranking_counts.rename(
        columns={"index": "hex", "hex": "bin_count"}
    )
    init_ranking = pd.merge(init_ranking_bins, init_ranking_counts, on="hex")
    if len(selected) > 0 and len(ranking) > 0:
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[["fake_site", "npv"]]
        new_selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected.append(new_selected, sort=True)
        selected = selected[["fake_site", "npv"]]
    elif len(selected) == 0 and len(ranking) > 0:
        selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected[["fake_site", "npv"]]
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[["fake_site", "npv"]]
    else:
        candidates = pd.DataFrame([])

for j in range(len(candidates)):
    temp_nb = pd.DataFrame([])
    for i in range(len(candidates)):
        next_best = (pd.DataFrame(candidates.iloc[i, :])).T
        noise_sites = selected.append(next_best, sort=True)
        noise_bins = pd.merge(init_ranking, noise_sites, on="fake_site")
        noise_bins_unique = noise_bins[noise_bins["bin_count"] == 1].copy()
        noise_bins_dups = noise_bins[noise_bins["bin_count"] > 1].copy()
        temp_noise_bins_dups = (
            noise_bins_dups.groupby("hex")["sig_mw"].sum().reset_index()
        )
        noise_bins_dups = noise_bins_dups.drop(["sig_mw"], axis=1)
        noise_bins_dups = pd.merge(noise_bins_dups, temp_noise_bins_dups, on="hex")
        noise_bins_unique["noise_new"] = noise_bins_unique["noise"]
        noise_bins_dups["noise_new"] = noise_new(
            noise_bins_dups["sig_mw"], noise_bins_dups["sig_db"],
        )
        noise_bins = noise_bins_unique.append(noise_bins_dups, sort=True)
        temp_noise_bins = noise_bins
        noise_bins = noise_bins[noise_bins.noise_new >= -7.0]

        noise_bins_agg = noise_bins.groupby("fake_site", as_index=False).agg(
            {"density_code": "mean", "Hour_GBs": "sum"}
        )
        noise_bins_agg["xnpv"] = noise_bins_agg.apply(
            lambda x: loop_npv(year_8, x["Hour_GBs"], x["density_code"]), axis=1
        ).astype(int)
        next_best["adj_npv"] = noise_bins_agg["xnpv"].min()
        next_best["sum_npv"] = noise_bins_agg["xnpv"].sum()
        next_best = next_best[["fake_site", "adj_npv", "sum_npv"]]
        if next_best["adj_npv"].values <= rev_min:
            bad_candidate = next_best["fake_site"].values
            full = 0
            break
        init_cand_bins = pd.merge(next_best, init_ranking, on="fake_site")
        init_cand_bins = init_cand_bins.nlargest(prox, "noise")
        init_cand_bins = init_cand_bins[["noise", "fake_site"]].mean().astype("int64")
        cand_bins = pd.merge(next_best, temp_noise_bins, on="fake_site")
        cand_bins = cand_bins.nlargest(prox, "noise_new")
        cand_bins = cand_bins[["noise_new"]].mean()
        if (abs(init_cand_bins["noise"] - cand_bins["noise_new"])) > sig_loss:
            bad_candidate = init_cand_bins["fake_site"].astype("int64")
            full = 0
            break
        temp_nb = temp_nb.append(next_best, sort=True)
        full = 1
    if full == 1:
        if len(candidates) == 0:
            break
        new_ranking = temp_nb
        new_ranking["rank"] = new_ranking["sum_npv"].rank(ascending=True)
        new_ranking = new_ranking.sort_values(by="rank").reset_index(drop="True")
        temp_selected = new_ranking[new_ranking["rank"] == 1]
        temp_selected = temp_selected[["fake_site"]]
        selected = selected.append(temp_selected, sort=True)
        new_ranking = new_ranking[["fake_site"]]
        candidates = pd.DataFrame(new_ranking.iloc[1:, :]).reset_index(drop=True)
    elif full == 0:
        if len(candidates) == 0:
            break
        candidates = candidates.to_numpy()
        candidates = np.delete(
            candidates, np.where(candidates == bad_candidate)[0], axis=0
        )
        candidates = pd.DataFrame(candidates).reset_index(drop=True)
        candidates = candidates.rename(columns={0: "fake_site"})

end_yr8 = timeit.default_timer()
print(end_yr8 - start_yr8)
print("Year_9", end=" ")
print(len(selected["fake_site"].unique()), end=" ")

pole_du = pole_du[1:11, :]
pole_u = pole_u[1:11, :]
pole_s = pole_s[1:11, :]
pole_r = pole_r[1:11, :]

start_yr9 = timeit.default_timer()
sites_yr9 = sites_yr9[["fake_site"]]
sites_9 = pd.merge(calc_sites, sites_yr9, on="fake_site")
year_9 = 9

if len(sites_yr9) > 0:
    sites = sites_9
    sites = sites.groupby("fake_site", as_index=False).agg(
        {"density_code": "mean", "Hour_GBs": "sum"}
    )
    sites["npv"] = sites.apply(
        lambda x: loop_npv(year_9, x["Hour_GBs"], x["density_code"]), axis=1
    ).astype(int)
    ranking = sites.loc[sites["npv"] > 0].copy()
    ranking["rank"] = ranking["npv"].rank(ascending=False)
    ranking = ranking.sort_values(by="rank")
    temp_ranking = ranking[["fake_site"]]
    init_ranking_bins = pd.merge(temp_ranking, sites_9, on="fake_site")
    init_ranking_counts = init_ranking_bins["hex"].value_counts().reset_index()
    init_ranking_counts = init_ranking_counts.rename(
        columns={"index": "hex", "hex": "bin_count"}
    )
    init_ranking = pd.merge(init_ranking_bins, init_ranking_counts, on="hex")
    if len(selected) > 0 and len(ranking) > 0:
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[["fake_site", "npv"]]
        new_selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected.append(new_selected, sort=True)
        selected = selected[["fake_site", "npv"]]
    elif len(selected) == 0 and len(ranking) > 0:
        selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected[["fake_site", "npv"]]
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[["fake_site", "npv"]]
    else:
        candidates = pd.DataFrame([])

for j in range(len(candidates)):
    temp_nb = pd.DataFrame([])
    for i in range(len(candidates)):
        next_best = (pd.DataFrame(candidates.iloc[i, :])).T
        noise_sites = selected.append(next_best, sort=True)
        noise_bins = pd.merge(init_ranking, noise_sites, on="fake_site")
        noise_bins_unique = noise_bins[noise_bins["bin_count"] == 1].copy()
        noise_bins_dups = noise_bins[noise_bins["bin_count"] > 1].copy()
        temp_noise_bins_dups = (
            noise_bins_dups.groupby("hex")["sig_mw"].sum().reset_index()
        )
        noise_bins_dups = noise_bins_dups.drop(["sig_mw"], axis=1)
        noise_bins_dups = pd.merge(noise_bins_dups, temp_noise_bins_dups, on="hex")
        noise_bins_unique["noise_new"] = noise_bins_unique["noise"]
        noise_bins_dups["noise_new"] = noise_new(
            noise_bins_dups["sig_mw"], noise_bins_dups["sig_db"],
        )
        noise_bins = noise_bins_unique.append(noise_bins_dups, sort=True)
        temp_noise_bins = noise_bins
        noise_bins = noise_bins[noise_bins.noise_new >= -7.0]

        noise_bins_agg = noise_bins.groupby("fake_site", as_index=False).agg(
            {"density_code": "mean", "Hour_GBs": "sum"}
        )
        noise_bins_agg["xnpv"] = noise_bins_agg.apply(
            lambda x: loop_npv(year_9, x["Hour_GBs"], x["density_code"]), axis=1
        ).astype(int)
        next_best["adj_npv"] = noise_bins_agg["xnpv"].min()
        next_best["sum_npv"] = noise_bins_agg["xnpv"].sum()
        next_best = next_best[["fake_site", "adj_npv", "sum_npv"]]
        if next_best["adj_npv"].values <= rev_min:
            bad_candidate = next_best["fake_site"].values
            full = 0
            break
        init_cand_bins = pd.merge(next_best, init_ranking, on="fake_site")
        init_cand_bins = init_cand_bins.nlargest(prox, "noise")
        init_cand_bins = init_cand_bins[["noise", "fake_site"]].mean().astype("int64")
        cand_bins = pd.merge(next_best, temp_noise_bins, on="fake_site")
        cand_bins = cand_bins.nlargest(prox, "noise_new")
        cand_bins = cand_bins[["noise_new"]].mean()
        if (abs(init_cand_bins["noise"] - cand_bins["noise_new"])) > sig_loss:
            bad_candidate = init_cand_bins["fake_site"].astype("int64")
            full = 0
            break
        temp_nb = temp_nb.append(next_best, sort=True)
        full = 1
    if full == 1:
        if len(candidates) == 0:
            break
        new_ranking = temp_nb
        new_ranking["rank"] = new_ranking["sum_npv"].rank(ascending=True)
        new_ranking = new_ranking.sort_values(by="rank").reset_index(drop="True")
        temp_selected = new_ranking[new_ranking["rank"] == 1]
        temp_selected = temp_selected[["fake_site"]]
        selected = selected.append(temp_selected, sort=True)
        new_ranking = new_ranking[["fake_site"]]
        candidates = pd.DataFrame(new_ranking.iloc[1:, :]).reset_index(drop=True)
    elif full == 0:
        if len(candidates) == 0:
            break
        candidates = candidates.to_numpy()
        candidates = np.delete(
            candidates, np.where(candidates == bad_candidate)[0], axis=0
        )
        candidates = pd.DataFrame(candidates).reset_index(drop=True)
        candidates = candidates.rename(columns={0: "fake_site"})

end_yr9 = timeit.default_timer()
print(end_yr9 - start_yr9)
print("Year_10", end=" ")
print(len(selected["fake_site"].unique()), end=" ")

pole_du = pole_du[1:11, :]
pole_u = pole_u[1:11, :]
pole_s = pole_s[1:11, :]
pole_r = pole_r[1:11, :]

start_yr10 = timeit.default_timer()
sites_yr10 = sites_yr10[["fake_site"]]
sites_10 = pd.merge(calc_sites, sites_yr10, on="fake_site")
year_10 = 10

if len(sites_yr10) > 0:
    sites = sites_10
    sites = sites.groupby("fake_site", as_index=False).agg(
        {"density_code": "mean", "Hour_GBs": "sum"}
    )
    sites["npv"] = sites.apply(
        lambda x: loop_npv(year_10, x["Hour_GBs"], x["density_code"]), axis=1
    ).astype(int)
    ranking = sites.loc[sites["npv"] > 0].copy()
    ranking["rank"] = ranking["npv"].rank(ascending=False)
    ranking = ranking.sort_values(by="rank")
    temp_ranking = ranking[["fake_site"]]
    init_ranking_bins = pd.merge(temp_ranking, sites_10, on="fake_site")
    init_ranking_counts = init_ranking_bins["hex"].value_counts().reset_index()
    init_ranking_counts = init_ranking_counts.rename(
        columns={"index": "hex", "hex": "bin_count"}
    )
    init_ranking = pd.merge(init_ranking_bins, init_ranking_counts, on="hex")
    if len(selected) > 0 and len(ranking) > 0:
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[["fake_site", "npv"]]
        new_selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected.append(new_selected, sort=True)
        selected = selected[["fake_site", "npv"]]
    elif len(selected) == 0 and len(ranking) > 0:
        selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected[["fake_site", "npv"]]
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[["fake_site", "npv"]]
    else:
        candidates = pd.DataFrame([])

for j in range(len(candidates)):
    temp_nb = pd.DataFrame([])
    for i in range(len(candidates)):
        next_best = (pd.DataFrame(candidates.iloc[i, :])).T
        noise_sites = selected.append(next_best, sort=True)
        noise_bins = pd.merge(init_ranking, noise_sites, on="fake_site")
        noise_bins_unique = noise_bins[noise_bins["bin_count"] == 1].copy()
        noise_bins_dups = noise_bins[noise_bins["bin_count"] > 1].copy()
        temp_noise_bins_dups = (
            noise_bins_dups.groupby("hex")["sig_mw"].sum().reset_index()
        )
        noise_bins_dups = noise_bins_dups.drop(["sig_mw"], axis=1)
        noise_bins_dups = pd.merge(noise_bins_dups, temp_noise_bins_dups, on="hex")
        noise_bins_unique["noise_new"] = noise_bins_unique["noise"]
        noise_bins_dups["noise_new"] = noise_new(
            noise_bins_dups["sig_mw"], noise_bins_dups["sig_db"],
        )
        noise_bins = noise_bins_unique.append(noise_bins_dups, sort=True)
        temp_noise_bins = noise_bins
        noise_bins = noise_bins[noise_bins.noise_new >= -7.0]
        noise_bins_agg = noise_bins.groupby("fake_site", as_index=False).agg(
            {"density_code": "mean", "Hour_GBs": "sum"}
        )
        noise_bins_agg["xnpv"] = noise_bins_agg.apply(
            lambda x: loop_npv(year_10, x["Hour_GBs"], x["density_code"]), axis=1
        ).astype(int)
        next_best["adj_npv"] = noise_bins_agg["xnpv"].min()
        next_best["sum_npv"] = noise_bins_agg["xnpv"].sum()
        next_best = next_best[["fake_site", "adj_npv", "sum_npv"]]
        if next_best["adj_npv"].values <= rev_min:
            bad_candidate = next_best["fake_site"].values
            full = 0
            break
        init_cand_bins = pd.merge(next_best, init_ranking, on="fake_site")
        init_cand_bins = init_cand_bins.nlargest(prox, "noise")
        init_cand_bins = init_cand_bins[["noise", "fake_site"]].mean().astype("int64")
        cand_bins = pd.merge(next_best, temp_noise_bins, on="fake_site")
        cand_bins = cand_bins.nlargest(prox, "noise_new")
        cand_bins = cand_bins[["noise_new"]].mean()
        if (abs(init_cand_bins["noise"] - cand_bins["noise_new"])) > sig_loss:
            bad_candidate = init_cand_bins["fake_site"].astype("int64")
            full = 0
            break
        temp_nb = temp_nb.append(next_best, sort=True)
        full = 1
    if full == 1:
        if len(candidates) == 0:
            break
        new_ranking = temp_nb
        new_ranking["rank"] = new_ranking["sum_npv"].rank(ascending=True)
        new_ranking = new_ranking.sort_values(by="rank").reset_index(drop="True")
        temp_selected = new_ranking[new_ranking["rank"] == 1]
        temp_selected = temp_selected[["fake_site"]]
        selected = selected.append(temp_selected, sort=True)
        new_ranking = new_ranking[["fake_site"]]
        candidates = pd.DataFrame(new_ranking.iloc[1:, :]).reset_index(drop=True)
    elif full == 0:
        if len(candidates) == 0:
            break
        candidates = candidates.to_numpy()
        candidates = np.delete(
            candidates, np.where(candidates == bad_candidate)[0], axis=0
        )
        candidates = pd.DataFrame(candidates).reset_index(drop=True)
        candidates = candidates.rename(columns={0: "fake_site"})

end_yr10 = timeit.default_timer()
print(end_yr10 - start_yr10)

end_time = timeit.default_timer()

print("final_sites:", end=" ")
print(len(selected["fake_site"].unique()))

print("site_count:", end=" ")
print(site_number)

print("total_sec:", end=" ")
print(end_time - init_time)

print("total_min:", end=" ")
print((end_time - init_time) / 60)

print("sec/site:", end=" ")
print((end_time - init_time) / site_number)

selected.to_csv("selected.csv", index=False)

content = "test"
mail = smtplib.SMTP("smtp.gmail.com", 587)
mail.ehlo()
mail.starttls()
mail.login("notifications.norris@gmail.com", "Keynes92")
mail.sendmail("notifications.norris@gmail.com", "matthew@mdnorris.com", content)
mail.close()

# coverage run rf_training.py test.py
# coverage xml --include="rf_training.py" -o cobertura.xml
