#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module allows MuSA to propagate information between cells.

Author: Esteban Alonso González - alonsoe@cesbio.cnes.fr
"""
import os
import time
import glob
import shutil
import numpy as np
import netCDF4 as nc
import datetime as dt
from scipy.spatial import distance
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.ndimage import rotate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# from sklearn.manifold import MDS
import statsmodels.stats.correlation_tools as ct
import modules.internal_fns as ifn
import modules.met_tools as met
import config as cfg
import constants as cnt
import modules.filters as flt
from modules.mds import landmark_MDS
from modules.internal_class import SnowEnsemble
if cfg.numerical_model == 'FSM2':
    import modules.fsm_tools as model
elif cfg.numerical_model == 'dIm':
    import modules.dIm_tools as model
elif cfg.numerical_model == 'snow17':
    import modules.snow17_tools as model
else:
    raise Exception('Model not implemented')


def GC(d, c):

    # d = np.abs(d, out=d) # necesary with negative distances
    # Pre-allocation
    rho = np.zeros(d.shape)  # dtype='float32')

    # Booleans selecting the three (two) categories of points.

    # Correlation for nearby points
    sel_points = d[d < c]
    if len(sel_points != 0):

        sel_points = sel_points / c
        rho[d < c] = (-1/4) * (sel_points)**5 + (1/2) * (sel_points)**4 + \
            (5/8) * (sel_points)**3 - (5/3) * (sel_points)**2 + 1

    # Correlation for mid-range points
    sel_points = d[np.logical_and(d >= c, d <= 2*c)]
    if len(sel_points != 0):

        sel_points = sel_points/c

        rho[np.logical_and(d >= c, d <= 2*c)] = (1/12) * (sel_points)**5 - \
            (1/2) * (sel_points)**4 + (5/8) * (sel_points)**3 + (5/3) * \
                    (sel_points)**2 - 5 * (sel_points) + 4 - (2/3) * \
                    (1/sel_points)

    # Correlation for far points (not necesary to run)
    # rho[d > 2*c] = 0

    # save memory
    # rho[rho < 0.01] = 0
    # sparse matrix
    # rho = spr.csc_matrix(rho)

    return rho


def fill_nan_arr(array):

    X = np.arange(0, array.shape[1])
    Y = np.arange(0, array.shape[0])

    x_grid, y_grid = np.meshgrid(X, Y)
    mask = ~np.isnan(array)
    x = x_grid[mask].reshape(-1)
    y = y_grid[mask].reshape(-1)

    points = np.array([x, y]).T
    values = array[mask].reshape(-1)

    interp_grid = griddata(points, values,
                           (x_grid, y_grid), method='nearest')

    return interp_grid


def regrid(data, out_x, out_y):
    m = max(data.shape[0], data.shape[1])
    y = np.linspace(0, 1.0/m, data.shape[0])
    x = np.linspace(0, 1.0/m, data.shape[1])
    interpolating_function = RegularGridInterpolator(
        (y, x), data.astype('float64'))

    yv, xv = np.meshgrid(np.linspace(0, 1.0/m, out_y),
                         np.linspace(0, 1.0/m, out_x))

    return interpolating_function((xv, yv))


def Sx(dem_arr):
    Sx_dmax = cfg.Sx_dmax
    DEM_res = cfg.DEM_res
    dmax_grid = int(Sx_dmax/DEM_res)

    rotated_arr = rotate(dem_arr, angle=cfg.Sx_angle-90, cval=np.nan)
    wind_field = rotated_arr.copy()
    wind_field[:] = np.nan
    for (x, y), Z in np.ndenumerate(rotated_arr):

        zwalk = rotated_arr[x, y+1:y + dmax_grid]
        if (len(zwalk) == 0):
            continue
        distance = (np.arange(y+1, y + len(zwalk)+1) - y)*DEM_res
        upward_angle = ((zwalk-Z) / distance)
        winstral = np.max(np.tan(upward_angle))

        wind_field[x, y] = winstral

    wind_field[np.isnan(wind_field)] = -1

    Sx = rotate(wind_field, angle=-(cfg.Sx_angle-90), mode='nearest')

    ids = np.where(Sx > 0)
    Sx = Sx[ids[0].min():ids[0].max(), ids[1].min():ids[1].max()]
    Sx = regrid(Sx, dem_arr.shape[0], dem_arr.shape[1])

    return Sx


def get_topo_arr():

    dem = cfg.dem_path
    topographic_features = cfg.topographic_features
    nc_dem_varname = cfg.nc_dem_varname

    dem = nc.Dataset(dem)

    dem_arr = dem.variables[nc_dem_varname][:]
    dem_arr = fill_nan_arr(dem_arr)

    # Grid spacing
    dels = 1

    x = np.arange(0, dem_arr.shape[1], dels, dtype='float32')
    y = np.arange(0, dem_arr.shape[0], dels, dtype='float32')
    X = np.tile(x, (len(y), 1)) * cfg.DEM_res
    Y = np.tile(y, (len(x), 1)).T * cfg.DEM_res

    # Slope
    px, py = np.gradient(dem_arr, 1)
    slope = np.sqrt(px ** 2 + py ** 2)
    slope = np.degrees(np.arctan(slope))

    # aspect
    aspect = 57.29578 * np.arctan2(py, - px)

    # Diurnal Anisotropic Heat
    amax = 202.500
    DAH = np.cos((amax - aspect) * np.pi / 180) *\
        np.arctan(slope * np.pi / 180)

    # TPI (vectorized sliding window)
    side = int(cfg.TPI_size//cfg.DEM_res)
    if side % 2:
        pass
    else:
        side = side + 1

    shape = (side, side)

    v = np.lib.stride_tricks.sliding_window_view(dem_arr, shape)
    v = np.mean(v, (2, 3))

    v = np.pad(v, ((dem_arr.shape[0] - v.shape[0])//2,
                   (dem_arr.shape[1] - v.shape[1])//2), 'constant',
               constant_values=np.nan)
    TPI = dem_arr - v
    TPI = fill_nan_arr(TPI)

    # winstral wind parameter
    winstral_sx = Sx(dem_arr)

    topo_dic = {
        'Ys': Y,
        'Xs': X,
        'Zs': dem_arr-np.min(dem_arr),
        'slope': slope,
        'DAH': DAH,
        'TPI': TPI,
        'Sx': winstral_sx}

    # remove useless features
    for topo_it, v in list(topographic_features.items()):
        if not v:
            del topo_dic[topo_it]

    return topo_dic


def calculate_distances():

    mask = cfg.nc_maks_path

    if cfg.topo_dict_external:
        topo_dic = ifn.io_read(cfg.topo_dict_external)
    else:
        topo_dic = get_topo_arr()

    topo_dic = {k: v.flatten() for k, v in topo_dic.items()}
    coords = np.array(list(topo_dic.values())).T

    if mask:  # remove cells out of mask
        mask = nc.Dataset(mask)
        mask_value = mask.variables['mask'][:].flatten()
        mask.close()
        coords = coords[mask_value == 1, :]

    orderows = np.arange(0, coords.shape[0])  # tentaive order of rows

    if cfg.dimension_reduction == 'PCA':

        x_scaled = StandardScaler().fit_transform(coords)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(x_scaled)
        # Order to force PD
        orderows = np.lexsort((coords[:, 0], coords[:, 1]))

    elif cfg.dimension_reduction == 'LMDS':
        lands = np.random.choice(range(0, coords.shape[0], 1),
                                 int(coords.shape[0]*0.1),
                                 replace=False)
        Dl2 = distance.cdist(coords[lands, :], coords, cfg.dist_algo)
        coords = landmark_MDS(Dl2, lands, 2)
        # Order to force PD
        orderows = np.lexsort((coords[:, 0], coords[:, 1]))

    else:
        pass

    if cfg.dimension_reduction:

        d = distance.cdist(coords[orderows, :],
                           coords[orderows, :],
                           'euclidean')
    else:
        d = distance.cdist(coords[orderows, :],
                           coords[orderows, :],
                           cfg.dist_algo)

    return d, orderows


def save_distance(d, orderows, prior_id):

    spatial_propagation_storage_path = cfg.spatial_propagation_storage_path
    mask = cfg.nc_maks_path

    name_dist = "dist_" + str(prior_id) + ".nc"
    name_dist = os.path.join(spatial_propagation_storage_path, name_dist)

    # recover distance matrix original/complete structure
    d = d[:, orderows][orderows]
    if mask:  # remove cells out of mask
        mask = nc.Dataset(mask)
        mask_value = mask.variables['mask'][:].flatten()
        mask.close()

        distmask = distance.cdist(mask_value.reshape(-1, 1),
                                  mask_value.reshape(-1, 1),
                                  "euclidean")

        distmask[~np.isnan(distmask)] = d.flatten()
        d = distmask

    f = nc.Dataset(name_dist, 'w', format='NETCDF4')

    f.createDimension('Y', d.shape[0])
    f.createDimension('X', d.shape[1])

    # float 64 is necesary, otherwise distance matrix is not PD
    distnc = f.createVariable('Dist', np.float64, ('Y', 'X'))

    distnc[:, :] = d

    f.close()

    # if (prior_id == 0):
    #     name_order = "order_row.blp"
    #     name_order = os.path.join(spatial_propagation_storage_path,
    #                  name_order)
    #     ifn.io_write(name_order, orderows)


def closePD(C):
    # TODO: Check Curriero method
    # TODO: Check other approaches https://stackoverflow.com/questions/10939213/how-can-i-calculate-the-nearest-positive-semi-definite-matrix
    # Another algo.

    if cfg.closePDmethod == 'clipped':  # faster
        return ct.cov_nearest(C, method="clipped")

    elif cfg.closePDmethod == 'nearest':
        return ct.cov_nearest(C, method="nearest")
    else:
        raise Exception("closePD method not implemented yet")


def GSC(mu, sigma, rho, orderows):

    N = cfg.ensemble_members
    mask = cfg.nc_maks_path

    # rho = get_rho()
    n = rho.shape[0]
    C = rho * sigma**2

    try:
        S = np.linalg.cholesky(C)
        # S = spcho.cholesky(C)

    except np.linalg.LinAlgError:
        if cfg.closePDmethod:
            print("rho is not positive-define, finding closer PD...")
            C = closePD(C)
            print("trying Cholesky decomposition again...")
            S = np.linalg.cholesky(C)
        # except spcho.CholmodNotPositiveDefiniteError:
        else:

            raise Exception('rho is not is not positive-definite.'
                            'Options:\n'
                            u'\u2022 try closePD = method\n'
                            u'\u2022 Change cut-off distance for the'
                            ' Gaspari and Cohn function\n'
                            u'\u2022 Change topographical setup\n'
                            u'\u2022 Check dist_algo\n'
                            u'\u2022 Perform PCA, to reduce dimensions\n'
                            u'\u2022 If the distances are read from'
                            ' an external file, ensure its precision'
                            ' is at least float64\n')

    # S is a lower triangular matrix loosely cooresponding to a standard
    # deviation matrix.

    z = np.random.normal(size=[n, N])
    x = mu + S@z
    n_lats, n_lons = ifn.get_dims()

    # reorder
    oldorder = np.where(np.arange(0, n)[:, None] == orderows[None, :])[1]
    x = x[oldorder, :]

    if mask:  # recover original structure

        mask = nc.Dataset(mask)
        mask_value = mask.variables['mask'][:].flatten()
        mask.close()

        real_x = np.zeros([n_lats * n_lons, N])
        real_x[:] = np.nan
        id_realx = np.squeeze(np.argwhere(mask_value == 1))
        real_x[id_realx, :] = x
        x = real_x

    # tidy matrix
    x = x[:, :, np.newaxis]
    x = x.reshape((n_lats, n_lons, N), order='C')
    # x = x.reshape((n_lats, n_lons, N), order='F')

    return x


def get_rho(prior_id):

    c = cfg.c

    d, orderows = calculate_distances()
    save_distance(d, orderows, prior_id)

    if all(x == c[0] for x in c):  # avoid to compute rho more than
        rho_uni = GC(d, c[0])
        rho = [rho_uni for _ in range(len(c))]
    else:
        rho = [GC(d, c_par) for c_par in c]

    return rho, orderows


def create_corelated_nc(prior_id):

    vars_to_perturbate = cfg.vars_to_perturbate
    perturbation_strategy = cfg.perturbation_strategy
    spatial_propagation_storage_path = cfg.spatial_propagation_storage_path
    mean_errors = cnt.mean_errors
    sd_errors = cnt.sd_errors
    upper_bounds = cnt.upper_bounds
    lower_bounds = cnt.lower_bounds

    N = cfg.ensemble_members
    n_lats, n_lons = ifn.get_dims()

    lat = np.arange(0, n_lats, 1)
    lon = np.arange(0, n_lons, 1)

    memb = np.arange(0, N, 1)

    # Path to intermediate file
    tmp_file_name = os.path.join(spatial_propagation_storage_path,
                                 str(prior_id)+'_tmp_GSC.nc')

    file_name = os.path.join(spatial_propagation_storage_path,
                             (str(prior_id) + '_GSC.nc'))

    try:  # remove temporal file if exists
        os.remove(tmp_file_name)
    except OSError:
        pass

    try:  # remove file if exists
        os.remove(file_name)
    except OSError:
        pass

    f = nc.Dataset(tmp_file_name, 'w', format='NETCDF4')

    f.createDimension('lat', len(lat))
    f.createDimension('lon', len(lon))
    f.createDimension('N', len(memb))
    f.createDimension('time', None)

    latitude = f.createVariable('Latitude', 'f4', 'lat')
    longitude = f.createVariable('Longitude', 'f4', 'lon')

    levels = f.createVariable('Members', 'i4', 'N')
    time = f.createVariable('Time', 'i4', 'time')

    latitude[:] = lat
    longitude[:] = lon

    levels[:] = memb

    # Add todays time
    today = dt.datetime.today()
    time_num = today.toordinal()
    time[0] = time_num

    # Atributes
    # Add global attributes
    f.description = "Spatially correlated Gaussian prior from \
        Cholesky-based Gaussian Sampler"
    f.history = "Created " + today.strftime("%d/%m/%y")

    # Add local attributes to variable instances
    longitude.units = 'cells'
    latitude.units = 'cells'
    time.units = 'days since Jan 01, 0001'
    levels.units = 'members'

    rho, orderows = get_rho(prior_id)
    # add GSC maps
    for count, var in enumerate(vars_to_perturbate):

        mu = mean_errors[var]
        sigma = sd_errors[var]

        gsc_arr = GSC(mu, sigma, rho[count], orderows)

        if perturbation_strategy[count] == "lognormal":
            gsc_arr = np.exp(gsc_arr)

        elif perturbation_strategy[count] in ["logitnormal_mult",
                                              "logitnormal_adi"]:

            gsc_arr = met.gexpit(gsc_arr, lower_bounds[var], upper_bounds[var])

        gsc_var = f.createVariable(var, 'f4', ('time', 'lat', 'lon', 'N'))
        gsc_var[0, :, :, :] = gsc_arr
    f.close()

    # Rename file to allow other processes to find it
    os.rename(tmp_file_name, file_name)


def generate_prior_maps_onenode(ini_DA_window):

    for id_win in range(len(ini_DA_window)):
        create_corelated_nc(id_win)


def generate_prior_maps(GSC_filenames, ini_DA_window, pbs_task_id):

    spatial_propagation_storage_path = cfg.spatial_propagation_storage_path

    for id_win in range(len(ini_DA_window)):
        # generate GSC in some pbs tasks
        if pbs_task_id == id_win:
            create_corelated_nc(id_win)
            break  # just a GSC per taks
        else:
            continue

    files = [os.path.join(spatial_propagation_storage_path, x)
             for x in GSC_filenames]

    # then wait until finish
    while True:
        if all(list(map(os.path.isfile, files))):
            break
        else:
            time.sleep(5)


def read_parameter(GSC_filename, lat_idx, lon_idx, var_tmp, mbr):

    f = nc.Dataset(GSC_filename)
    parameter = f.variables[var_tmp][:, lat_idx,  lon_idx, mbr]
    f.close()
    return parameter


def domain_steps():

    date_ini = cfg.date_ini
    date_end = cfg.date_end
    season_ini_day = cfg.season_ini_day
    season_ini_month = cfg.season_ini_month

    date_ini = dt.datetime.strptime(date_ini, "%Y-%m-%d %H:%M")
    date_end = dt.datetime.strptime(date_end, "%Y-%m-%d %H:%M")

    del_t = np.asarray([date_ini + dt.timedelta(hours=x)
                        for x in range(int((date_end-date_ini).total_seconds()
                                           / 3600) + 1)])
    days = [date.day for date in del_t]
    months = [date.month for date in del_t]
    hours = [date.hour for date in del_t]

    season_ini_cuts = np.argwhere((np.asarray(days) == season_ini_day) &
                                  (np.asarray(months) == season_ini_month) &
                                  (np.asarray(hours) == 0))

    return season_ini_cuts


def prepare_forcing(lat_idx, lon_idx):

    dates_obs = ifn.get_dates_obs()
    observations = ifn.obs_array(dates_obs, lat_idx, lon_idx)
    time_dict = ifn.simulation_steps(observations, dates_obs)
    main_forcing = model.forcing_table(lat_idx, lon_idx)
    main_forcing = model.unit_conversion(main_forcing)

    return main_forcing, time_dict, observations


def generate_obs_mask(pbs_task_id):

    name_obs_mask = "obs_mask.blp"
    name_obs_mask = os.path.join(cfg.spatial_propagation_storage_path,
                                 name_obs_mask)

    if pbs_task_id == 0:
        obs_mask()
    else:
        while True:
            if os.path.isfile(name_obs_mask):
                break
            else:
                time.sleep(5)


def obs_mask():

    nc_obs_path = cfg.nc_obs_path
    mask = cfg.nc_maks_path
    obs_var_names = cfg.obs_var_names

    if len(obs_var_names) > 1:
        raise Exception('comprobar comportamiento con ams d euna variable')

    files = glob.glob(nc_obs_path + "*.nc")
    # TODO: let the user define the prefix of the observations
    files.sort()

    # obs_dates = ifn.get_dates_obs()

    # ini_seasons = domain_steps()

    array_obs = np.empty((len(obs_var_names),)+ifn.get_dims())
    array_obs[:] = np.nan

    for cont, obs_var in enumerate(obs_var_names):

        tmp_storage = []

        for i, ncfile in enumerate(files):

            data_temp = nc.Dataset(ncfile)
            nc_value = data_temp.variables[obs_var][:, :, :]
            nc_value = nc_value.filled(np.nan)
            tmp_storage.extend(nc_value)
            data_temp.close()

        tmp_storage = np.dstack(tmp_storage)
        tmp_storage = np.rollaxis(tmp_storage, -1)
        tmp_storage[~np.isnan(tmp_storage)] = 1
        tmp_storage = np.nansum(tmp_storage, axis=0)

    array_obs[cont, :, :] = tmp_storage
    array_obs = np.nansum(array_obs, axis=0)

    # mask
    if mask:
        mask = nc.Dataset(mask)
        mask_value = mask.variables['mask'][:]
        mask.close()
        array_obs[mask_value != 1] = 0

    name_obs_mask = "obs_mask.blp"
    name_obs_mask = os.path.join(cfg.spatial_propagation_storage_path,
                                 name_obs_mask)

    ifn.io_write(name_obs_mask, array_obs)

    return True


def get_idrow_from_cor(lat_idx, lon_idx):

    n_lats, n_lons = ifn.get_dims()

    idrow = n_lons*lat_idx + lon_idx

    return idrow


def read_distances(lat_idx, lon_idx):

    spatial_propagation_storage_path = cfg.spatial_propagation_storage_path
    c = cfg.c
    c = max(c)

    name_dist = "dist_0.nc"
    name_dist = os.path.join(spatial_propagation_storage_path, name_dist)

    idrow = get_idrow_from_cor(lat_idx, lon_idx)

    f = nc.Dataset(name_dist, format='NETCDF4')
    distances = f.variables["Dist"][idrow, :]
    f.close()

    distances[distances > c*c] = np.nan

    # obs mask array
    name_mask_file = "obs_mask.blp"
    name_mask_file = os.path.join(spatial_propagation_storage_path,
                                  name_mask_file)

    data_mask = ifn.io_read(name_mask_file)

    data_mask = data_mask.flatten()

    distances[data_mask < 1] = np.nan

    return distances


def create_neigb(lat_idx, lon_idx, step, j):

    distances = read_distances(lat_idx, lon_idx)

    n_lats, n_lons = ifn.get_dims()

    id_neigb = np.asarray(np.where(~np.isnan(distances)))

    lats_neig = np.floor(id_neigb / n_lons)
    lons_neig = id_neigb-(lats_neig*n_lons)

    neigb = np.concatenate((lats_neig, lons_neig), axis=0).T.astype(int)

    if j == 0:
        neigb = ["{step}pri_ensbl_{lat}_{lon}.pkl.blp".format(step=step,
                                                              lat=neigb[x, 0],
                                                              lon=neigb[x, 1])
                 for x in range(neigb.shape[0])]

        neigb = [os.path.join(cfg.save_ensemble_path, neigb[x])
                 for x in range(len(neigb))]

    else:
        neigb = ["{step}_{j}it_ensbl_{lat}_{lon}.pkl.blp".
                 format(step=step,
                        j=j-1,
                        lat=neigb[x, 0],
                        lon=neigb[x, 1])
                 for x in range(neigb.shape[0])]

        neigb = [os.path.join(cfg.save_ensemble_path, neigb[x])
                 for x in range(len(neigb))]

    # add current cell
    if j == 0:
        current_path = "{step}pri_ensbl_{lat}_{lon}.pkl.blp".format(
            step=step, lat=lat_idx, lon=lon_idx)

    else:
        current_path = "{step}_{j}it_ensbl_{lat}_{lon}.pkl.blp".\
            format(step=step,
                   j=j-1,
                   lat=lat_idx,
                   lon=lon_idx)

    current_path = os.path.join(cfg.save_ensemble_path, current_path)
    neigb.append(current_path)

    # remove files if they do not exist
    check = [i for i in neigb if os.path.isfile(i)]

    return check


def get_neig_info(lat_idx, lon_idx, step, j):

    files = create_neigb(lat_idx, lon_idx, step, j)
    if len(files) == 0:
        return None

    neig_obs = []
    neig_pred_obs = []
    neig_r_cov = []
    neig_lat = []
    neig_long = []

    var_to_assim = cfg.var_to_assim
    r_cov = cfg.r_cov

    for count, file in enumerate(files):

        ens_tmp = ifn.io_read(file)

        list_state = ens_tmp.state_membres

        predicted = flt.get_predicitons(list_state, var_to_assim)
        obs = ens_tmp.observations

        if np.isnan(obs).all():
            continue

        obs_masked, predicted, tmp_r_cov = \
            flt.tidy_obs_pred_rcov(predicted, obs, r_cov)

        if predicted.ndim == 1:  # F**k numpy
            predicted = predicted[np.newaxis, :]

        if tmp_r_cov.ndim == 0:  # F**k numpy
            tmp_r_cov = tmp_r_cov[np.newaxis]

        lat_idx_ng = ens_tmp.lat_idx
        lon_idx_ng = ens_tmp.lon_idx

        lat_idx_ng = np.ones(obs_masked.shape) * lat_idx_ng
        lon_idx_ng = np.ones(obs_masked.shape) * lon_idx_ng

        neig_obs.append(obs_masked)
        neig_pred_obs.append(predicted)
        neig_r_cov.append(tmp_r_cov)
        neig_lat.append(lat_idx_ng)
        neig_long.append(lon_idx_ng)

    if len(neig_obs) == 0:  # If no observations in the neig, do nothing
        return neig_obs, neig_pred_obs, neig_r_cov, neig_lat, neig_long

    else:
        neig_obs = np.concatenate(neig_obs, axis=0)
        neig_pred_obs = np.concatenate(neig_pred_obs, axis=0)
        neig_r_cov = np.concatenate(neig_r_cov, axis=0)
        neig_lat = np.concatenate(neig_lat, axis=0)
        neig_long = np.concatenate(neig_long, axis=0)

        return neig_obs, neig_pred_obs, neig_r_cov, neig_lat, neig_long


def generate_local_rho(curren_lat, current_lon, neig_lat, neig_long):

    c = cfg.c
    spatial_propagation_storage_path = cfg.spatial_propagation_storage_path

    name_dist = "dist_0.nc"
    name_dist = os.path.join(spatial_propagation_storage_path, name_dist)

    current_id = get_idrow_from_cor(curren_lat, current_lon)
    neig_ids = get_idrow_from_cor(neig_lat, neig_long)

    f = nc.Dataset(name_dist)
    d_curent_neig = f.variables["Dist"][neig_ids.flatten().astype(int),
                                        current_id]
    f.close()

    f = nc.Dataset(name_dist)
    d_neig_neig = f.variables["Dist"][neig_ids.flatten().astype(int),
                                      neig_ids.flatten().astype(int)]
    f.close()

    rho_par_predicted_obs = [GC(d_curent_neig, c_par) for c_par in c]
    rho_predicted_obs = GC(d_neig_neig, max(c))

    return rho_par_predicted_obs, rho_predicted_obs


def create_ensemble_cell(lat_idx, lon_idx, ini_DA_window, step, gsc_count):

    main_forcing, time_dict, observations = prepare_forcing(lat_idx, lon_idx)
    if ifn.forcing_check(main_forcing):
        print("NA's found in: " + str(lat_idx) + "," + str(lon_idx))
        return None

    if step == 0:

        Ensemble = SnowEnsemble(lat_idx, lon_idx, time_dict)
    else:
        # Open cell to create new ensemble
        name_ensemble = "{step}_{j}it_ensbl_{lat}_{lon}.pkl.blp".\
            format(step=step-1,
                   j=cfg.Kalman_iterations - 1,
                   lat=lat_idx,
                   lon=lon_idx)

        file = os.path.join(cfg.output_path, name_ensemble)

        Ensemble = ifn.io_read(file)

    # subset forcing and observations
    observations_sbst = observations[time_dict["Assimilaiton_steps"][step]:
                                     time_dict["Assimilaiton_steps"][step
                                                                     + 1]]

    forcing_sbst = main_forcing[time_dict["Assimilaiton_steps"][step]:
                                time_dict["Assimilaiton_steps"][step + 1]]\
        .copy()

    if time_dict["Assimilaiton_steps"][step] in ini_DA_window:
        GSC_filename = (str(gsc_count) + '_GSC.nc')

        Ensemble.create(forcing_sbst, observations_sbst, step,
                        readGSC=True, GSC_filename=GSC_filename)

    else:  # for filters
        raise Exception('Filters not implemented yet in spatial propagation')
        Ensemble.create(forcing_sbst, observations_sbst, step)

    # Save ensembles, update: I cant, if save space cell without neigb will
    # show cero values
    # Ensemble.save_space()
    name_ensemble = "{step}pri_ensbl_{lat}_{lon}.pkl.blp".format(step=step,
                                                                 lat=lat_idx,
                                                                 lon=lon_idx)

    name_ensemble = os.path.join(cfg.save_ensemble_path, name_ensemble)
    ifn.io_write(name_ensemble, Ensemble)


def wait_for_ensembles(step, pbs_task_id, j=None):

    grid = ifn.expand_grid()

    if j is None and step == 0:
        files = glob.glob(os.path.join(cfg.save_ensemble_path,
                          "{step}pri*.pkl.blp".format(step=step)))
    else:
        files = glob.glob(os.path.join(cfg.save_ensemble_path,
                          "{step}_{j}it*.pkl.blp".format(step=step, j=j)))

    # wait until finish
    while True:
        if len(grid) == len(files):
            # try to desyncrhonice to speedup IO
            time.sleep(1 + int(np.random.randint(0, 20, 1)))
            # free a bit of space, remove previous iteration from first task
            if j is not None:
                # try to remove prior files
                if pbs_task_id == 0:
                    rm_files = glob.glob(os.path.join(cfg.save_ensemble_path,
                                                      "*pri*.pkl.blp"))
                    for f in rm_files:
                        if os.path.isfile(f):
                            os.remove(f)

                # [or] Solution in case of no iterations. If not,
                # it will not clean
                if pbs_task_id == 0 and j > 0 or cfg.Kalman_iterations == 1:
                    rm_files = glob.glob(os.path.join(cfg.save_ensemble_path,
                                                      "{step}_{j}it*.pkl.blp".
                                                      format(step=step,
                                                             j=j-1)))
                    for f in rm_files:
                        if os.path.isfile(f):
                            os.remove(f)

                    # if end of the Kalman iterations move ensembles to results
                    if j == cfg.Kalman_iterations-1:
                        for f in files:
                            shutil.move(f, cfg.output_path)

                # if not first task and end of iterations, wait for
                # ensembles in results
                if pbs_task_id != 0 and j == cfg.Kalman_iterations-1:
                    files = glob.glob(os.path.join(cfg.output_path,
                                      "{step}_{j}it*.pkl.blp".format(step=step,
                                                                     j=j)))
                    while True:
                        if len(grid) == len(files):
                            break
                        else:
                            time.sleep(5)
                            files = glob.glob(
                                os.path.join(
                                    cfg.output_path,
                                    "{step}_{j}it*.pkl.blp".format(step=step,
                                                                   j=j)))

            break

        else:
            time.sleep(5)
            if j is None:
                files = glob.glob(os.path.join(cfg.save_ensemble_path,
                                  "{step}pri*.pkl.blp".format(step=step)))
            else:
                files = glob.glob(os.path.join(cfg.save_ensemble_path,
                                  "{step}_{j}it*.pkl.blp".format(step=step,
                                                                 j=j)))


def spatial_assim(lat_idx, lon_idx, step, j):

    vars_to_perturbate = cfg.vars_to_perturbate

    # Open cell to assim
    if j == 0:
        file = "{step}pri_ensbl_{lat}_{lon}.pkl.blp".format(step=step,
                                                            lat=lat_idx,
                                                            lon=lon_idx)
    else:
        file = "{step}_{j}it_ensbl_{lat}_{lon}.pkl.blp".format(step=step,
                                                               j=j-1,
                                                               lat=lat_idx,
                                                               lon=lon_idx)

    file = os.path.join(cfg.save_ensemble_path, file)

    try:  # If current cell do not exist return None

        Ensemble = ifn.io_read(file)

    except FileNotFoundError:
        print('Not found: ' + file)

        return None

    # try to update with neig inf, if any problem just copy prior

    try:
        save_space_flag = True
        neig_obs, neig_pred_obs, neig_r_cov, neig_lat, neig_long = \
            get_neig_info(lat_idx, lon_idx, step, j)

        # if no obs do nothing
        if len(neig_obs) == 0:
            Ensemble.kalman_update(create=False)

        else:
            # create neig rho
            rho_par_predicted_obs, rho_predicted_obs = \
                generate_local_rho(lat_idx, lon_idx, neig_lat, neig_long)

            prior = np.ones((len(vars_to_perturbate), Ensemble.members))
            for cont, var in enumerate(vars_to_perturbate):
                if j == 0:
                    var_tmp = [Ensemble.noise[x][var]
                               for x in range(Ensemble.members)]
                else:
                    var_tmp = [Ensemble.noise_kalman[x][var]
                               for x in range(Ensemble.members)]
                var_tmp = np.asarray(var_tmp)
                var_tmp = np.squeeze(var_tmp)
                # HACK: next lines have to be modified with time varying
                # perturbations
                # var_tmp = np.squeeze(var_tmp[:, mask])
                prior[cont, :] = var_tmp[:, 0]

            # translate lognormal variables to normal distribution
            prior = flt.transform_space(prior, 'to_normal')

            updated_pars = flt.ens_klm(prior, neig_obs, neig_pred_obs,
                                       cfg.Kalman_iterations, neig_r_cov,
                                       rho_AB=rho_par_predicted_obs,
                                       rho_BB=rho_predicted_obs,
                                       stochastic=False)

            updated_pars = flt.transform_space(updated_pars, 'from_normal')

            Ensemble.kalman_update(step, updated_pars,
                                   create=True, iteration=j)

    except Exception as ex:  # if any error, dont update
        # TODO: Fix this except. guess al possible errors
        print('({ex}) Cel not updated: {lat}:lat_idx, {lon}:lon_idx'.
              format(lat=lat_idx, lon=lon_idx, ex=ex))

        save_space_flag = False
        Ensemble.kalman_update(create=False)
        return None

    # Save updated ensemble
    if j < cfg.Kalman_iterations-1 and save_space_flag:
        Ensemble.save_space()

    name_ensemble = "{step}_{j}it_ensbl_{lat}_{lon}.pkl.blp".format(
        step=step, j=j, lat=lat_idx, lon=lon_idx)

    name_ensemble = os.path.join(cfg.save_ensemble_path, name_ensemble)
    ifn.io_write(name_ensemble, Ensemble)

    return None


def collect_results(lat_idx, lon_idx):

    date_ini = cfg.date_ini
    date_end = cfg.date_end

    # remove prior files
#    rm_files = glob.glob(os.path.join(cfg.save_ensemble_path, "*pri*.blp"))
#    for f in rm_files:
#        if os.path.isfile(f):
#            os.remove(f)

    # create dates:
    date_ini = dt.datetime.strptime(date_ini, "%Y-%m-%d %H:%M")
    date_end = dt.datetime.strptime(date_end, "%Y-%m-%d %H:%M")
    del_t = [date_ini + dt.timedelta(hours=n)
             for n in range(int((date_end - date_ini).days * 24 + 24))]

    # loop over files to retrieve results
    ini_DA_window = domain_steps()

    # create filenames
    DA_Results = ifn.init_result(del_t, DA=True)
    updated_FSM = ifn.init_result(del_t)
    sd_FSM = ifn.init_result(del_t)
    OL_FSM = ifn.init_result(del_t)

    # HACK: fake time_dict
    time_dict = {'Assimilaiton_steps':
                 np.append(ini_DA_window, len(del_t))}
    # loop over DA steps
    for step in range(len(ini_DA_window)):

        # create ensemble name
        fname = os.path.join(
            cfg.output_path,
            "{step}_{j}it_ensbl_{lat_idx}_{lon_idx}.pkl.blp".format(
                step=step,
                j=cfg.Kalman_iterations - 1,
                lat_idx=lat_idx,
                lon_idx=lon_idx))

        # Open file
        try:
            Ensemble = ifn.io_read(fname)
        except FileNotFoundError:
            continue

        # Rm de ensemble file
        if os.path.isfile(fname):
            os.remove(fname)

        step_results = {}
        # extract psoterior parameters
        for cont, var_p in enumerate(cfg.vars_to_perturbate):

            # Get perturbation parameters"""
            noise_ens_temp = [Ensemble.noise[x][var_p]
                              for x in range(len(Ensemble.noise))]
            noise_ens_temp = np.vstack(noise_ens_temp)

            noise_tmp_avg = np.average(noise_ens_temp, axis=0,
                                       weights=Ensemble.wgth)
            noise_tmp_sd = flt.weighted_std(noise_ens_temp, axis=0,
                                            weights=Ensemble.wgth)

            step_results[var_p + "_noise_mean"] = noise_tmp_avg
            step_results[var_p + "_noise_sd"] = noise_tmp_sd

        model.storeDA(DA_Results, step_results, Ensemble.observations,
                      time_dict, step)

        model.store_updatedsim(updated_FSM, sd_FSM, Ensemble,
                               time_dict, step)

    # the whole OL is stored in the last Ensemble
    try:
        model.storeOL(OL_FSM, Ensemble, Ensemble.observations,
                      time_dict, step)
    except NameError:
        pass

    # Write results
    cell_data = {"DA_Results": DA_Results,
                 "OL_FSM": OL_FSM,
                 "updated_FSM": updated_FSM,
                 "sd_FSM": sd_FSM}

    filename = ("cell_" + str(lat_idx) + "_" + str(lon_idx) + ".pkl.blp")
    filename = os.path.join(cfg.output_path, filename)

    ifn.io_write(filename, cell_data)
