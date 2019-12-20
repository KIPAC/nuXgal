import healpy as hp
import numpy as np
import matplotlib.pyplot as plt


map_original = hp.read_map('dataverse_files/545/2.5e+20_gp20/cib_fullmission.hpx.fits', verbose=False)
mask_bool = hp.read_map('dataverse_files/545/2.5e+20_gp20/mask_bool.hpx.fits', verbose=False, dtype=bool)
mask_apod = hp.read_map('dataverse_files/545/2.5e+20_gp20/mask_apod.hpx.fits', verbose=False, dtype=float)

map = map_original * mask_apod
map[np.where(np.isnan(map))] = hp.UNSEEN

r = hp.rotator.Rotator(coord=['G','C'])


mask_equatorial = r.rotate_map_alms(mask_bool)
hp.fitsfunc.write_map('mask_equatorial.fits', mask_equatorial, overwrite=True)



map_equatorial = r.rotate_map_alms(map)
hp.fitsfunc.write_map('map_equatorial.fits', map_equatorial, overwrite=True)



alm = hp.sphtfunc.map2alm(map_equatorial)
map_small = hp.sphtfunc.alm2map(nside=128, alms=alm)
hp.fitsfunc.write_map('map_equatorial_128.fits', map_small, overwrite=True)
