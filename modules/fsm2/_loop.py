"""Fused time loop: run all timesteps without returning to Python."""

from numba import njit

from ._timestep import fsm2_timestep


@njit(cache=False, fastmath=True)
def fsm2_run(params, dt, zT, zU, trans,
              elev, LW, Ps, Qa, Rf, Sdif, Sdir, Sf, Ta, Ua,
              alb0, vegh, VAI,
              albs, Tsrf, Dsnw, Nsnow, Qcan, Rgrn, Sice, Sliq, Sveg,
              Tcan, Tsnow, Tsoil, Tveg, Vsmc,
              Nsnow_out, Tsrf_out, Tsoil_out, Tveg_out, Wflx_out,
              diag_fsnow, diag_albs, diag_H, diag_LE, diag_LWout,
              diag_LWsub, diag_Melt, diag_Roff, diag_snd, diag_snw,
              diag_subl, diag_svg, diag_SWout, diag_SWsub, diag_Tsub,
              diag_Usub,
              guardar_estado,
              perf_Dsnw, perf_Rgrn, perf_Sice, perf_Sliq, perf_Tsnow,
              perf_Qcan, perf_Sveg, perf_Tcan, perf_Vsmc):
    """Run every timestep and point, updating state and output arrays."""
    nt = elev.shape[0]
    npuntos = albs.shape[0]

    for it in range(nt):
        elev_t = elev[it]
        for ip in range(npuntos):
            (Nsnow_i, Tsrf_i, albs_i, fsnow_i, H_i, LE_i, LWout_i,
             LWsub_i, Melt_i, Roff_i, snd_i, snw_i, subl_i, svg_i,
             SWout_i, SWsub_i, Tsub_i, Usub_i, Wflx_i) = fsm2_timestep(
                params, dt, elev_t, zT, zU,
                LW[it, ip], Ps[it, ip], Qa[it, ip], Rf[it, ip],
                Sdif[it, ip], Sdir[it, ip], Sf[it, ip], Ta[it, ip],
                trans, Ua[it, ip], alb0[ip], vegh[ip], VAI[ip],
                albs[ip], Tsrf[ip], Dsnw[ip], Nsnow[ip], Qcan[ip],
                Rgrn[ip], Sice[ip], Sliq[ip], Sveg[ip], Tcan[ip],
                Tsnow[ip], Tsoil[ip], Tveg[ip], Vsmc[ip]
            )

            Nsnow[ip] = Nsnow_i
            Tsrf[ip] = Tsrf_i
            albs[ip] = albs_i

            Nsnow_out[it, ip] = Nsnow_i
            Tsrf_out[it, ip] = Tsrf_i
            diag_fsnow[it, ip] = fsnow_i
            diag_albs[it, ip] = albs_i
            diag_H[it, ip] = H_i
            diag_LE[it, ip] = LE_i
            diag_LWout[it, ip] = LWout_i
            diag_LWsub[it, ip] = LWsub_i
            diag_Melt[it, ip] = Melt_i
            diag_Roff[it, ip] = Roff_i
            diag_snd[it, ip] = snd_i
            diag_snw[it, ip] = snw_i
            diag_subl[it, ip] = subl_i
            diag_svg[it, ip] = svg_i
            diag_SWout[it, ip] = SWout_i
            diag_SWsub[it, ip] = SWsub_i
            diag_Tsub[it, ip] = Tsub_i
            diag_Usub[it, ip] = Usub_i
            Wflx_out[it, ip, :] = Wflx_i
            Tsoil_out[it, ip, :] = Tsoil[ip]
            Tveg_out[it, ip, :] = Tveg[ip]

            if guardar_estado:
                perf_Dsnw[it, ip, :] = Dsnw[ip]
                perf_Rgrn[it, ip, :] = Rgrn[ip]
                perf_Sice[it, ip, :] = Sice[ip]
                perf_Sliq[it, ip, :] = Sliq[ip]
                perf_Tsnow[it, ip, :] = Tsnow[ip]
                perf_Qcan[it, ip, :] = Qcan[ip]
                perf_Sveg[it, ip, :] = Sveg[ip]
                perf_Tcan[it, ip, :] = Tcan[ip]
                perf_Vsmc[it, ip, :] = Vsmc[ip]
