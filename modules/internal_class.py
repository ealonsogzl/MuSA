#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Here The main classes of MuSA are stored.

Author: Esteban Alonso González - e.alonsogzl@gmail.com
"""
import config as cfg
import numpy as np
import pandas as pd
import modules.fsm_tools as fsm
import modules.met_tools as met


class GeneralConfig:
    # TODO: Create a class to store all the information provided by the user.
    # TODO: Create GeneralConfig.check() to check the setup make sense
    def __init__(self):

        # Domain
        self.implementation = cfg.implementation

        # Directories
        self.nc_path = cfg.nc_path

        # Time variables
        self.dates = cfg.dates_obs

        # Varnames
        self.snowdepth = cfg.var_name_snowdepth
        self.latitude = cfg.lat_name_var
        self.longitude = cfg.lon_name_var

        # FSM information
        self.fsm_src = cfg.fs_src_path       # Directory of the FSM source code
        self.tmp_path = cfg.tmp_path         # Temporal path to compile FSM


class SnowEnsemble:
    """
    Main class containing the ensemble of simulations
    (rows are timesteps)
    """

    def __init__(self, temp_dest):
        self.members = cfg.ensemble_members
        self.temp_dest = temp_dest

        # Inicialice open loop storage lists
        self.origin_state = pd.DataFrame()
        self.origin_dump = []

        # Inicialice lists of members
        self.state_membres = [0 for i in range(self.members)]
        self.out_members = [0 for i in range(self.members)]
        self.noise = [0 for i in range(self.members)]

        if cfg.filter_algorithm == "Kalman":
            self.noise_kalman = [0 for i in range(self.members)]
            self.out_members_kalman = [0 for i in range(self.members)]

        # Inicialice prior weights = 1
        self.wgth = np.ones(self.members)

        # Inicialice step value
        self.step = -1

    def create(self, forcing_sbst, step):

        self.step = step
        fsm.fsm_forcing_wrt(forcing_sbst, self.temp_dest)

        # Write init or dump file from previous run if step != 0
        if step == 0:
            fsm.write_init(self.temp_dest)
        else:
            fsm.write_dump(self.origin_dump[step - 1], self.temp_dest)

        # create open loop simulation
        fsm.fsm_run(self.temp_dest)

        # read FSM outputs
        origin_state_tmp, origin_dump_tmp =\
            fsm.fsm_read_output(self.temp_dest)

        # add fSCA (or SCA) if it is in var_to_assim
        if "fSCA" or "SCA" in cfg.var_to_assim:
            origin_state_tmp["fSCA"] = met.fSCA(origin_state_tmp)

            if "SCA" in cfg.var_to_assim:
                origin_state_tmp["SCA"] = met.SCA(origin_state_tmp)

        # Store FSM outputs
        self.origin_state = pd.concat([self.origin_state,
                                       origin_state_tmp.copy()])
        self.origin_dump.append(origin_dump_tmp.copy())

        # Avoid ensemble generation if direct insertion
        if cfg.assimilation_strategy == "direct_insertion":
            return None

        if (cfg.redraw_prior and
            cfg.filter_algorithm == "PBS" and
            cfg.assimilation_strategy == "filtering" and
                step != 0):

            func_shape = met.get_shape_from_noise(self.noise, self.wgth)

        # Ensemble generator
        for mbr in range(self.members):

            if step == 0:
                member_forcing, noise_tmp = \
                    met.perturb_parameters(forcing_sbst)
            else:
                # if PBS/importance resampling is used, use the noise
                # of the previous assimilation step or redraw.
                if cfg.filter_algorithm == "PBS":
                    if (cfg.redraw_prior and
                        step != 0 and
                            cfg.assimilation_strategy == "filtering"):

                        # Create new perturbation parameters
                        noise_tmp = met.redraw(func_shape.copy())
                        member_forcing, noise_tmp = \
                            met.perturb_parameters(forcing_sbst,
                                                   noise_tmp, update=True)

                    else:
                        # Use the posterior parameters
                        noise_tmp = list(self.noise[mbr].values())
                        noise_tmp = np.vstack(noise_tmp)
                        # Take last perturbation values
                        noise_tmp = noise_tmp[:, np.shape(noise_tmp)[1] - 1]
                        member_forcing, noise_tmp = \
                            met.perturb_parameters(forcing_sbst,
                                                   noise_tmp, update=True)
                else:
                    # if kalman is used, use the posterior noise of the
                    # previous run
                    noise_tmp = list(self.noise_kalman[mbr].values())
                    noise_tmp = np.vstack(noise_tmp)
                    # Take last perturbation values
                    noise_tmp = noise_tmp[:, np.shape(noise_tmp)[1] - 1]
                    member_forcing, noise_tmp = \
                        met.perturb_parameters(forcing_sbst,
                                               noise_tmp, update=True)

            # writte perturbed forcing
            fsm.fsm_forcing_wrt(member_forcing, self.temp_dest)

            if step == 0:
                fsm.write_init(self.temp_dest)
            else:
                if cfg.filter_algorithm == "PBS":
                    fsm.write_dump(self.out_members[mbr], self.temp_dest)
                else:  # if kalman, write updated dump
                    fsm.write_dump(self.out_members_kalman[mbr],
                                   self.temp_dest)

            fsm.fsm_run(self.temp_dest)

            state_tmp, dump_tmp = fsm.fsm_read_output(self.temp_dest)

            if "fSCA" in cfg.var_to_assim:
                state_tmp["fSCA"] = met.fSCA(state_tmp)

                if "SCA" in cfg.var_to_assim:
                    state_tmp["SCA"] = met.SCA(state_tmp)

            # store FSM outputs and perturbation parameters
            self.state_membres[mbr] = state_tmp.copy()

            self.out_members[mbr] = dump_tmp.copy()

            self.noise[mbr] = noise_tmp.copy()

    def kalman_update(self, forcing_sbst=None, step=None, updated_pars=None,
                      create=None, iteration=None):

        if create:  # If there is observational data update the ensemble
            # Ensemble generator
            for mbr in range(self.members):

                noise_tmp = updated_pars[:, mbr]
                member_forcing, noise_k_tmp = \
                    met.perturb_parameters(forcing_sbst, noise_tmp,
                                           update=True)

                fsm.fsm_forcing_wrt(member_forcing, self.temp_dest)

                if step == 0:
                    fsm.write_init(self.temp_dest)
                else:
                    fsm.write_dump(self.out_members_kalman[mbr],
                                   self.temp_dest)

                fsm.fsm_run(self.temp_dest)

                state_tmp, dump_tmp = fsm.fsm_read_output(self.temp_dest)

                if "fSCA" in cfg.var_to_assim:
                    state_tmp["fSCA"] = met.fSCA(state_tmp)

                    if "SCA" in cfg.var_to_assim:
                        state_tmp["SCA"] = met.SCA(state_tmp)

                self.state_membres[mbr] = state_tmp.copy()

                self.noise_kalman[mbr] = noise_k_tmp.copy()

                if iteration == (cfg.Kalman_iterations - 1):

                    self.out_members_kalman[mbr] = dump_tmp.copy()

        else:  # if there is not obs data just write the kalman noise
            self.noise_kalman = self.noise.copy()
            self.out_members_kalman = self.out_members.copy()

    def resample(self, resampled_particles):

        # Particles
        new_out = [self.out_members[x].copy() for x in resampled_particles]
        self.out_members = new_out.copy()

        # Noise
        new_out = [self.noise[x].copy() for x in resampled_particles]
        self.noise = new_out.copy()
