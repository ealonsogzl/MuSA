import os, sys, argparse, yaml
import pandas as pd
project_root=os.getcwd()
sys.path.append(project_root)
from modules.dem_tools import RegridDEMtoForcings
import modules.prepareForcingsZarr as prepForcing_tools
import modules.prepareRunTile_tools as prepRuntile_tools
import modules.internal_fns as ifn

#---custom functions---
def load_config(
        ConfigFile:str=None,
        ) -> dict:
    '''
    Function to load a YAML configuration file.
    '''
    with open(ConfigFile, "r") as f:
        return yaml.safe_load(f)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Regrid DEM to Forcings")

    def str2bool(value: str) -> bool:
        if isinstance(value, bool):
            return value
        value = value.strip().lower()
        if value in {"true", "1", "yes", "y", "t"}:
            return True
        if value in {"false", "0", "no", "n", "f"}:
            return False
        raise argparse.ArgumentTypeError(
            f"Invalid boolean value: {value}. Use True/False."
        )

    parser.add_argument("--implementation",
                        type=str,
                        default="open_loop",
                        help="Implementation type (default: open_loop)")
    
    parser.add_argument("--date_ini",
                        type=str,
                        default="2018-09-01 00:00",
                        help="Initial date for the simulation (default: 2018-09-01 00:00)")

    parser.add_argument("--date_end",
                        type=str,
                        default="2020-08-30 23:00",
                        help="End date for the simulation (default: 2020-08-30 21:00)")

    parser.add_argument("--snow_model",
                        type=str,
                        default="FSM2",
                        help="Snow model to use (default: FSM2)")
    parser.add_argument("--rootdirMuSAruns",
                        type=str,
                        default=os.getcwd(),
                        help="Root directory for MuSA runs (default: current working directory)"
                        )
    parser.add_argument("--model_only_sites",
                        type=str2bool,
                        default=False,
                        help="Flag to indicate if only model sites should be considered (default: False)"
                        )
    
    parser.add_argument("--tilefile", 
                        type=str,
                        default="/kyukon/data/gent/vo/000/gvo00090/SNOWSHOP/auxdata/mountain_tiles/Alps_tiles.txt",
                        help="Path to the tile file")

    parser.add_argument("--idx_tile", 
                        type=int,
                        default=0,
                        help="Index of the tile to process (default: 0)")
    
    parser.add_argument("--remove_output_cells",
                        type=str2bool,
                        default=False,
                        help="Flag to indicate if output cells should be removed after the run (default: False)")
    
    args = parser.parse_args()
    
    return args


class PrepareRunTile:
    def __init__(self, 
                 tx:int, 
                 ty:int, 
                 rootdirMuSAruns:str, 
                 date_ini:str, 
                 date_end:str, 
                 snow_model:str, 
                 implementation:str, 
                 model_only_sites:bool,
                 remove_output_cells:bool
                 ):
        self.tx=tx
        self.ty=ty
        self.rootdirMuSAruns=rootdirMuSAruns
        self.date_ini=date_ini
        self.date_end=date_end
        self.snow_model=snow_model
        self.implementation=implementation
        self.model_only_sites=model_only_sites
        self.remove_output_cells=remove_output_cells

    def runPreprocessing(self):
        ''' 
        Function that prepares the run of MuSA for a specific tile (tx,ty) and time period (date_ini, date_end).
        '''
        #generate specific directories for the run of MuSA for the specific tile (tx,ty)
        rootdirRun, forcing_dir, dem_dir=prepRuntile_tools.CreateDirectoriesMuSArunTile(
                                                        tx=self.tx,
                                                        ty=self.ty,
                                                        rootdirMuSAruns=self.rootdirMuSAruns
                                                        )
        #check if forcings are availble for the specified time period
        check_forcings_store=ifn.check_forcings_timerange(
            date_ini=self.date_ini,
            date_end=self.date_end,
            forcing_dir=forcing_dir,
            verbose=True
            )
        
        if check_forcings_store is None:
            prepForcing_tools.CreateZarrTransformedForcings(
                tx=self.tx,
                ty=self.ty,
                date_ini=self.date_ini,
                date_end=self.date_end,
                savedir=forcing_dir,
                filename="forcings.zarr"
                )
            print("Forcings zarr file created successfully.", file=sys.stderr)
        
        #check the DEM -> generate for tx,ty if required
        dem_var, dem_res=RegridDEMtoForcings(
                datadir_forcings=forcing_dir, 
                savedir=dem_dir
            )

        #adjust the config-file
        out_path=prepRuntile_tools.adjust_config_file(
                snowmodel=self.snow_model,
                rootdirRun=rootdirRun,
                dem_varname=dem_var,
                dem_res=dem_res,
                date_ini=self.date_ini,
                date_end=self.date_end,
                implementation=self.implementation,
                model_only_sites=self.model_only_sites,
                remove_output_cells=self.remove_output_cells
            )
        print("Preprocessing complete!", file=sys.stderr)

        return out_path

#---main function---
def main():
    #parse the command line arguments
    args = parse_arguments()

    #get the tiles
    tiles=pd.read_csv(args.tilefile,header=0)
    tx=tiles.iloc[args.idx_tile]["tx"]
    ty=tiles.iloc[args.idx_tile]["ty"]

    #create an instance of the PrepareRunTile class and run the preprocessing
    args_dict = vars(args)
    del args_dict["tilefile"]
    del args_dict["idx_tile"]

    prepclass=PrepareRunTile(tx=tx, ty=ty, **args_dict)
    #run the preprocessing and get the path to the adjusted config file
    config_file=prepclass.runPreprocessing()
    print(config_file)

if __name__ == "__main__":
    main()