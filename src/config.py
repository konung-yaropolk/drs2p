from typing import Optional, Literal
from dataclasses import dataclass, field, asdict

# RunConfig = single source of truth for all defaults these are a source of defaults unless other configs specify otherwise
@dataclass  
class TriggerConfig:
    #required and trigger specific
    label: str
    trig_number: int
    #pipeline control
    run_derivatives: Optional[bool] = None
    run_traces: Optional[bool] = None
    roi_detection: Optional[str] = None
    roi_file_index: Optional[int] = None

    # ALL analysis defaults live here and only here
    relative_values: Optional[bool] = None # plot and calculate in for of dF/F0 (almost always true unless certain exotic cases)
    sync_coef: Optional[float] = None
    frame_lag_derivatives: Optional[int] = None 

    #stimulation parameters
    stim_1_name: Optional[str] = None #the stimulator name for the first stimulation, they can not be the same
    stim_2_name: Optional[str] = None
    drs_pattern: Optional[list] = None #the pattern if the there are 2 stimulators(always) the length of the array is always 2.  The length of the subarrays denotes the number of stimulations in a given episode
    step_duration: Optional[float] = None #seconds duration of each stimulation duration inside a pattern, 10s for DRS L4C-10s-L5C, same for single stimulation (10s is enough for vesicles to renew and avoid potential side effects) epoch duration is calculated from this pattern
    n_epochs: Optional[int] = None #amount of times the signal is repeated, like 10 for DRS L4C-10s-L5C of 1 for single stimulations
    start_from_epoch: Optional[int] = None # start the processing from this epoch (ignore the previous epochs) 
  
    #derivatives calculation specific params
    resp_duration: Optional[float] = None # in sec, expected response duration tonic(3s), l4-l5(2s), single(1.5) this is for derivative calculation calcium indicator spedific dynamics variable, less is better, but you have to be sure that the peak of the signal is located within window, sometimes needs fine tuning

    # overall traces plotting specific params
    baseline_duraton: Optional[float] = None # time taken before trigger to calculate the baseline (in s) 
    time_before_trig: Optional[float] = None # time taken before the trigger the triggerv (zero timepoint) to be shown in the overal graph. It does not affect the calculations, and only affect the overall plot. None means take all recording from the trig to the end
    time_after_trig: Optional[float] = None # the length of the expextrd recording time after the trigger (to be shown in the overall graph). It does not affect the calculations, and only affect the overall plot. None means take all recording from the trig to the end
    vertical_shift: Optional[float] = None # vertical vertical_shift for all-traces graph in dF/F0 units, set 0 to make vertical_shift the same as largest responce
    vertical_shift_of_trig: Optional[int] = None # use vertical shift from this trig to plot in the same scales

    #tracecals
    sigmas_treshold: Optional[float] = None  # how many sigmas (standard deviations) signal has to exceed the baseline noise to be considered as a response
    SD_filter_of_trig: Optional[int] = None     # use binarization based on SD from this trig to compare the same ROIs

    # csv input format (depends on the imagej csv export parameters). Do not change if it works fine
    mean_col_order: Optional[int] = None # imagej default
    cols_per_roi: Optional[int] = None # imagej default

@dataclass
class MovieConfig: #since one yaml can specify multipl experiments, there is a separate class for experiment configs and separate for the whole run config file
    #pipeline control
    run_derivatives: Optional[bool] = None
    run_traces: Optional[bool] = None
    roi_detection: Optional[str] = None
    roi_file_index: Optional[int] = None
    movie_stabilization: Optional[bool] = None
    
    # ALL analysis defaults live here and only here
    relative_values: Optional[bool] = None # plot and calculate in for of dF/F0 (almost always true unless certain exotic cases)
    sync_coef: Optional[float] = None
    frame_lag_derivatives: Optional[int] = None 

    #stimulation parameters
    stim_1_name: Optional[str] = None #the stimulator name for the first stimulation, they can not be the same
    stim_2_name: Optional[str] = None
    drs_pattern: Optional[list] = None #the pattern if the there are 2 stimulators(always) the length of the array is always 2.  The length of the subarrays denotes the number of stimulations in a given episode
    step_duration: Optional[float] = None #seconds duration of each stimulation duration inside a pattern, 10s for DRS L4C-10s-L5C, same for single stimulation (10s is enough for vesicles to renew and avoid potential side effects) epoch duration is calculated from this pattern
    n_epochs: Optional[int] = None #amount of times the signal is repeated, like 10 for DRS L4C-10s-L5C of 1 for single stimulations
    start_from_epoch: Optional[int] = None # start the processing from this epoch (ignore the previous epochs) 
  
    #derivatives calculation specific params
    resp_duration: Optional[float] = None # in sec, expected response duration tonic(3s), l4-l5(2s), single(1.5) this is for derivative calculation calcium indicator spedific dynamics variable, less is better, but you have to be sure that the peak of the signal is located within window, sometimes needs fine tuning


    # overall traces plotting specific params
    baseline_duraton: Optional[float] = None # time taken before trigger to calculate the baseline (in s) 
    time_before_trig: Optional[float] = None # time taken before the trigger the triggerv (zero timepoint) to be shown in the overal graph. It does not affect the calculations, and only affect the overall plot. None means take all recording from the trig to the end
    time_after_trig: Optional[float] = None # the length of the expextrd recording time after the trigger (to be shown in the overall graph). It does not affect the calculations, and only affect the overall plot. None means take all recording from the trig to the end
    vertical_shift: Optional[float] = None # vertical vertical_shift for all-traces graph in dF/F0 units, set 0 to make vertical_shift the same as largest responce
    vertical_shift_of_trig: Optional[int] = None # use vertical shift from this trig to plot in the same scales

    #tracecals
    sigmas_treshold: Optional[float] = None  # how many sigmas (standard deviations) signal has to exceed the baseline noise to be considered as a response
    SD_filter_of_trig: Optional[int] = None     # use binarization based on SD from this trig to compare the same ROIs


    # csv input format (depends on the imagej csv export parameters). Do not change if it works fine
    mean_col_order: Optional[int] = None # imagej default
    cols_per_roi: Optional[int] = None # imagej default

    # movie specific
    #required
    file_name: str='' #actual tiff file name
    #optional, filled out after and then written back to yaml 
    events: Optional[list] = None
    seconds_per_frame: Optional[float] = None
    movie_duration: Optional[float] = None
    n_frames: Optional[int] = None

    #movie specific not optuinal
    triggers: list[TriggerConfig] = field(default_factory=list) # since one yaml can specify multipl experiments, there is a separate class for experiment configs and separate for the whole run config file

@dataclass  
class RunConfig:
    #required
    working_dir: str='' # is always set from yaml file location in main()
    fiji_path: str='' # path to fiji(or imageJ) executable
    #pipeline control
    run_derivatives: bool = True
    run_traces: bool = True
    multiprocessing: bool = False
    processes_limit: int = 10
    roi_detection: Literal['all', 'manual', 'skip', "fixed"] = 'manual' #"all" - takes rois for all derrivatives, manual - asks user for each, skip - skips, fixed - determined by roi_file_index
    roi_file_index: Optional[int] = 2
    movie_stabilization: bool = True
    
    # ALL analysis defaults live here and only here
    relative_values: bool = True # plot and calculate in for of dF/F0 (almost always true unless certain exotic cases)
    sync_coef: float = -0.003 ## Adjust the sampling interval to account for the
    # clock synchronization of the microscope's hardware and PC
    # Coefficient estimated experimentally
    # -0.0031556459008686036  more precise
    # -0.0029183722446345     old one estimation
    # -0.00313                compromise
    frame_lag_derivatives: int =-1 # lag in frames to calculate derivatives, to avoid artifacts at the edges of epochs


    #stimulation parameters
    stim_1_name: str = '#1' #the stimulator name for the first stimulation, they can not be the same
    stim_2_name: str = '#2'
    drs_pattern: list = field(default_factory=lambda: [[0,1],[1, 0]]) #the pattern if the there are 2 stimulators(always) the length of the array is always 2.  The length of the subarrays denotes the number of stimulations in a given episode
    step_duration: float = 10.0 #seconds duration of each stimulation duration inside a pattern, 10s for DRS L4C-10s-L5C, same for single stimulation (10s is enough for vesicles to renew and avoid potential side effects) epoch duration is calculated from this pattern
    n_epochs: int=1 #amount of times the signal is repeated, like 10 for DRS L4C-10s-L5C of 1 for single stimulations
    start_from_epoch: int = 1 # start the processing from this epoch (ignore the previous epochs) 
    
    #derivatives calculation specific params
    resp_duration: float = 4.0 # in sec, expected response duration tonic(3s), l4-l5(2s), single(1.5) this is for derivative calculation calcium indicator spedific dynamics variable, less is better, but you have to be sure that the peak of the signal is located within window, sometimes needs fine tuning


    # overall traces plotting specific params
    baseline_duraton: float = 10.0 # time taken before trigger to calculate the baseline (in s) 
    time_before_trig: float = 10.0 # time taken before the trigger the triggerv (zero timepoint) to be shown in the overal graph. It does not affect the calculations, and only affect the overall plot. None means take all recording from the trig to the end
    time_after_trig: Optional[float] = None # the length of the expextrd recording time after the trigger (to be shown in the overall graph). It does not affect the calculations, and only affect the overall plot. None means take all recording from the trig to the end
    vertical_shift: float=0.0 # vertical vertical_shift for all-traces graph in dF/F0 units, set 0 to make vertical_shift the same as largest responce
    vertical_shift_of_trig: int = 0 # use vertical shift from this trig to plot in the same scales

    #tracecals
    sigmas_treshold: float = 5.0  # how many sigmas (standard deviations) signal has to exceed the baseline noise to be considered as a response
    SD_filter_of_trig: int = 0     # use binarization based on SD from this trig to compare the same ROIs


    # csv input format (depends on the imagej csv export parameters). Do not change if it works fine
    mean_col_order: int = 2 # imagej default
    cols_per_roi: int = 4 # imagej default


    movies: list[MovieConfig] = field(default_factory=list)