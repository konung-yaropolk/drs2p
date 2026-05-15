import re
import os
import argparse
from ruamel.yaml import YAML

from config import RunConfig, MovieConfig, TriggerConfig
from trigger import Trigger 


def parse_metadata(file_path: str):
    txt_path = file_path[:-4] + '.txt'
    with open(txt_path, 'r') as file:
        trigger = '"[Event '
        strings = file.readlines()

        string = strings[12]
        if not string.startswith('"T Dimension"'):
            raise ValueError

        n_slides = int(re.findall(r'\t"([^[]*), ', string)[0])
        t_duration = float(re.findall(r'- ([^[]*)\ \[', string)[0])
        t_resolution = t_duration / n_slides

        events = [
            (strings[i+1][18:-2], float(strings[i+2][15:-6])/1000)
            for i, line in enumerate(strings)
            if trigger in line
        ]

    return events, t_resolution, t_duration, n_slides

def resolve_configs(child_config, parent_config):
    """
    Fill in None fields on child from parent.
    Резолвить ієрархію атрибутів дата класів
    """
    for field_name, value in vars(child_config).items():
        if value is None and hasattr(parent_config, field_name):
            parent_value = getattr(parent_config, field_name)
            if parent_value is not None:
                setattr(child_config, field_name, parent_value)

def main(config_path):
    #import parsers to read from yaml files
    yaml_parser = YAML()
    yaml_parser.preserve_quotes = True
    #oarse yaml file
    with open(config_path) as file:
        raw = yaml_parser.load(file)
    #populate the data objects: RunConfig, MovieConfig, TriggerConfig - 
    run_config = RunConfig(
        movies=[
            MovieConfig(
                triggers=[TriggerConfig(**t) for t in f.get("triggers", [])],
                **{k: v for k, v in f.items() if k != "triggers"}
         )
            for f in raw.get("movies", [])],
        **{k: v for k, v in raw.items() if k != "movies"}) #extract everything except movies
    
    # set working_dir from yaml file location
    run_config.working_dir = os.path.dirname(os.path.abspath(config_path)) + '/'
    
    for movie_config in run_config.movies:
        resolve_configs(movie_config, run_config)
        for trigger_config in movie_config.triggers:
            resolve_configs(trigger_config, movie_config);
    
    modified = False

    # now check each movie and parse metadata if missing this will create movie objects and run the analysis but also  check if the script already had run 
    #and if not it will parse the metadata from the description file (like the timestamps from the events and movie duration etc)
    for i, movie_config in enumerate(run_config.movies):
        if movie_config.events is None:
            # create Movie once per tiff the metadata reading method is called at the time of initialization 
            file_path = os.path.join(run_config.working_dir,movie_config.file_name)
            events, t_resolution, t_duration, n_slides = parse_metadata(file_path)
            #edit the config file
            movie_config.events = events
            movie_config.seconds_per_frame = t_resolution
            movie_config.movie_duration = t_duration
            movie_config.n_frames = n_slides
            #edit the raw yaml data to be able to write it in
            raw['movies'][i]['events'] = events
            raw['movies'][i]['seconds_per_frame'] = t_resolution
            raw['movies'][i]['movie_duration'] = t_duration
            raw['movies'][i]['n_frames'] = n_slides
            #this flag tells us if we will need to amend the yaml file with the stuff from the parser
            modified = True

    if modified:
        with open(config_path, 'w') as file:
            yaml_parser.dump(raw, file)  # preserves formatting
    

    # v_shifts = {}
    # filters = {}
    # ampls={}
    for movie_config in run_config.movies:
        # sort triggers - independent ones first, dependent ones second
        # this ensures SD_filter_of_trig and vertical_shift_of_trig
        # always have their reference trigger results available
        # independent = sorted(
        #     [t for t in movie_config.triggers
        #      if not t.SD_filter_of_trig
        #      and not t.vertical_shift_of_trig],
        #     key=lambda t: t.trig_number
        # )
        # dependent = sorted(
        #     [t for t in movie_config.triggers
        #      if t.SD_filter_of_trig
        #      or t.vertical_shift_of_trig],
        #     key=lambda t: t.trig_number
        # )
        for trigger in sorted(movie_config.triggers, key=lambda t: t.trig_number):
            try:
                trigger_analysis = Trigger(
                    run_config=run_config,
                    movie_config=movie_config,
                    trigger_config=trigger,
                    # v_shifts=v_shifts,
                    # filters=filters
                )
                trigger_analysis.run()
                
            #     # # collect results for next triggers
            #     # v_shifts.update(trigger_analysis.v_shifts_return)
            #     # filters.update(trigger_analysis.filters_return)
                
            #     # print(trigger_analysis.log)
            
            except Exception as e:
                print(f'Error in trigger {trigger.trig_number} '
                      f'({trigger.label}): {repr(e)}')
            
            finally:
                if 'trigger_analysis' in locals():
                    del trigger_analysis # they get deleted even if there is an error
   
    # if multiprocessing:
    #     import multiprocessing as mp

    #     cores = mp.cpu_count()          # CPU cores count
    #     jobs = len(to_do_list)          # jobs to do count

    #     if processes_limit == 0:
    #         processes_limit = 1000

    #     threads = min(cores-2, jobs,
    #                   processes_limit)
    #     try:
    #         pool = mp.Pool(threads)
    #     except ValueError:
    #         print('No one file listed, there is nothing to do.')
    #         return 0

    #     v_shifts = {}
    #     filters = {}

    #     def spread_jobs(jobs):
    #         processes = [pool.apply_async(worker, args=(item, run_derivatives_calculation, run_traces_calculation, v_shifts, filters))
    #                      for item in jobs]
    #         output = [p.get() for p in processes]
        #     return output

        # print('\nParallel processing mode activated:')
        # print('Please, ensure if you have enough RAM for multiprocessing.')
        # print('If processing went wrong, please, use "processes_limit" option in the settings.py')
        # print('{0} cpu cores per queue of {1} files found, pool of {2} processes created.'.format(
        #     cores, jobs, threads))
        # print('\nJob started...\n')

        # # separating the jobs that have to be done first,
        # # because they do not use the results of the previous calculations
        # do_first = [i for i in to_do_list if not (i[1]
        #             ['vertical_shift_of_trig'] or
        #             i[1]['SD_filter_of_trig'])]
        # do_second = [i for i in to_do_list if (i[1]
        #              ['vertical_shift_of_trig'] or
        #              i[1]['SD_filter_of_trig'])]

        # output = spread_jobs(do_first)

        # for i in output:
        #     v_shifts.update(i[0])
        #     filters.update(i[1][0])

        # output.extend(spread_jobs(do_second))

        # errors = [[i[2]+':\n', 'derivatives : ' + i[3]+'\n',
        #            'calculations:   ' + i[4]+'\n', '\n'] for i in output if (i[3] or i[4])]
        # msg = [item for sublist in errors for item in sublist] if errors else [
        #     '✅ --no errors--\n']

        # print('\n\nAll done.\n')
        # print('Errors: \n')
        # print(*msg)

    # else:
    #     output = []
    #     for item in to_do_list:
    #         v_shifts = {}
    #         filters = {}
    #         output.append(worker(item, run_derivatives_calculation,
    #                              run_traces_calculation, v_shifts, filters))
    #         v_shifts = output[-1][0]
    #         filters = output[-1][1][0]
    #         # ampls = output[-1][1][1]
    #         # aucs = output[-1][1][2]

    #         if output[-1][3]:
    #             print(output[-1][3])
    #         if output[-1][4]:
    #             print(output[-1][4])

    # if postprocessing_summary and run_traces_calculation:
    #     try:
    #         generate_postprocessing_summary(output)
    #     except IndexError as e:
    #         print(
    #             'Postprocesssing: Index error - only one timeframe in a boundle so there is nothing to compare')

def entry_point():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True, 
                       help='please, add "--path path/to/your/yaml/file"')
    args = parser.parse_args()
    main(config_path=args.path)

# to run the script first act env(e.g. source venv2/bin/activate), then python classes.py --path 'F:/Lab Work Files/2-photon/2025_09_18/experiment.yaml'
if __name__ == '__main__':
    entry_point()