def main(
    working_dir=s.working_dir,
    to_do_list=s.to_do_list,
    run_derivatives_calculation=s.run_derivatives_calculation,
    run_traces_calculation=s.run_traces_calculation,
    resp_duration=s.resp_duration,
    step_duration=s.step_duration,
    n_epochs=s.n_epochs,
    drs_pattern=s.drs_pattern,
    stim_1_name=s.stim_1_name,
    stim_2_name=s.stim_2_name,
    relative_values=s.relative_values,
    mean_col_order=s.mean_col_order,
    cols_per_roi=s.cols_per_roi,
    time_before_trig=s.time_before_trig,
    baseline_duraton=s.baseline_duraton,
    sync_coef=s.sync_coef,
    frame_lag_derivatives=s.frame_lag_derivatives,
    sigmas_treshold=s.sigmas_treshold,
    vertical_shift=s.vertical_shift,
    vertical_shift_of_trig=s.vertical_shift_of_trig,
    SD_filter_of_trig=s.SD_filter_of_trig,
    time_after_trig=s.time_after_trig,
    multiprocessing=s.multiprocessing,
    processes_limit=s.processes_limit,
):

    for item in to_do_list:

        # setting default parameters if they are missing in the to_do_list
        item[1].setdefault("output_suffix", "")
        item[1].setdefault("working_dir", working_dir)
        item[1].setdefault("resp_duration", resp_duration)
        item[1].setdefault("drs_pattern", drs_pattern)
        item[1].setdefault("step_duration", step_duration)
        item[1].setdefault("n_epochs", n_epochs)
        item[1].setdefault("start_from_epoch", 1)
        item[1].setdefault("trig_number", 1)
        item[1].setdefault("time_before_trig", time_before_trig)
        item[1].setdefault("time_after_trig", time_after_trig)
        item[1].setdefault("baseline_duraton", baseline_duraton)
        item[1].setdefault("sync_coef", sync_coef)
        item[1].setdefault("frame_lag_derivatives", frame_lag_derivatives)
        item[1].setdefault("relative_values", relative_values)
        item[1].setdefault("mean_col_order", mean_col_order)
        item[1].setdefault("cols_per_roi", cols_per_roi)
        item[1].setdefault("stim_1_name", stim_1_name)
        item[1].setdefault("stim_2_name", stim_2_name)
        item[1].setdefault("sigmas_treshold", sigmas_treshold)
        item[1].setdefault("vertical_shift", vertical_shift)
        item[1].setdefault("vertical_shift_of_trig", vertical_shift_of_trig)
        item[1].setdefault("SD_filter_of_trig", SD_filter_of_trig)

    if multiprocessing:
        import multiprocessing as mp

        cores = mp.cpu_count()  # CPU cores count
        jobs = len(to_do_list)  # jobs to do count

        if processes_limit == 0:
            processes_limit = 1000

        threads = min(cores - 2, jobs, processes_limit)
        try:
            pool = mp.Pool(threads)
        except ValueError:
            print("No one file listed, there is nothing to do.")
            return 0

        v_shifts = {}
        filters = {}

        def spread_jobs(jobs):
            processes = [
                pool.apply_async(
                    worker,
                    args=(
                        item,
                        run_derivatives_calculation,
                        run_traces_calculation,
                        v_shifts,
                        filters,
                    ),
                )
                for item in jobs
            ]
            output = [p.get() for p in processes]
            return output

        print("\nParallel processing mode activated:")
        print("Please, ensure if you have enough RAM for multiprocessing.")
        print(
            'If processing went wrong, please, use "processes_limit" option in the settings.py'
        )
        print(
            "{0} cpu cores per queue of {1} files found, pool of {2} processes created.".format(
                cores, jobs, threads
            )
        )
        print("\nJob started...\n")

        # separating the jobs that have to be done first,
        # because they do not use the results of the previous calculations
        do_first = [
            i
            for i in to_do_list
            if not (i[1]["vertical_shift_of_trig"] or i[1]["SD_filter_of_trig"])
        ]
        do_second = [
            i
            for i in to_do_list
            if (i[1]["vertical_shift_of_trig"] or i[1]["SD_filter_of_trig"])
        ]

        output = spread_jobs(do_first)

        for i in output:
            v_shifts.update(i[0])
            filters.update(i[1][0])

        output.extend(spread_jobs(do_second))

        errors = [
            [
                i[2] + ":\n",
                "derivatives : " + i[3] + "\n",
                "calculations:   " + i[4] + "\n",
                "\n",
            ]
            for i in output
            if (i[3] or i[4])
        ]
        msg = (
            [item for sublist in errors for item in sublist]
            if errors
            else ["✅ --no errors--\n"]
        )

        print("\n\nAll done! ✨ 🍰 ✨\n")
        print("Errors: \n")
        print(*msg)

    else:
        output = []
        for item in to_do_list:
            v_shifts = {}
            filters = {}
            output.append(
                worker(
                    item,
                    run_derivatives_calculation,
                    run_traces_calculation,
                    v_shifts,
                    filters,
                )
            )
            v_shifts = output[-1][0]
            filters = output[-1][1][0]
            # ampls = output[-1][1][1]
            # aucs = output[-1][1][2]

            if output[-1][3]:
                print(output[-1][3])
            if output[-1][4]:
                print(output[-1][4])

        print("\n\nAll done! ✨ 🍰 ✨\n")

    if postprocessing_summary and run_traces_calculation:
        try:
            generate_postprocessing_summary(output)
        except IndexError as e:
            print(
                "Postprocesssing: Index error - only one timeframe in a boundle so there is nothing to compare"
            )
