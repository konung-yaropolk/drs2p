from derivatives import DerivativesCalc 
from traces import TracesCalc

class Trigger:
    def __init__(self,
                 run_config,
                 movie_config,
                 trigger_config,
                 ** misc):
        self.run_config = run_config
        self.movie_config = movie_config
        self.trigger_config = trigger_config
        self.derivatives = DerivativesCalc(run_config,movie_config,trigger_config)
        self.log = ' \n'
        self.traces = TracesCalc(run_config,movie_config,trigger_config)
        # self.v_shifts = v_shifts
        # self.filters = filters
    def run(self):
        if self.trigger_config.run_derivatives: 
            self.derivatives.run()
        if self.trigger_config.run_traces:
            self.traces.run()
        self.log += self.derivatives.log
        self.log += self.traces.log
    