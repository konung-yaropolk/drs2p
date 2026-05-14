from derivatives import DerivativesCalc 
from traces import TracesCalc
from roi_detection import RoiDetector
import os

DERIVATIVES_SUBFOLDER_NAME = "_DERIVATIVES_auto_"

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
        #ask if we want bouton detection
        if self.trigger_config.roi_detection != 'skip':
            self._roi_detection()
        if self.trigger_config.run_traces:
            self.traces.run()
        self.log += self.derivatives.log
        self.log += self.traces.log

    def _roi_detection(self):

        #detect all the derivative files
        derivatives_dir = os.path.join(
        self.derivatives.path,
        self.derivatives.file + DERIVATIVES_SUBFOLDER_NAME + self.trigger_config.label)
        if not os.path.exists(derivatives_dir):
            print(f"Derivatives folder not found: {derivatives_dir}")
            return
        tiffs = [f for f in os.listdir(derivatives_dir) 
             if f.startswith('_DERIVATIVES') 
             and f.endswith('.tif')
             and not f.startswith('._')]
        if not tiffs:
            print(f"No tiffs found in {derivatives_dir}")
            return
        rois_to_detect = []
        if self.trigger_config.roi_detection == 'fixed':
            rois_to_detect = [tiffs[int(self.trigger_config.roi_file_index) - 1]];
        elif self.trigger_config.roi_detection == 'all':
            rois_to_detect = tiffs
        else:
            for i, tiff in enumerate(tiffs):
                print(f"  {i+1}. {tiff}")
            print("  0. Detect All ROI")
            print("  -1. Skip")
            selection = input("Select tiff for ROI detection: ")
            if selection == '-1':
                print("Skipping ROI detection")
                rois_to_detect = []
            elif selection == '0':
                rois_to_detect = tiffs
            else:
                rois_to_detect = [tiffs[int(selection) - 1]]
        for selected_tiff in rois_to_detect:
                RoiDetector(
                fiji_path=self.run_config.fiji_path,
                derivative_tiff=selected_tiff,
                dir=derivatives_dir
            ).run()

       

    
    