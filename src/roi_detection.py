# roi_detector.py
import subprocess
import os

class RoiDetector:
    def __init__(self, fiji_path, derivative_tiff, dir):
        self.fiji_path = fiji_path
        self.derivative_tiff = derivative_tiff
        self.dir = dir
    
    def run(self):
        macro = self._generate_macro()
        macro_path = os.path.join(self.dir, '_temp_macro.ijm')
        
        with open(macro_path, 'w') as f:
            f.write(macro)
        
        try:
            subprocess.run([self.fiji_path, '--run', macro_path], 
                         check=True, timeout=300)
        finally:
            if os.path.exists(macro_path):
                os.remove(macro_path)
    
    def _generate_macro(self):
        roi_path = os.path.join(
            self.dir, f"Roi_{self.derivative_tiff}.zip")
        
        return f"""
function detectAndSaveParticles(savepath) {{
    run("16-bit");
    run("Detect Particles", "ch1i ch1l ch1a=6 ch1s=4 rois=Ovals add=[All detections] summary=Reset");
    run("Select All");
    roiManager("Select All");
    n = roiManager("count");
    getDimensions(w_img, h_img, d, t, s);
    for (i=n-1; i>=0; i--) {{
        roiManager("Select", i);
        getSelectionBounds(x, y, w, h);
        if (x < w_img*0.05 || y < h_img*0.05 ||
            (x+w) > w_img*0.95 || (y+h) > h_img*0.95) {{
            roiManager("Delete");
        }}
    }}
    roiManager("Save", savepath);
    roiManager("Delete");
    run("Clear Results");
    selectWindow("Results");
    run("Close");
    selectWindow("Log");
    run("Close");
    selectWindow("Summary");
    run("Close");
    close("*");
}}

open("{self.derivative_tiff}");
title = getTitle();
detectAndSaveParticles("{roi_path}");
eval("script", "System.exit(0);");
"""