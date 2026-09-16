# roi_detector.py
import subprocess
import os
import tifffile
import roifile
import numpy as np
import csv
import ssd_guard
from skimage.draw import ellipse
from helpers import Helpers


class RoiDetector:
    def __init__(self, fiji_path, derivative_tiff, dir):
        self.fiji_path = Helpers().normalize_path(fiji_path)
        self.derivative_tiff = derivative_tiff
        self.dir = dir
    
    def run(self):
        macro = self._generate_macro()
        macro_path = Helpers().normalize_path(os.path.join(self.dir, '_temp_macro.ijm'))

        # not routed through ssd_guard: this macro is deleted again in the
        # finally block below, so it never pre-exists and there is nothing to
        # compare it against
        with open(macro_path, 'w') as f:
            f.write(macro)
        
        try:
            subprocess.run([self.fiji_path, '--run', macro_path], 
                         check=True, timeout=300)
        finally:
            if os.path.exists(macro_path):
                os.remove(macro_path)
    
    def _generate_macro(self):
        roi_path = Helpers().normalize_path(os.path.join(
            self.dir, f"Roi_{self.derivative_tiff}.zip"))
        
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

open("{Helpers().normalize_path(os.path.join(self.dir, self.derivative_tiff), target="posix")}");
title = getTitle();
detectAndSaveParticles("{Helpers().normalize_path(roi_path, target="posix")}");
//eval("script", "System.exit(0);");    // Mac specific exit
run("Quit");                            // Windows specific exit
"""

class RoiMeasurer:
    def __init__(self, dir, tiff, roi_zip, output_csv, seconds_per_frame):
        self.dir = Helpers().normalize_path(dir)
        self.tiff = Helpers().normalize_path(tiff)
        self.roi_zip = Helpers().normalize_path(roi_zip)
        self.output_csv = Helpers().normalize_path(output_csv)
        self.seconds_per_frame = seconds_per_frame
    def load_rois(self):
        if self.roi_zip.endswith('.zip'):
            return roifile.roiread(self.roi_zip)
        elif self.roi_zip.endswith('.roi'):
            return [roifile.roiread(self.roi_zip)]
        else:
            raise ValueError(f"Unknown ROI file format: {self.roi_zip}")
    def roi_to_mask_freehand(self, roi, shape):
        """
        Direct port of ImageJ's PolygonFiller algorithm
        for freehand/polygon ROIs
        """
        import math

        coords = roi.coordinates()
        if coords is None or len(coords) == 0:
            return np.zeros(shape, dtype=bool)

        # get coordinates as floats
        xf = coords[:, 0].astype(float) 
        yf = coords[:, 1].astype(float)
        n = len(xf)
    
        # offsets for coordinate system
        x_offset = 0  # coords already in global image space
        y_offset = 0
        
        height, width = shape
        
        # allocate edge table arrays
        max_edges = n
        ex = np.zeros(max_edges)      # x coordinate of edge at current y
        ey1 = np.zeros(max_edges, dtype=int)  # y start of edge
        ey2 = np.zeros(max_edges, dtype=int)  # y end of edge
        eslope = np.zeros(max_edges)   # slope of edge
        sedge = np.zeros(max_edges, dtype=int)  # sorted edge indices
        aedge = np.zeros(max_edges * 2, dtype=int)  # active edge indices
        
        # build edge table
        y_min = 2**31 - 1
        y_max = -(2**31)
        edges = 0
        poly_start = 0
    
        for i in range(n):
            iplus1 = poly_start if i == n-1 else i+1
            
            if math.isnan(xf[iplus1]):
                iplus1 = poly_start
            if math.isnan(xf[i]):
                poly_start = i + 1
                continue
            
            y1f = yf[i] + y_offset
            y2f = yf[iplus1] + y_offset
            x1f = xf[i] + x_offset
            x2f = xf[iplus1] + x_offset
            
            y1 = int(round(y1f))
            y2 = int(round(y2f))
            
            if y1 == y2 or (y1 <= 0 and y2 <= 0):
                continue
            
            if y1 > y2:
                y1, y2 = y2, y1
                y1f, y2f = y2f, y1f
                x1f, x2f = x2f, x1f
        
            slope = (x2f - x1f) / (y2f - y1f)
            ex[edges] = x1f + (y1 - y1f + 0.5) * slope + 1e-8
            ey1[edges] = y1
            ey2[edges] = y2
            eslope[edges] = slope
            
            if y1 < y_min:
                y_min = y1
            if y2 > y_max:
                y_max = y2
            
            edges += 1
    
        for i in range(edges):
            sedge[i] = i
        
        active_edges = 0
        
        # helper functions
        def sort_active_edges():
            nonlocal active_edges
            for i in range(active_edges):
                min_idx = i
                for j in range(i, active_edges):
                    if ex[aedge[j]] < ex[aedge[min_idx]]:
                        min_idx = j
                aedge[i], aedge[min_idx] = aedge[min_idx], aedge[i]
        
        def remove_inactive_edges(y):
            nonlocal active_edges
            i = 0
            while i < active_edges:
                index = aedge[i]
                if y < ey1[index] or y >= ey2[index]:
                    for j in range(i, active_edges - 1):
                        aedge[j] = aedge[j+1]
                    active_edges -= 1
                else:
                    i += 1
        
        def activate_edges(y):
            nonlocal active_edges
            for i in range(edges):
                edge = sedge[i]
                if y == ey1[edge]:
                    index = 0
                    while index < active_edges and ex[edge] > ex[aedge[index]]:
                        index += 1
                    for j in range(active_edges - 1, index - 1, -1):
                        aedge[j+1] = aedge[j]
                    aedge[index] = edge
                    active_edges += 1
    
        def update_x_coordinates():
            nonlocal active_edges
            x1 = -float('inf')
            sorted_edges = True
            for i in range(active_edges):
                index = aedge[i]
                x2 = ex[index] + eslope[index]
                ex[index] = x2
                if x2 < x1:
                    sorted_edges = False
                x1 = x2
            if not sorted_edges:
                sort_active_edges()
    
        # shift x values for starting y
        mask = np.zeros(shape, dtype=bool)
        y_start = max(y_min, 0)
        
        if y_min != 0:
            for i in range(edges):
                index = sedge[i]
                if ey1[index] < y_start and ey2[index] >= y_start:
                    ex[index] += eslope[index] * (y_start - ey1[index])
                    aedge[active_edges] = index
                    active_edges += 1
            sort_active_edges()
    
        # fill mask scan line by scan line
        for y in range(y_start, min(height, y_max + 1)):
            remove_inactive_edges(y)
            activate_edges(y)
            
            for i in range(0, active_edges, 2):
                x1 = int(ex[aedge[i]] + 0.5)
                x1 = max(0, min(width, x1))
                x2 = int(ex[aedge[i+1]] + 0.5)
                x2 = max(0, min(width, x2))
                mask[y, x1:x2] = True
            
            update_x_coordinates()
    
        return mask
    def roi_to_mask(self, roi, shape):
        """Convert ImagejRoi oval to boolean mask"""
        mask = np.zeros(shape, dtype=bool)
        roi_type = roi.roitype
        if roi_type==2:
            xbase = roi.left
            ybase = roi.top
            width = roi.right - roi.left
            height = roi.bottom - roi.top
            xradius = width / 2.0
            yradius = height / 2.0

            for y in range(height):
                for x in range(width):
                    dx = (x + 0.5 - xradius) / xradius
                    dy = (y + 0.5 - yradius) / yradius
                    if dx*dx + dy*dy <= 1.0:
                        mask[ybase + y, xbase + x] = True
        elif roi_type in (7, 8):  # freehand, traced
           mask = self.roi_to_mask_freehand(roi, shape)
        elif roi_type == 0:  # polygon
           mask = self.roi_to_mask_freehand(roi, shape)
        elif roi_type == 1:  # rect
           mask[roi.top:roi.bottom, roi.left:roi.right] = True
        else:
            print(f"WARNING: Unsupported ROI type {roi.roitype} for ROI {roi.name} — skipping")
        
        return mask
    def run(self):
        print(f"Measuring ROIs from {self.roi_zip}")
        
        # load ROIs
        rois = self.load_rois()
        print(f"Found {len(rois)} ROIs")
        
        results = []
        #load tiff file and go through tiff pages in the movie one by one 
        tiff_path = Helpers().normalize_path(os.path.join(self.dir, self.tiff))
        with tifffile.TiffFile(tiff_path) as tif:
            n_frames = len(tif.pages)
            #csv header, start with empty first cell for frames column
            header = [""]
            for roi in rois:
                name = roi.name
                header.extend([
                    f'Area({name})',
                    f'Mean({name})',
                    f'Min({name})',
                    f'Max({name})',
                ])
            results.append(header)
            print(f"Processing {n_frames} frames...")
            for frame_idx, page in enumerate(tif.pages):
                frame = page.asarray()
                row = [frame_idx + 1] # 1-indexed like ImageJ
                for roi in rois:
                    mask = self.roi_to_mask(roi, frame.shape)
                    pixels = frame[mask].astype(float)
                
                    if len(pixels) == 0:
                        row.extend([0, 0, 0, 0])
                        continue

                    area = np.sum(mask)
                    mean = round(float(np.mean(pixels)), 3)
                    min_val = round(float(np.min(pixels)), 3)
                    max_val = round(float(np.max(pixels)), 3)

                    row.extend([area, mean, min_val, max_val])
                
                results.append(row)
                
                # progress update every 100 frames
                if frame_idx % 100 == 0:
                    print(f"  frame {frame_idx}/{n_frames}")
        with ssd_guard.guarded_open(self.output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(results)

        print(f"Saved to {self.output_csv}")
        

    
        
