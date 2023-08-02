import json
import pandas as pd
import numpy as np
import cv2
import os
from RoadEngin.PMG.Post_processing import *


class PMG_postprocessing:
    """
    This class is used to store the parameters for the post-processing of the PMG model.
    
    Attributes:
        unit (str): Unit of the project. Default is 'ft'.
        distance_between_frames (int): Distance between frames in the project. Default is 20.
        frame_width (int): Width of the frame in the project. Default is 24.
        y_top (int): Top of the frame in the project. Default is 350.
        kernel_size (int): Kernel size for the dilation of the cracks. Default is 5.
        iterations (int): Number of iterations for the dilation of the cracks. Default is 1.
        borders (list): List of the borders of the project. Default is ['FPAV','CUN','VEG','TRR','CAR'].
        threshold_predominantly (float): Threshold for the predominantly cracks. Default is 0.9.
        centroids_mindist (int): Minimum distance between centroids. Default is 80.
        threshold_area (int): Threshold for the area of the cracks. Default is 5600.
        abs_min_area (int): Minimum area of the cracks. Default is 1000.
        FB_min_area (int): Minimum area of the FB cracks. Default is 8000.
        FB_min_area2 (int): Minimum area of the FB cracks. Default is 20000.
        FJ_min_length (int): Minimum length of the FJ cracks. Default is 250.
        min_area_crk_end (int): Minimum area of the cracks. Default is 300.
        PA_min_area_to_evaluate (int): Minimum area of the PA cracks. Default is 62500.
        DC_min_area_to_evaluate (int): Minimum area of the DC cracks. Default is 5600.
        DC_min_area_exclusion (int): Minimum area of the DC cracks. Default is 160.
        DC_true_percentage (float): Threshold for the DC cracks. Default is 0.70.
        square_ratio (int): Ratio for the square of the DC cracks. Default is 3.
        BCH_min_area_to_evaluate (int): Minimum area of the BCH cracks. Default is 5600.
        BCH_true_percentage (float): Threshold for the BCH cracks. Default is 0.70.
        PCH_min_area (int): Minimum area of the PCH cracks. Default is 90000.
        consider_RMs (bool): Consider the RMs for the PA cracks. Default is True.
        min_area (int): Minimum area of the cracks. Default is 400.
        DSU_min_area (int): Minimum area of the DSU cracks. Default is 62500.
        thrRMarea (int): Threshold for the RMs. Default is 25000.
        min_area_PCH (int): Minimum area of the PCH cracks. Default is 60000.
        crk_no_crk_percentage (dict): Dictionary with the percentage of the cracks. Default is {2:.15, 3:.30}.
        big_area (list): List of the big areas. Default is ['DSU_1','DSU_2','DSU_3','PA_1','PA_2','PA_3'].
        threshold_percentage (float): Threshold for the percentage of the cracks. Default is 60/100.
        min_area_to_shrink (int): Minimum area to shrink. Default is 125000.
        min_linear_area (int): Minimum linear area. Default is 600.
        min_dist_border (int): Minimum distance to the border. Default is 50.
        min_area_int (int): Minimum area of the cracks. Default is 2500.
        shrink_kernel (int): Kernel size for the shrink of the cracks. Default is 40.
        related_damages (dict): Dictionary with the related damages. 
        related_damages_key_wins (dict): Dictionary with the related damages.
        FR_dict (dict): Dictionary with the FRs.
    """
    def __init__(self, **kwargs):
        # initial setup
        self.unit = kwargs.get('unit', 'ft')
        self.distance_between_frames = kwargs.get('distance_between_frames', 20)
        self.frame_width = kwargs.get('frame_width', 24)
        self.y_top = kwargs.get('y_top', 350)
        # dilate and shrink
        self.kernel_size = kwargs.get('kernel_size', 5)
        self.iterations = kwargs.get('iterations', 1)
        # margins
        self.borders = kwargs.get('borders', ['FPAV','CUN','VEG','TRR','CAR'])
        # out transformation
        self.threshold_predominantly = kwargs.get('threshold_predominantly', 0.9)
        self.centroids_mindist = kwargs.get('centroids_mindist', 80)
        self.threshold_area = kwargs.get('threshold_area', 5600)
        self.abs_min_area = kwargs.get('abs_min_area', 1000)
        self.FB_min_area = kwargs.get('FB_min_area', 8000)
        self.FB_min_area2 = kwargs.get('FB_min_area2', 20000)
        self.FJ_min_length = kwargs.get('FJ_min_length', 250)
        self.min_area_crk_end = kwargs.get('min_area_crk_end', 300)
        self.PA_min_area_to_evaluate = kwargs.get('PA_min_area_to_evaluate', 62500)
        self.DC_min_area_to_evaluate = kwargs.get('DC_min_area_to_evaluate', 5600)
        self.DC_min_area_exclusion = kwargs.get('DC_min_area_exclusion', 160)
        self.DC_true_percentage = kwargs.get('DC_true_percentage', 0.70)
        self.square_ratio = kwargs.get('square_ratio', 3)
        self.BCH_min_area_to_evaluate = kwargs.get('BCH_min_area_to_evaluate', 5600)
        self.BCH_true_percentage = kwargs.get('BCH_true_percentage', 0.70)
        self.PCH_min_area = kwargs.get('PCH_min_area', 90000)
        self.consider_RMs = kwargs.get('consider_RMs', True)
        self.min_area = kwargs.get('min_area', 400)
        self.DSU_min_area = kwargs.get('DSU_min_area', 62500)
        self.thrRMarea = kwargs.get('thrRMarea', 25000)
        self.min_area_PCH = kwargs.get('min_area_PCH', 60000)
        self.crk_no_crk_percentage = kwargs.get('crk_no_crk_percentage', {2:.15, 3:.30})
        self.big_area = kwargs.get('big_area', ['DSU_1','DSU_2','DSU_3','PA_1','PA_2','PA_3'])
        self.threshold_percentage = kwargs.get('threshold_percentage', 60/100)
        self.min_area_to_shrink = kwargs.get('min_area_to_shrink', 125000)
        self.min_linear_area = kwargs.get('min_linear_area', 600)
        self.min_dist_border = kwargs.get('min_dist_border', 50)
        self.min_area_int = kwargs.get('min_area_int', 2500)
        self.shrink_kernel = kwargs.get('shrink_kernel', 40)
        self.related_damages = kwargs.get('related_damages', {
            'FJL_3':['FCL_3','FCL_2','FCL_1','FB_3','FB_2','FB_1','FL_3','FL_2','FL_1'],
            'PCH_2':['PCH_3','PC_2','PC_3','PC_1','FB_2','FB_3','FB_1'],
            'PCH_3':['PCH_2','PC_3','PC_2','PC_1','FB_3','FB_2','FB_1'],
        })
        self.related_damages_key_wins = kwargs.get('related_damages_key_wins', {
            'FB_1':['PC_1'],
            'FB_2':['PC_1','PC_2'],
            'FB_3':['PC_1','PC_2'],
        })
        self.FR_dict = kwargs.get('FR_dict', {
            'FR_1':['FB_1','FB_2','FL_2','FT_2','FL_1','FT_1','PC_1','FML_1'],
            'FR_2':['FB_2','FB_1','FB_3','FL_3','FT_3','FL_2','FT_2','PC_1','PC_2','FML_2'],
            'FR_3':['FB_3','FB_2','FL_3','FT_3','FL_2','FT_2','FL_1','FT_1','PC_2','PC_3','FML_3'],
            'FR_S':['FB_S','FL_S','FT_S','FL_1','FT_1','PC_S','FB_1','PC_1'],
        })
    
    def get_reports(self, out, converter):
        """
        This function is used to get the reports of the PMG model. RUN ALL with engin.ai labels changing at the end to converter
        
        Args: 
            out (array): Array with the predictions of the PMG model.
            converter (dict): Dictionary with the conversion of the labels.
            
        Returns:
            shapes (list): List with the shapes of the cracks.
            distressesFrameMap (list): List with the distresses of the cracks.
        """

        # dilate
        crack_outs, other_damage_outs = dilate_detections(out, CONV_LU_INV, self.kernel_size, iterations = self.iterations)

        # Define FRs: joinin with the closest one following a hirarchy list...
        crack_outs = mix_FR_damages(crack_outs,self.FR_dict)
        # mix same type
        crack_outs = mix_same_type(crack_outs, self.threshold_predominantly)
        # Small cracks are joined to biggest which are intersecting the delated area
        # crack_outs = combine_cracks(crack_outs, self.threshold_area, self.abs_min_area)
        # FJT and other damages can be joined if they are in touch with other cracks, following the hirarchy.
        crack_outs = join_related_damages(crack_outs,self.related_damages)
        # FBs could be PCs, so they are joined if they are in touch with PC, following the hirarchy.
        crack_outs = join_related_damages(crack_outs,self.related_damages_key_wins,key_dmg_wins=True)
        crack_outs = mix_same_type(crack_outs, self.threshold_predominantly)
        # if there are FB with small area, they can be join to intersected cracks
        crack_outs = join_small_FBs(crack_outs, self.FB_min_area)
        crack_outs = FB2FL(crack_outs, self.FB_min_area2)
        # clean smalls
        crack_outs = clean_outs_dict(crack_outs,self.min_area_crk_end)

        # pass PA to DSU
        other_damage_outs = PA2DSU(other_damage_outs)
        # mix same type
        other_damage_outs = mix_same_type(other_damage_outs, self.threshold_predominantly)
        # small PAs, DCs, and BCHs usually involved crack areas, if that's the case the area of PA can affect to the severities of the cracks.
        other_damage_outs, small_exPAs = evaluate_PAs(other_damage_outs, self.PA_min_area_to_evaluate)
        other_damage_outs, small_exDCs = evaluate_DCs(out, other_damage_outs, self.DC_min_area_to_evaluate, self.DC_min_area_exclusion, self.DC_true_percentage,self.square_ratio)
        other_damage_outs, small_exBCHs = evaluate_BCHs(out, other_damage_outs, self.BCH_min_area_to_evaluate, self.BCH_min_area_exclusion, self.BCH_true_percentage)
        # version 9 PCH requires an evaluation
        other_damage_outs = evaluate_PCHs(other_damage_outs, self.PCH_min_area)
        # check if pch is small enough to be joined to a crack
        crack_outs,other_damage_outs = check_PCHs(crack_outs,other_damage_outs,self.min_area_PCH)
        # cracks real area to perform analysis of areas
        crack_outs_ra = crack_outs_real_area(out, crack_outs)
        # analysis crack no crack
        small_exs = [small_exPAs,small_exDCs,small_exBCHs]
        # crack_outs_ra2, small_exs = crk_no_crk_analysis(crack_outs_ra.copy(), small_exs, self.crk_no_crk_percentage)
        # if crack_outs_ra2 != crack_outs_ra: # not yet, small false detections are causing wrong transformations. Predicted itself is better for now.
        #     crack_outs = real_area_2_outs(crack_outs_ra2,kernel_size)
        # add small dcs, rms to PA (only for dynates)
        other_damage_outs = PA_adding_smalls(other_damage_outs,small_exs,self.consider_RMs)
        # clean garbage
        other_damage_outs = clean_outs_dict(other_damage_outs,self.min_area)
        # Evaluate DSU
        other_damage_outs = evaluate_DSUs(other_damage_outs, self.DSU_min_area)
        # pass RMs to PA
        other_damage_outs = RMs2PA(other_damage_outs, self.thrRMarea)

        # get margen
        margen = get_margen(out, self.borders, self.y_top)
        t_area_px = np.sum(margen)
        # other dmgs big area
        other_damage_outs = big_area_analysis(out, other_damage_outs, margen, self.big_area, self.threshold_predominantly, self.centroids_mindist, self.threshold_percentage)

        # get results (shapes)
        # cracks_results = get_cracks_results(out, crack_outs, min_area_to_shrink, min_linear_area, min_dist_border,shrink_kernel)
        cracks_results = get_cracks_results2(out,crack_outs, other_damage_outs, margen, converter, self.min_linear_area, self.min_dist_border)
        # no_cracks_results = get_other_damages_results(other_damage_outs, min_area_int, min_dist_border)
        no_cracks_results = get_other_damages_results2(other_damage_outs,margen)


        ## Change label names
        shapes = cracks_results + no_cracks_results
        shapes = change_labels_shapes(shapes,converter)
        
        ## Dimensions report - for PCI
        height = out.shape[0] - self.y_top
        px2h = self.distance_between_frames / height
        px2w = (self.frame_width * self.distance_between_frames) / (t_area_px * px2h)
        distressesFrameMap = []
        for shape in shapes:
            area, length, counts = np.nan, np.nan, np.nan
            cont = np.array(shape['points'])
            x, y = cont[:,0] * px2w, cont[:,1] * px2h
            if shape['labelType'] == 'polyline':
                length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
            else:
                area = 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
            if '13' in shape['label']:
                counts = np.ceil(area / 5.5) # ASTM D6433
            distressesFrameMap.append({
                'label'     : shape['label'],
                'area_ft2'  : np.round(area,2),
                'length_ft' : np.round(length,2),
                'counts'    : np.round(counts,2),
            })
            
        return shapes, distressesFrameMap
        

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd._libs.tslibs.timestamps.Timestamp):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


def save_json(json_dict, out_path):
    out_file = open("myfile.json", "w")
    with open(out_path, "w") as outfile:
        json.dump(json_dict, outfile, indent=4, cls=NpEncoder)