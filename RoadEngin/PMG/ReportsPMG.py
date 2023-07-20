import json
import pandas as pd
import numpy as np
import cv2
import os
from RoadEngin.PMG.Post_processing import *


class PMG_postprocessing:
    unit = 'ft'
    distance_between_frames = 20
    frame_width = 24
    y_top = 250
    
    kernel_size = 20
    iterations = 1

    threshold_predominantly = 0.5
    centroids_mindist=80
    threshold_area = 5600
    abs_min_area = 1000
    FB_min_area = 8000
    FB_min_area2=20000
    FJ_min_length = 250
    min_area_crk_end = 300

    PA_min_area_to_evaluate = 62500
    DC_min_area_to_evaluate = 5600
    DC_min_area_exclusion = 160
    DC_true_percentage = 0.70
    square_ratio = 3
    BCH_min_area_to_evaluate = 5600
    BCH_min_area_exclusion = 400
    BCH_true_percentage = 0.70
    PCH_min_area = 90000
    consider_RMs=True
    min_area=400
    DSU_min_area = 62500
    thrRMarea = 25000
    min_area_PCH = 60000

    crk_no_crk_percentage = {2:.15, 3:.30}
    big_area = ['DSU_1','DSU_2','DSU_3','PA_1','PA_2','PA_3']
    threshold_percentage=60/100

    min_area_to_shrink = 125000
    min_linear_area = 600
    min_dist_border = 50
    min_area_int = 2500
    shrink_kernel = 40

    borders = ['FPAV','CUN','VEG','TRR']
    
    related_damages = { 'FJL_3':['FCL_3','FCL_2','FCL_1','FB_3','FB_2','FB_1','FL_3','FL_2','FL_1'],
                        'FJL_2':['FCL_2','FCL_1','FCL_3','FB_2','FB_1','FB_3','FL_2','FL_1','FL_3'],
                        'FJL_1':['FCL_1','FCL_2','FCL_3','FB_1','FB_2','FB_3','FL_1','FL_2','FL_3'],
                        'FJL_S':['FCL_S','FCL_1','FCL_2','FCL_3','FB_S','FB_1','FB_2','FB_3','FL_S','FL_1','FL_2','FL_3'],
                        'FJT_3':['FCT_3','FCT_2','FCT_1','FB_3','FB_2','FB_1','FT_3','FT_2','FT_1'],
                        'FJT_2':['FCT_2','FCT_1','FCT_3','FB_2','FB_1','FB_3','FT_2','FT_1','FT_3'],
                        'FJT_1':['FCT_1','FCT_2','FCT_3','FB_1','FB_2','FB_3','FT_1','FT_2','FT_3'],
                        'FJT_S':['FCT_S','FCT_1','FCT_2','FCT_3','FB_S','FB_1','FB_2','FB_3','FT_S','FT_1','FT_2','FT_3'],
                        'PCH_2':['PCH_3','PC_2','PC_3','PC_1','FB_2','FB_3','FB_1'],
                        'PCH_3':['PCH_2','PC_3','PC_2','PC_1','FB_3','FB_2','FB_1'],
                    }

    related_damages_key_wins = {# join small cracks can replace this when there is more info
        'PC_1':['FB_1','FB_2'],
        'PC_2':['FB_2','FB_1','FB_3'],
        'PC_3':['FB_3','FB_2'],
        'FCL_1':['FB_1','FB_2'],
        'FCL_2':['FB_2','FB_1','FB_3'],
        'FCL_3':['FB_3','FB_2','FB_1'],
        'FCL_S':['FB_1','FB_2','FB_S'],
    }

    FR_dict = { 'FR_1':['FB_1','FB_2','FL_2','FT_2','FL_1','FT_1','PC_1','FML_1'],
                'FR_2':['FB_2','FB_1','FB_3','FL_3','FT_3','FL_2','FT_2','PC_1','PC_2','FML_2'],
                'FR_3':['FB_3','FB_2','FL_3','FT_3','FL_2','FT_2','FL_1','FT_1','PC_2','PC_3','FML_3'],
                'FR_S':['FB_S','FL_S','FT_S','FL_1','FT_1','PC_S','FB_1','PC_1']}
    
    def set_dilate_params(self,kernel_size,iterations) -> None:
        self.kernel_size = kernel_size
        self.iterations = iterations
        
    def set_borders(self,borders:list):
        self.borders = borders
        
    def set_frame_dimension(self, frame_length = distance_between_frames, frame_width = frame_width, unit = unit, ytop_ROI = y_top):
        self.distance_between_frames = frame_length
        self.frame_width = frame_width
        self.unit = unit
        self.y_top = ytop_ROI
        
    def get_reports(self, out, converter):
        '''
        RUN ALL with engin.ai labels changing at the end to converter
        out: inference array
        returns the shapes and distresses dictionary with dimensions.
        '''

        # dilate
        crack_outs, other_damage_outs = dilate_detections(out, CONV_LU_INV, self.kernel_size, iterations = self.iterations)

        # Define FRs: joinin with the closest one following a hirarchy list...
        crack_outs = mix_FR_damages(crack_outs,self.FR_dict)
        # mix same type
        crack_outs = mix_same_type(crack_outs, self.threshold_predominantly)
        # Small cracks are joined to biggest which are intersecting the delated area
        crack_outs = combine_cracks(crack_outs, self.threshold_area, self.abs_min_area)
        # FJT and other damages can be joined if they are in touch with other cracks, following the hirarchy.
        crack_outs = join_related_damages(crack_outs,self.related_damages)
        # FBs could be PCs, so they are joined if they are in touch with PC, following the hirarchy.
        crack_outs = join_related_damages(crack_outs,self.related_damages_key_wins,key_dmg_wins=True)
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
    
        
    
class PMG_project:

    project = {}

    def __init__(self, swin_model, unit = 'ft', distance_between_frames = 20):
        self.unit = unit
        self.dbf = distance_between_frames

    def load_project(self, xls, ):
        df_xls = xls if type(xls) == pd.core.frame.DataFrame else pd.read_excel(xls)

    def load_project_from_dict(self, project_info: dict):
        number_of_videos = len(project_info.keys())
        number_of_frames = np.sum([len(project_info[v]['frames']) for v in project_info.keys()])
        if number_of_videos > 0 and number_of_frames > number_of_videos:
            print(f'Project loaded succesfully! \n{number_of_videos} videos.\n{number_of_frames} frames.')
            self.project = project_info
        else:
            print('Missing videos and/or frames in the dictionary.')

    def frame_length(self):
        print(self.distance_between_frames,self.unit)
        

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