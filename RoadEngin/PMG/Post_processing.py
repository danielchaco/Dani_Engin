import pandas as pd
import numpy as np
import cv2
import os
from RoadEngin.utils.constants import *


def get_contours(img):
    kernel = np.ones((5, 5), np.uint8)
    ret, thresh = cv2.threshold(img.astype(np.uint8), 0.3, 1.01, 0)
    img_dilated = cv2.dilate(thresh, kernel, iterations=1)

    cnts = cv2.findContours(img_dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    return cnts


def clean_outs_dict(outs_dict: dict, min_area=10):
    return {damage: out_dict for damage, out_dict in outs_dict.items() if np.sum(out_dict != 0) > min_area}


def get_damages(out, CONV_LU_INV: dict):
    '''
    returns engin label names reported in out.
    CONV_LU_INV: inverse of CONV_LU dictionary
    '''
    return [CONV_LU_INV[i] for i in np.unique(out) if i != 0]


def get_centroid(cont):
    '''
    returns coordinates of the centroid
    '''
    M = cv2.moments(cont)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return cx, cy


def get_main_centroid(conts):
    areas, cxs, cys = [], [], []
    for cont in conts:
        areas.append(cv2.contourArea(cont))
        cx, cy = get_centroid(cont)
        cxs.append(cx)
        cys.append(cy)
    return int(np.dot(areas, cxs) / np.sum(areas)), int(np.dot(areas, cys) / np.sum(areas))


def get_main_cx(conts):
    areas, cxs = [], []
    for cont in conts:
        areas.append(cv2.contourArea(cont))
        cx, _ = get_centroid(cont)
        cxs.append(cx)
    return int(np.dot(areas, cxs) / np.sum(areas))


def array_in_list(a1, lista):
    '''
    To find if a contours is in a list of contours.
    '''
    for a2 in lista:
        if np.array_equal(a1, a2):
            return True
    return False


def get_dist_of_two_regions(conts1, conts2):
    cx1, cy1 = get_main_centroid(conts1)
    cx2, cy2 = get_main_centroid(conts2)
    return ((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2) ** .5


def dilate_detections(out, CONV_LU_INV, kernel_size=30, iterations=1):
    '''
    Returns two dictionaries with cracks and no cracks damage detections. The
    values will be binary masks of 0 and 1 of the dilating results.
    out: detections reshaped in a proper size (img.shape)
    CONV_LU_INV: dictionary to provide the engin label from detections
    kernel_size: the size in pixels of the kernel to dilate.
    iterations: number of dilatings will be performed.
    '''

    out_ids = np.unique(out)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    crack_outs, other_damage_outs = {}, {}
    for out_id in out_ids:
        damage = CONV_LU_INV[out_id]
        if damage in CRACKS:
            dmg_out = np.array(out == out_id, dtype=np.uint8)
            dilated = cv2.dilate(dmg_out, kernel, iterations=iterations)
            if damage in crack_outs.keys():
                crack_outs[damage] = np.array(crack_outs[damage] + dilated != 0,dtype=np.uint8)
            else:
                crack_outs[damage] = dilated
        elif damage in OTHER_DISTRESS:
            dmg_out = np.array(out == out_id, dtype=np.uint8)
            dilated = cv2.dilate(dmg_out, kernel, iterations=iterations)
            if damage in other_damage_outs.keys():
                other_damage_outs[damage] = np.array(other_damage_outs[damage] + dilated != 0,dtype=np.uint8)
            else:
                other_damage_outs[damage] = dilated

    return crack_outs, other_damage_outs


def bdamage_in_damages(damages, bdamage='FR_'):
    '''
    look up function. returns a list of findings.
    '''
    return [damage for damage in damages if bdamage in damage]


def bdamages_in_damages(damages, bdamages=['FR_', 'FJ']):
    if type(bdamages) == list:
        return [bdamage_in_damages(damages, bdamage) for bdamage in bdamages]
    elif type(bdamages) == str:
        return bdamage_in_damages(damages, bdamages)


def mix_FR_damages(crack_outs, FR_dict: dict):
    '''
    Mix FR detections with other cracks following a hirarchy listed in FR_dict.
    Other cracks can be mixed with the other cracks.
    Isolated FR cracks will remain as FRs for further analysis.
    Returns crack_outs with the integration of FRs.
    This function should be excecuted if len(FRs) > 0.
    '''
    FRs = bdamage_in_damages(crack_outs.keys(), 'FR_')
    if len(FRs) > 0:
        new_FR_dict = {}
        cracks = crack_outs.keys()
        for fr, hirarchy in FR_dict.items():
            other_cracks = [element for element in cracks if element not in hirarchy and element not in FRs]
            if fr in FRs:
                new_FR_dict[fr] = hirarchy + other_cracks
        FR_dict = new_FR_dict
        for FR in FRs:
            for damage in FR_dict[FR]:
                if damage in cracks and np.sum(crack_outs[FR] != 0) > 10:
                    out_comb = crack_outs[damage] + crack_outs[FR]
                    conts = get_contours(np.array(out_comb != 0))
                    for cont in get_contours(crack_outs[FR]):
                        if not array_in_list(cont, conts):
                            crack_outs[FR] = cv2.fillPoly(crack_outs[FR], pts=[cont], color=0)
                    out_comb[crack_outs[FR] != 0] = 0
                    crack_outs[damage] = out_comb
    return clean_outs_dict(crack_outs)


def mix_same_type(outs_dict, threshold_predominantly=0.6, centroids_mindist=40):
    '''
    combine severity of same type of damages based on predominantly of area > 60%
    or major area in case of centroids ara close each other 16cm ~ 40px.
    if areas are about 41 to 59 %, damages won't be mixed.
    '''
    if len(outs_dict.keys()) > 0:
        df = pd.DataFrame(outs_dict.keys(), columns=['label'])
        df[['typ', 'sev']] = [df.label[i].split('_') if len(df.label[i].split('_')) == 2 else [df.label[i], np.nan] for
                              i in df.index]
        df.sort_values(['typ', 'sev'], ascending=False, inplace=True, ignore_index=True)

        df_g = df.groupby('typ').count()
        for typ in df_g[df_g.label > 1].index:
            damages = df[df.typ == typ].label.to_list()
            for i, d1 in enumerate(damages):
                if i != len(damages):
                    for d2 in damages[i + 1:]:
                        out_comb = np.array(outs_dict[d1] != 0) * 1 + np.array(outs_dict[d2] != 0) * 2
                        conts = get_contours(out_comb)
                        for cont in conts:
                            stencil = np.zeros(out_comb.shape, dtype=np.uint8)  # .astype(img.dtype)
                            stencil = cv2.fillPoly(stencil, [cont], 1)
                            out_temp = out_comb * stencil
                            if len(np.unique(out_temp)) > 2:
                                if np.sum(out_temp == 1) / np.sum(
                                        (out_temp == 2) | (out_temp == 1)) >= threshold_predominantly:
                                    outs_dict[d1] = cv2.fillPoly(outs_dict[d1], pts=[cont], color=1)
                                    outs_dict[d2] = cv2.fillPoly(outs_dict[d2], pts=[cont], color=0)
                                elif np.sum(out_temp == 2) / np.sum(
                                        (out_temp == 2) | (out_temp == 1)) >= threshold_predominantly:
                                    outs_dict[d1] = cv2.fillPoly(outs_dict[d1], pts=[cont], color=0)
                                    outs_dict[d2] = cv2.fillPoly(outs_dict[d2], pts=[cont], color=1)
                                elif get_dist_of_two_regions(get_contours((out_temp == 1) | (out_temp == 3)),
                                                             get_contours((out_temp == 2) | (
                                                                     out_temp == 3))) < centroids_mindist:
                                    if np.sum(out_temp == 1) > np.sum(out_temp == 2):
                                        outs_dict[d1] = cv2.fillPoly(outs_dict[d1], pts=[cont], color=1)
                                        outs_dict[d2] = cv2.fillPoly(outs_dict[d2], pts=[cont], color=0)
                                    if np.sum(out_temp == 2) > np.sum(out_temp == 2):
                                        outs_dict[d1] = cv2.fillPoly(outs_dict[d1], pts=[cont], color=0)
                                        outs_dict[d2] = cv2.fillPoly(outs_dict[d2], pts=[cont], color=1)

    return clean_outs_dict(outs_dict)


def combine_cracks(crack_outs, threshold_area=4000, abs_min_area=625):
    '''
    combine small cracks with with biggest ones. absolute min area damages will be dropped.
    '''
    data = []
    for damage, crack_out in crack_outs.items():
        conts = get_contours(crack_out)
        for cont in conts:
            data.append([damage, cont, cv2.contourArea(cont)])

    df = pd.DataFrame(data, columns=['damage', 'cont', 'area'])
    df.sort_values('area', inplace=True, ignore_index=True, ascending=False)
    bigger_cracks = df[df.area >= threshold_area].damage.unique()
    small_out = np.zeros(crack_out.shape, dtype=np.uint8)

    for i in df[df.area < threshold_area].index:
        cont = df.cont[i]
        crack_outs[df.damage[i]] = cv2.fillPoly(crack_outs[df.damage[i]], pts=[cont], color=0)
        cv2.fillPoly(small_out, pts=[cont], color=1)

    for big_crack in bigger_cracks:
        out_comb = crack_outs[big_crack] + small_out
        conts = get_contours(out_comb)
        small_conts = get_contours(small_out)
        if len(small_conts) > 0:
            for small_cont in small_conts:
                if not array_in_list(small_cont, conts):
                    crack_outs[big_crack] = cv2.fillPoly(crack_outs[big_crack], pts=[small_cont], color=1)
                    cv2.fillPoly(small_out, pts=[small_cont], color=0)
        else:
            break

    if np.sum(small_out != 0) > abs_min_area:
        for i in df[(df.area < threshold_area) & (df.area > abs_min_area)].index:
            cont = df.cont[i]
            stencil = np.zeros(crack_out.shape, dtype=np.uint8)
            stencil = cv2.fillPoly(stencil, [cont], 1)
            out_temp = small_out * stencil
            if np.sum(out_temp) > 0:
                crack_outs[df.damage[i]] = cv2.fillPoly(crack_outs[df.damage[i]], pts=[cont], color=1)

    return clean_outs_dict(crack_outs)


def join_related_damages(outs_dict, related_damages, key_dmg_wins = False):
    '''
    Join related damages due to common errors in version 9 - bigger one wins if
    key_dmg_wins is set False
    '''
    for d1 in related_damages.keys():
        if d1 in outs_dict.keys():
            for d2 in related_damages[d1]:
                if d2 in outs_dict.keys():
                    out_comb = np.array(outs_dict[d1] != 0) * 1 + np.array(outs_dict[d2] != 0) * 2
                    conts = get_contours(out_comb)
                    for cont in conts:
                        stencil = np.zeros(out_comb.shape, dtype=np.uint8)
                        stencil = cv2.fillPoly(stencil, [cont], 1)
                        out_temp = out_comb * stencil
                        if len(np.unique(out_temp)) > 2:
                            if key_dmg_wins:
                                outs_dict[d1] = cv2.fillPoly(outs_dict[d1], pts=[cont], color=1)
                                outs_dict[d2] = cv2.fillPoly(outs_dict[d2], pts=[cont], color=0)
                            else:
                                if np.sum(out_temp == 1) > np.sum(out_temp == 2):
                                    outs_dict[d1] = cv2.fillPoly(outs_dict[d1], pts=[cont], color=1)
                                    outs_dict[d2] = cv2.fillPoly(outs_dict[d2], pts=[cont], color=0)
                                elif np.sum(out_temp == 2) > np.sum(out_temp == 1):
                                    outs_dict[d1] = cv2.fillPoly(outs_dict[d1], pts=[cont], color=0)
                                    outs_dict[d2] = cv2.fillPoly(outs_dict[d2], pts=[cont], color=1)
    return clean_outs_dict(outs_dict)



def join_small_FBs(crack_outs, FB_min_area=3750):
    '''
    Small FBs: join them to other cracks... area<30cm*20cm ~ 3750.0 px2
    '''
    FBs = bdamage_in_damages(crack_outs.keys(), 'FB_')
    if len(FBs) > 0:
        priority = ['FB_1', 'FB_2', 'FB_3', 'PC_1', 'PC_2', 'PC_3', 'FT_1', 'FT_2', 'FT_3', 'FL_1', 'FL_2', 'FL_3']
        priority = [value for value in priority if value in crack_outs.keys()]
        no_priority = [damage for damage in crack_outs.keys() if damage not in priority]

        for FB in FBs:
            conts_fb = get_contours(crack_outs[FB])
            for cont_fb in conts_fb:
                if cv2.contourArea(cont_fb) < FB_min_area:
                    stencil = np.zeros(crack_outs[FB].shape, dtype=np.uint8)
                    stencil = cv2.fillPoly(stencil, [cont_fb], 1)
                    for damage in priority + no_priority:
                        if damage != FB:
                            out_damage = crack_outs[damage]
                            if np.sum(out_damage * stencil) > 0:
                                crack_outs[damage] = cv2.fillPoly(crack_outs[damage], [cont_fb], 1)
                                crack_outs[FB] = cv2.fillPoly(crack_outs[FB], [cont_fb], 0)
                                break
    return clean_outs_dict(crack_outs)


def evaluate_FJs(crack_outs, FJ_min_length=250):
    '''
    FJT, FJL: if it is touching FB, move it to FB detection (version 9 tunned), if the lenght is small,
    take it as FL or FT. Min length 1 m ~ 250
    This is due to version 9.0 and tunned
    '''
    FJs = bdamage_in_damages(crack_outs.keys(), 'FJ_')
    if len(FJs) > 0:
        for FJ in FJs:
            out_FJ = crack_outs[FJ]
            conts = get_contours(out_FJ)
            for cont in conts:
                _, _, w, h = cv2.boundingRect(cont)
                if max(w, h) < FJ_min_length:
                    crack_outs[FJ] = cv2.fillPoly(crack_outs[FJ], [cont], 0)
                    new_label = FJ.replace('J', '')
                    if new_label not in crack_outs.keys():
                        crack_outs[new_label] = np.zeros(out_FJ.shape, dtype=np.uint8)
                    crack_outs[new_label] = cv2.fillPoly(crack_outs[new_label], [cont], 1)
    return clean_outs_dict(crack_outs)


def dropPA1(other_damage_outs):
    '''
    After first revision, version 9.2 wrongly detects PA_1. This will require
    more attention, in the meantime they will be just dropped at the begining.
    '''
    if 'PA_1' in other_damage_outs.keys():
        del other_damage_outs['PA_1']
    return other_damage_outs


def PA2DSU(other_damage_outs):
    '''
    For the PMG project of 3/14/2023 inference showed PA detections when pavement
    has DSU.
    '''
    for label in other_damage_outs.keys():
        if 'PA' in label:
            if 'DSU_'+label.split('_')[-1] in other_damage_outs.keys():
                other_damage_outs['DSU_'+label.split('_')[-1]] += other_damage_outs[label]
                other_damage_outs[label] = other_damage_outs[label]*0
    return clean_outs_dict(other_damage_outs)


def evaluate_PAs(other_damage_outs, PA_min_area_to_evaluate=3750):
    '''
    small PAs: can affect cracks severity or can be joined to PAs DSU. small_exPAs for further analysis.
    20cm x 30xm ~ 3750 px2
    returns an update of other_damage_outs and an array of small PAs which can not count as DCs for further analysis.
    '''
    PAs = bdamage_in_damages(other_damage_outs.keys(), 'PA_')
    if len(PAs) > 0:
        small_exPAs = np.zeros(other_damage_outs[PAs[0]].shape, dtype=np.uint8)
        for PA in PAs:
            out_PA = other_damage_outs[PA]
            conts = get_contours(out_PA)
            for cont in conts:
                _, _, w, h = cv2.boundingRect(cont)
                area = cv2.contourArea(cont)
                if w * h < PA_min_area_to_evaluate:
                    cv2.fillPoly(small_exPAs, pts=[cont], color=1)
                    cv2.fillPoly(out_PA, pts=[cont], color=0)
            other_damage_outs[PA] = out_PA

        return other_damage_outs, small_exPAs
    else:
        return other_damage_outs, []


def evaluate_DCs(out, other_damage_outs, DC_min_area_to_evaluate=5600, DC_min_area_exclusion=160,
                 DC_true_percentage=0.70, square_ratio=3):
    '''
    small DCs: can affect cracks severity or can be joined to PAs this will be
    stored in small_exDCs for further analysis.

    DCs to evaluate are those which:
    * its area is less than 30cm x 30cm ~ 5600 px2.
    * its shape is like a circle, in this case a circle inside a square will get
    ~78% of the area, so it is proposed to use 70% as threshold.
    * To complement the statement above it is added a square_ratio limit of 3
    where DCs can be rectangular, after that DCs behaves like thick cracks.
    * DCs overlaping crack areas will also be consider for further analysis.

    Small DCs: with areas less than 5cm x 5cm ~ 160 px2 will be dropped - v9.2

    big DCs: can be joined to potholes or can count as potholes depending on the
    area and methodology, which requires another functions to determine it.

    To run this function Real or detected area is used instead of dilated.

    Returns the update of other_damage_outs, and an array of small DCs which can
    not count as DCs and are used for further analysis (crack_no_crack_analysis).
    '''
    small_exDCs = np.zeros(out.shape, dtype=np.uint8)
    DCs = bdamage_in_damages(other_damage_outs.keys(), 'DC_')
    if len(DCs) > 0:
        cracks_out = np.array(np.isin(out, CRACKS_ID), dtype=np.uint8)
        for DC in DCs:
            out_DC = np.array(out == CONV_LU[DC], dtype=np.uint8)  # *1
            conts = get_contours(out_DC)
            for cont in conts:
                _, _, w, h = cv2.boundingRect(cont)
                area = cv2.contourArea(cont)
                stencil = np.zeros(out.shape, dtype=np.uint8)
                stencil = cv2.fillPoly(stencil, [cont], 1)
                out_temp = cracks_out * stencil
                if (w * h < DC_min_area_to_evaluate) and (
                        area < DC_min_area_exclusion or area / (w * h) < DC_true_percentage or max(w, h) / min(w,
                                                                                                               h) > square_ratio or np.sum(
                    out_temp) > 10):
                    cv2.fillPoly(small_exDCs, pts=[cont], color=1)
                    cv2.fillPoly(out_DC, pts=[cont], color=0)
            other_damage_outs[DC] = out_DC

    return clean_outs_dict(other_damage_outs), small_exDCs


def evaluate_BCHs(out, other_damage_outs, BCH_min_area_to_evaluate=5600, BCH_min_area_exclusion=400,
                  BCH_true_percentage=0.70):
    '''
    Similar to evaluate_BCHs.

    To run this function Real area is used instead of dilated.

    Returns the update of other_damage_outs, and an array of small BCHs which can
    not count as BCHs for further analysis.
    '''
    small_exBCHs = np.zeros(out.shape, dtype=np.uint8)
    BCHs = bdamage_in_damages(other_damage_outs.keys(), 'BCH_')
    if len(BCHs) > 0:
        for BCH in BCHs:
            out_BCH = np.array(out == CONV_LU[BCH], dtype=np.uint8) * 1
            conts = get_contours(out_BCH)
            for cont in conts:
                _, _, w, h = cv2.boundingRect(cont)
                area = cv2.contourArea(cont)
                if (area < BCH_min_area_exclusion) or (
                        w * h < BCH_min_area_to_evaluate and area / (w * h) < BCH_true_percentage):
                    cv2.fillPoly(small_exBCHs, pts=[cont], color=1)
                    cv2.fillPoly(out_BCH, pts=[cont], color=0)
            other_damage_outs[BCH] = out_BCH

    return clean_outs_dict(other_damage_outs), small_exBCHs


def evaluate_PCHs(other_damage_outs, PCH_min_area=90000):
    '''
    Version 9 is not detecting patching pretty well, so for small areas they will be removed: w * h < 30cm x 50cm ~ 9000 px2,
    and bigger areas bbox will be set as the correct dimensions.
    No crack should be reported over patches, but if they are, they can affect its severity.

    Returns the update of other_damage_outs.
    '''
    PCHs = bdamage_in_damages(other_damage_outs.keys(), 'PCH_')
    if len(PCHs) > 0:
        for PCH in PCHs:
            out_PCH = other_damage_outs[PCH]
            conts = get_contours(out_PCH)
            for cont in conts:
                _, _, w, h = cv2.boundingRect(cont)
                if w * h < PCH_min_area:
                    cv2.fillPoly(out_PCH, pts=[cont], color=0)
            other_damage_outs[PCH] = out_PCH

    return clean_outs_dict(other_damage_outs)


def check_PCHs(crack_outs,other_damage_outs,min_area_PCH):
    '''
    Swan is not good with PCHs, infering sealed cracks as borders of PCH
    '''
    PCHs = bdamage_in_damages(other_damage_outs.keys(), 'PCH_')
    if len(PCHs) > 0:
        for PCH in PCHs:
            out_PCH = other_damage_outs[PCH]
            if np.sum(out_PCH!=0) < min_area_PCH:
                for crk, out_crk in crack_outs.items():
                    out_com = out_PCH + out_crk*2
                    conts = get_contours(out_com)
                    for cont in conts:
                        stencil = np.zeros(out_com.shape,np.uint8)
                        stencil = cv2.fillPoly(stencil, pts=[cont], color=1) * out_com
                        if np.sum(stencil==1) < np.sum(stencil==2):
                            other_damage_outs[PCH] = cv2.fillPoly(out_PCH, pts=[cont], color=0)
                            crack_outs[crk] = cv2.fillPoly(out_crk, pts=[cont], color=1)
    return crack_outs,other_damage_outs



def evaluate_DSUs(other_damage_outs, DSU_min_area=9000):
    '''
    Version 9 leaves some areas of DSU that can be removed.
    Similar to patching analysis.

    Returns the update of other_damage_outs.
    '''
    DSUs = bdamage_in_damages(other_damage_outs.keys(), 'DSU_')
    if len(DSUs) > 0:
        for DSU in DSUs:
            out_DSU = other_damage_outs[DSU]
            conts = get_contours(out_DSU)
            for cont in conts:
                _, _, w, h = cv2.boundingRect(cont)
                if w * h < DSU_min_area:
                    cv2.fillPoly(out_DSU, pts=[cont], color=0)
            other_damage_outs[DSU] = out_DSU

    return clean_outs_dict(other_damage_outs)


def remove_RMs(other_damage_outs):
    '''
    version 9.2 detects RMs were they aren't, so after PA analysis, RMs can be
    dropped. PA analysis is performed by PA_adding_smalls.
    '''
    RMs = bdamage_in_damages(other_damage_outs.keys(), 'RM')
    if len(RMs) > 0:
        for RM in RMs:
            del other_damage_outs[RM]

    return other_damage_outs


def RMs2PA(other_damage_outs, thrRMarea = 25000):
    '''
    PCI method consideres RMs as PA midium severity.
    '''
    RMs = bdamage_in_damages(other_damage_outs.keys(), 'RM')
    if len(RMs) > 0:
        for RM in RMs:
            out_RM = other_damage_outs[RM]
            zeros = np.zeros(out_RM.shape, dtype=np.uint8)
            conts = get_contours(out_RM)
            for cont in conts:
                if cv2.contourArea(cont) > thrRMarea:
                    if 'PA_3' in other_damage_outs.keys():
                        other_damage_outs['PA_3'] += cv2.fillPoly(zeros.copy(), pts=[cont], color=1)
                    else:
                        other_damage_outs['PA_3'] = cv2.fillPoly(zeros.copy(), pts=[cont], color=1)
                else:
                    if 'PA_2' in other_damage_outs.keys():
                        other_damage_outs['PA_2'] += cv2.fillPoly(zeros.copy(), pts=[cont], color=1)
                    else:
                        other_damage_outs['PA_2'] = cv2.fillPoly(zeros.copy(), pts=[cont], color=1)
    return other_damage_outs


def crack_outs_real_area(out, crack_outs):
    '''
    returns a dictionary of damages with real area or out areas in a list of arrays of independet damages
    with same type of typology and severity.
    '''
    cracks_out = np.array(np.isin(out, CRACKS_ID), dtype=np.uint8)
    new_crack_outs = {}
    for damage in crack_outs.keys():
        conts = get_contours(crack_outs[damage])
        damage_list = []
        for cont in conts:
            stencil = np.zeros(out.shape, dtype=np.uint8)
            stencil = cv2.fillPoly(stencil, [cont], 1)
            real_out = cracks_out * stencil
            damage_list.append(np.array(real_out != 0, dtype=np.uint8))
        new_crack_outs[damage] = damage_list
    return new_crack_outs


def real_area_2_outs(crack_outs_ra, kernel_size=None):
    '''
    temporal function until define the reports and the whole pipeline.
    rearrenge to crack_outs order, crack_outs_real_area inverse function.
    to use after crack_no_crack analysis, if needed.
    '''
    crack_outs = {}
    for label, outs in crack_outs_ra.items():
        dmg_out = np.array(np.sum(crack_outs_ra[label], axis=0) != 0, dtype=np.uint8)
        if kernel_size:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            dilated = cv2.dilate(dmg_out, kernel, iterations=1)
            crack_outs[label] = dilated
        else:
            crack_outs[label] = dmg_out
    return crack_outs


def get_big_bbox(df_bboxs):
    x_min = df_bboxs.x.min()
    y_min = df_bboxs.y.min()
    x_max = (df_bboxs.x + df_bboxs.w).max()
    y_max = (df_bboxs.y + df_bboxs.h).max()
    w = x_max - x_min
    h = y_max - y_min
    return [x_min, y_min, w, h]


def centroid_isin(centroid, bbox):
    x_min, y_min, w, h = bbox
    cx, cy = centroid
    return cx > x_min and cx < x_min + w and cy > y_min and cy < y_min + h


def crk_no_crk_analysis(crack_outs_ra, small_exs, crk_no_crk_percentage={2: .15, 3: .30}):
    '''
    damages such as PA, DC, PCH can affect crack severities. This function allows to analyze if the damage
    matches a cracked area, then check the severity in terms of comparison of areas. If PA area is more than
    20cm x 20cm and does not match cracked area, it will return to the small_exPAs array.
    DCs that do not match will be analyzed with PA areas.
    To performe this analyzis, real area will help more than dilated.

    small_exs: could be a list of arrays of small damages that affect cracks or an array in out format.
    crk_no_crk_percentage: is the percentage to consider if the damage will be affected. {2:.15, 3:.3}

    returns: update of crack_outs_ra and small_exs
    '''

    if type(small_exs) != list:
        small_ex_out = small_exs.copy()
    else:
        small_ex_out = np.zeros(small_exs[0].shape, dtype=np.uint8)
        for small_ex in small_exs:
            small_ex_out[small_ex != 0] = 1

    area_all_small_exs = np.sum(small_ex_out != 0)
    if area_all_small_exs != 0:
        cont_exs = get_contours(small_ex_out.astype(np.uint8))
        centroids = []
        for cont_ex in cont_exs:
            centroids.append(get_centroid(cont_ex))

        new_crack_outs_ra = {}  # crack_outs_ra.copy
        for label_name in crack_outs_ra.keys():
            typ, severity = label_name.split('_')
            for j, damage in enumerate(crack_outs_ra[label_name]):
                out_damage = damage.copy()
                area_damage = np.sum(out_damage != 0)
                chng = False
                if area_all_small_exs / area_damage > crk_no_crk_percentage[2]:
                    conts = get_contours(out_damage)
                    bboxs = []
                    for con in conts:
                        bboxs.append(cv2.boundingRect(con))
                    df_bboxs = pd.DataFrame(bboxs, columns=['x', 'y', 'w', 'h'])
                    big_bbox = get_big_bbox(df_bboxs)

                    area_ex = 0
                    for i, centroid in enumerate(centroids):
                        cx, cy = centroid
                        if centroid_isin(centroid, big_bbox):
                            cv2.fillPoly(small_ex_out, [cont_exs[i]], 0)
                            cv2.fillPoly(out_damage, [cont_exs[i]], 1)
                            area_ex += cv2.contourArea(cont_exs[i])
                    if area_ex / area_damage > crk_no_crk_percentage[3]:
                        # damage should be severity 3
                        if severity != 3:
                            new_label = typ + '_3'
                            chng = True
                    elif area_ex / area_damage > crk_no_crk_percentage[2]:
                        # damage should be severity 2 or 3
                        if severity not in [2, 3]:
                            new_label = typ + '_2'
                            chng = True

                    if chng:
                        try:
                            new_crack_outs_ra[new_label] += [out_damage]
                        except:
                            new_crack_outs_ra[new_label] = [out_damage]
                    else:
                        try:
                            new_crack_outs_ra[label_name] += [out_damage]
                        except:
                            new_crack_outs_ra[label_name] = [out_damage]

        if type(small_exs) != list:
            small_exs[small_ex_out == 0] = 0
        else:
            for i, small_ex in enumerate(small_exs):
                small_ex[small_ex_out == 0] = 0
                small_exs[i] = small_ex

        return new_crack_outs_ra, small_exs
    else:
        return crack_outs_ra, small_exs


def PA_adding_smalls(other_damage_outs, small_exs, consider_RMs=True):
    '''
    RMs, small_exs after crk_no_crk_analysis... can be join the PA areas or form new ones.
    '''
    PAs, RMs = bdamages_in_damages(other_damage_outs.keys(), ['PA_', 'RM'])

    if len(PAs) > 0:

        if type(small_exs) != list:
            small_ex_out = small_exs.copy()
        else:
            small_ex_out = np.zeros(small_exs[0].shape, dtype=np.uint8)
            for small_ex in small_exs:
                small_ex_out[small_ex != 0] = 1
        if len(RMs) > 0 and consider_RMs:
            for RM in RMs:
                small_ex_out = other_damage_outs[RM] + small_ex_out

        cont_exs = get_contours(small_ex_out)

        for PA in PAs:
            out_com = other_damage_outs[PA] + small_ex_out
            conts_com = get_contours(out_com)
            for cont in cont_exs:
                if not array_in_list(cont, conts_com):
                    other_damage_outs[PA] = cv2.fillPoly(other_damage_outs[PA], pts=[cont], color=1)
                    cv2.fillPoly(small_ex_out, pts=[cont], color=0)

    return other_damage_outs


def check_bbox_limits(bbox, lld, rld, shape, min_dist_border):
    x, y, w, h = bbox
    x2 = x + w
    y2 = y + h
    x = lld if x - lld <= min_dist_border else x
    y = 0 if y <= min_dist_border else y
    x2 = rld if rld - x2 <= min_dist_border else x2
    y2 = shape[0] if shape[0] - y2 <= min_dist_border else y2
    return x, y, x2 - x, y2 - y


def get_bbox_points(bbox):
    x, y, w, h = bbox
    return [[x, y], [x, y + h], [x + w, y + h], [x + w, y]]


def get_intersection_point(point1, point2, bbox, side='top'):
    '''
    side: top, bottom, left, right
    '''
    x, y, w, h = bbox
    m = (point2[0] - point1[0]) / (point2[1] - point1[1])
    b = point1[1] - m * point1[0]
    y1 = int(m * x + b)
    y2 = int(m * (x + w) + b)
    x1 = int((y - b) / m)
    x2 = int((y + h - b) / m)
    if side == 'top':
        if x1 >= x and x1 <= x + w:
            return x1, y
        elif y1 >= y and y1 <= point1[1]:
            return x, y1
        elif y2 >= y and y2 <= point1[1]:
            return x + w, y2
    elif side == 'bottom':
        if x2 >= x and x2 <= x + w:
            return x2, y + h
        elif y1 <= y + h and y1 >= point1[1]:
            return x, y1
        elif y2 <= y + h and y2 >= point1[1]:
            return x + w, y2
    elif side == 'left':
        if y1 >= y and y1 <= y + h:
            return x, y1
        elif x1 >= x and x1 <= point1[0]:
            return x1, y
        elif x2 >= x and x1 <= point1[0]:
            return x2, y + h
    elif side == 'right':
        if y2 >= y and y2 <= y + h:
            return x + w, y2
        elif x1 <= x + w and x1 >= point1[0]:
            return x1, y
        elif x2 <= x + w and x2 >= point1[0]:
            return x2, y + h


def get_polyline(out_damage, bbox, square_length=15, first_end_points=False):
    x, y, w, h = bbox
    square_length = min(square_length, int(w / 4), int(h / 4))
    zeros = np.zeros(out_damage.shape, dtype=np.uint8)
    conts = get_contours(out_damage)
    M = cv2.moments(conts[0])
    theta = abs(0.5 * np.arctan2(2 * M["mu11"], M["mu20"] - M["mu02"]))
    points = []
    if theta >= np.pi / 4 and theta <= 3 * np.pi / 4:
        num_interations = int(h / square_length)
        square_length = int(h / num_interations)
        limits = [(i * square_length, (i + 1) * square_length) if i < num_interations - 1 else (i * square_length, h)
                  for i in range(num_interations)]
        for l1, l2 in limits:
            roi = cv2.rectangle(zeros.copy(), (x, y + l1), (x + w, y + l2), 1, -1)
            conts = get_contours(roi * out_damage)
            if len(conts) > 0:
                points.append([*get_centroid(conts[0])])
        if first_end_points:
            first_point = get_intersection_point(points[0], points[1], bbox, side='top')
            end_point = get_intersection_point(points[-1], points[-2], bbox, side='bottom')
    else:
        num_interations = int(w / square_length)
        square_length = int(w / num_interations)
        limits = [(i * square_length, (i + 1) * square_length) if i < num_interations - 1 else (i * square_length, w)
                  for i in range(num_interations)]
        for l1, l2 in limits:
            roi = cv2.rectangle(zeros.copy(), (x + l1, y), (x + l2, y + h), 1, -1)
            try:
                conts = get_contours(roi * out_damage)
                if len(conts) > 0 and cv2.contourArea(conts[0]) > 1:
                    points.append([*get_centroid(conts[0])])
            except:
                continue
        if first_end_points:
            first_point = get_intersection_point(points[0], points[1], bbox, side='left')
            end_point = get_intersection_point(points[-1], points[-2], bbox, side='right')
    if first_end_points:
        return [[*first_point]] + points + [[*end_point]]
    else:
        return points
    
    
def FB2FL(crack_outs_dict, FB_min_area=20000):
    '''
    swan results 3/16/2023 shown that FB is detected in most of the cases where
    other cracks appear, so for small ones they will be assigned to FL, if they
    are not assigned previously to other cathegory.
    When FB detections are more accurate, this function can be deleted in get_report()
    '''
    FBs = bdamage_in_damages(crack_outs_dict.keys(), 'FB_')
    if len(FBs) > 0:
        for FB in FBs:
            out_FB = crack_outs_dict[FB]
            zeros = np.zeros(out_FB.shape,np.uint8)
            conts = get_contours(out_FB)
            for cont in conts:
                if cv2.contourArea(cont) < FB_min_area:
                    zeros = cv2.fillPoly(zeros, [cont], 1)
            if np.sum(zeros) > 0:
                nlabel = 'FL_' + FB.split('_')[-1]
                if nlabel in crack_outs_dict.keys():
                    out_nlabel = crack_outs_dict[nlabel]
                    out_nlabel[zeros*out_FB!=0] = 1
                    crack_outs_dict[nlabel] = out_nlabel
                else:
                    crack_outs_dict[nlabel] = zeros
                out_FB[zeros!=0] = 0
                crack_outs_dict[FB] = out_FB
    return clean_outs_dict(crack_outs_dict)

def get_margen(out, borders = ['FPAV','CUN','VEG','TRR'], y_top = 400):
    '''
    to get the area of interest based on borders and y_top
    '''
    margen = np.isin(out,[CONV_LU[b] for b in borders]).astype(np.uint8)
    cv2.rectangle(margen,(0,0),(margen.shape[1],400),1,-1)
    return np.array(margen==0,np.uint8)


def big_area_analysis(out, other_damage_outs, margen,big_area, threshold_predominantly, centroids_mindist,threshold_percentage=70/100):
    '''
    Compare big area damages (PA, DSU) to see if they can be change to the total
    area given by margen, starting with the biggest one, removing the relatives,
    and taking off the non relatives damages to the whole area.
    It is recommended to use margen without y_top to get more info of DSU...
    returns an updated version of other_damage_outs
    '''
    data = []
    for ba in big_area:
        if ba in other_damage_outs.keys():
            data.append([ba,np.sum(other_damage_outs[ba]*margen!=0)])
    if len(data)>0:
        df = pd.DataFrame(data,columns=['label','area'])
        df.sort_values('area',ascending=False,inplace=True,ignore_index=True)
        # crack_area = np.sum([crack_outs[label] for label in crack_outs.keys()], axis = 0)
        # crack_area = np.array(crack_area!=0,np.uint8)*margen
        # crack_area = np.sum(crack_area) SOM PINT LD PCH
        crack_area = np.sum(np.array(np.isin(out,CRACKS_ID),np.uint8)*margen)
        other_damage = np.zeros(out.shape,np.uint8)
        for dmg in ['PCH_1', 'PCH_2', 'PCH_3']:
            if dmg in other_damage_outs.keys():
                other_damage += other_damage_outs[dmg]
        other_damage = np.sum(other_damage!=0)
        other_detections = np.sum(np.array(np.isin(out,[CONV_LU[dmg] for dmg in ['SOM', 'PINT', 'LD', 'TRR']]),np.uint8)*margen)
        if df.area[0]/(np.sum(margen)-crack_area-other_damage-other_detections) > threshold_percentage:
            other_damage_outs[df.label[0]] = margen.copy()
            other_damage_outs = mix_same_type(other_damage_outs, threshold_predominantly, centroids_mindist)
            for label in other_damage_outs.keys():
                out_big = other_damage_outs[df.label[0]]
                if label != df.label[0]:
                    out_big *= np.array(other_damage_outs[label]==0,np.uint8)
                other_damage_outs[df.label[0]] = out_big
        return other_damage_outs
    else:
        return other_damage_outs


def get_cracks_results2(out, crack_outs, other_damage_outs, margen, min_linear_area = 800, min_dist_border = 50):
    '''
    returns polygons and polylines of area and linear cracks. Shapes.
    if the area of the bbox is grater than 1mx2m it will require to shrink or
    just use the bbox of the original detection.
    in cases of damages colser to the borders, the bbox or polyline will be
    adjusted to the border.
    20 cm ~ 50 px.

    shrink_kernel: if None, just the original detection will be used to get the bbox.
    '''
    # getting Patching mask to avoid overlays
    PCHs = bdamage_in_damages(other_damage_outs.keys(), 'PCH_')
    out_PCHs = np.zeros(out.shape,np.uint8)
    if len(PCHs) > 0:
        for PCH in PCHs:
            out_PCHs[other_damage_outs[PCH]!=0] = 1
    plantilla = margen.copy()
    plantilla[out_PCHs!=0] = 0
    # genereting polygons and polylines
    shapes = []
    for damage, crack_out in crack_outs.items():
        if damage in AREA_CRACKS:
            conts = get_contours(crack_out*plantilla)
            for cont in conts:
                shape = {}
                shape['label'] = damage
                shape['labelType'] = 'polygon'
                shape['points'] = cont.reshape(len(cont),2)
                shapes.append(shape)

        if damage in LINE_CRACKS:
            conts = get_contours(crack_out*plantilla)
            for cont in conts:
                bbox = cv2.boundingRect(cont)
                area = cv2.contourArea(cont)
                if area > min_linear_area:
                    bbox = check_bbox_limits(bbox,0,out.shape[1],out.shape,min_dist_border)
                    stencil = np.zeros(out.shape,dtype=np.uint8)
                    cv2.fillPoly(stencil, [cont], 1)
                    damage_out = np.isin(out,CRACKS_ID) * stencil
                    if np.sum(damage_out) > 0:
                        shape = {}
                        shape['label'] = damage
                        shape['labelType'] = 'polyline'
                        shape['points'] = get_polyline(damage_out, bbox)
                        shapes.append(shape)
    return shapes


def get_other_damages_results2(other_damage_outs,margen):
    '''
    creates polygons of no crack damages allowing overlaping.
    '''
    shapes = []
    for damage, other_damage_out in other_damage_outs.items():
        conts = get_contours(np.array(other_damage_out!=0,np.uint8)*margen)
        for cont in conts:
            shape = {}
            shape['label'] = damage
            shape['labelType'] = 'polygon'
            shape['points'] = cont.reshape(len(cont),2)
            shapes.append(shape)
    return shapes


def change_labels_shapes(shapes, converter: dict):
    for shape in shapes:
        shape['label'] = converter[shape['label']]
    return shapes


def gcs_url(gc_path, bucket):
    '''
    from the "project" path and bucket name, public gcs url is generated
    '''
    path = gc_path.split('project/')[-1]
    url = f'https://storage.googleapis.com/{bucket}/{path}'
    return url