#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 09:58:31 2022

@author: derek
"""


import os
import _pickle as pickle
import pandas as pd
import numpy as np
import torch
import glob
import cv2
import time
import string
import re
import copy
import sys

from scipy import interpolate


#%% Utility functions
def line_to_point(line,point):
    """
    Given a line defined by two points, finds the distance from that line to the third point
    line - (x0,y0,x1,y1) as floats
    point - (x,y) as floats
    Returns
    -------
    distance - float >= 0
    """
    
    numerator = np.abs((line[2]-line[0])*(line[1]-point[1]) - (line[3]-line[1])*(line[0]-point[0]))
    denominator = np.sqrt((line[2]-line[0])**2 +(line[3]-line[1])**2)
    
    return numerator / (denominator + 1e-08)




class Curvilinear_Homography():
    
    def safe_name(func):
        """
        Wrapper function, catches camera names that aren't capitalized 
        """
        
        def new_func(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except KeyError:
                #print(args,kwargs)
                if type(kwargs["name"]) == list:
                    kwargs["name"] = [item.upper() for item in kwargs["name"]]
                elif type(kwargs["name"]) == str:
                    kwargs["name"] = kwargs["name"].upper()
                return func(*args, **kwargs)
        return new_func
    
    #%% Initialization and Setup Functions
    
    
    
    """
    
    3 coordinate systems are utilized in Curvilinear_Homography:
        -image coordinates
        - space coordinates (state plane coordinates) in feet
        - roadway coordianates / curvilinear coordinates in feet
    
    After fitting, each value of self.correspondence contains:
        H     - np array of size [3,3] used for image to space perspective transform
        H_inv - used for space to image perspective transform on ground plane
        P     - np array of size [3,4] used for space to image transform
        corr_pts - list of [x,y] points in image space that are fit for transform
        space_pts - corresponding list of [x,y] points in space (state plane coordinates in feet)
        state_plane_pts - same as space_pts but [x,y,id] (name is included)
        vps - z vanishing point [x,y] in image coordinates
        extents - xmin,xmax,ymin,ymax in roadway coordinates
        extents_space - list of array of [x,y] points defining boundary in state plane coordinates
        
    """
    
    
    def __init__(self, 
                 save_file = None,
                 space_dir = None,
                 im_dir = None):
        """
        Initializes homography object.
        
        save_file - None or str - if str, specifies path to cached homography object
        space_dir - None or str - path to directory with csv files of attributes labeled in space coordinates
        im_dir    - None or str - path to directory with cpkl files of attributes labeled in image coordinates
        """
        
        # intialize correspondence
        
        self.correspondence = {}
        if save_file is not None:
            with open(save_file,"rb") as f:
                # everything in correspondence is pickleable without object definitions to allow compatibility after class definitions change
                self.correspondence,spline_params = pickle.load(f)
            
            # reload  parameters of curvilinear axis spline
            # rather than the spline itself for better pickle reloading compatibility
                
        
        elif space_dir is None or im_dir is None:
            raise IOError("Either save_file or space_dir and im_dir must not be None")
        
        else:
            #self.generate(space_dir,im_dir)
            
            #  fit the axis spline once and collect extents
            self._fit_spline(space_dir)
            self._get_extents()    
        
        
        # object class info doesn't really belong in homography but it's unclear
        # where else it should go, and this avoids having to pass it around 
        # for use in height estimation
        self.class_dims = {
                "sedan":[16,6,4],
                "midsize":[18,6.5,5],
                "van":[20,6,6.5],
                "pickup":[20,6,5],
                "semi":[55,9,14],
                "truck (other)":[25,9,14],
                "truck": [25,9,14],
                "motorcycle":[7,3,4],
                "trailer":[16,7,3],
                "other":[18,6.5,5]
            }
        
        self.class_dict = { "sedan":0,
                    "midsize":1,
                    "van":2,
                    "pickup":3,
                    "semi":4,
                    "truck (other)":5,
                    "truck": 5,
                    "motorcycle":6,
                    "trailer":7,
                    0:"sedan",
                    1:"midsize",
                    2:"van",
                    3:"pickup",
                    4:"semi",
                    5:"truck (other)",
                    6:"motorcycle",
                    7:"trailer"
                    }
        
        
    
    def generate(self,space_dir,
                 im_dir,
                 downsample     = 1,
                 max_proj_error = 0.25,
                 scale_factor   = 0.5,
                 ADD_PROJ       = True):
        """
        Loads all available camera homographies from the specified paths.
        after running, self.correspondence is a dict with one key for each <camera>_<direction>
        
        space_dir      - str - path to directory with csv files of attributes labeled in space coordinates
        im_dir         - str - path to directory with cpkl files of attributes labeled in image coordinates
        downsample     - int - specifies downsampling ratio for image coordinates
        max_proj_error - float - max allowable positional error (ft) between point and selected corresponding point on spline, 
                                 lower will exclude more points from homography computation
        scale_factor   - float - sampling frequency (ft) along spline, lower is slower but more accurate
        ADD_PROJ       - bool - if true, compute points along yellow line to use in homography
        """
        
        
        for direction in ["EB","WB"]:
            ### State space, do once
            
            ae_x = []
            ae_y = []
            ae_id = []
            
            for file in os.listdir(space_dir):
                if direction.lower() not in file:
                    continue
                
                # load all points
                dataframe = pd.read_csv(os.path.join(space_dir,file))
                try:
                    dataframe = dataframe[dataframe['point_pos'].notnull()]
                    attribute_name = file.split(".csv")[0]
                    feature_idx = dataframe["point_id"].tolist()
                    st_id = [attribute_name + "_" + item for item in feature_idx]
                    
                    st_x = dataframe["st_x"].tolist()
                    st_y = dataframe["st_y"].tolist()
                
                    ae_x  += st_x
                    ae_y  += st_y
                    ae_id += st_id
                except:
                    dataframe = dataframe[dataframe['side'].notnull()]
                    attribute_name = file.split(".csv")[0]
                    feature_idx = dataframe["id"].tolist()
                    side        = dataframe["side"].tolist()
                    st_id = [attribute_name + str(side[i]) + "_" + str(feature_idx[i]) for i in range(len(feature_idx))]
                    
                    st_x = dataframe["st_x"].tolist()
                    st_y = dataframe["st_y"].tolist()
                
                    ae_x  += st_x
                    ae_y  += st_y
                    ae_id += st_id
            
            
            # Find a-d end point of all d2 lane markers
            d2 = {}
            d3 = {}
            
            ae_spl_x = []
            ae_spl_y = []
            
            for i in range(len(ae_x)):
                if "d2" in ae_id[i]:
                    if ae_id[i].split("_")[-1] in ["a","d"]:
                        num = ae_id[i].split("_")[-2]
                        if num not in d2.keys():
                            d2[num] = [(ae_x[i],ae_y[i])]
                        else:
                            d2[num].append((ae_x[i],ae_y[i]))
                elif "d3" in ae_id[i]:
                    if ae_id[i].split("_")[-1] in ["a","d"]:
                        num = ae_id[i].split("_")[-2]
                        if num not in d3.keys():
                            d3[num] = [(ae_x[i],ae_y[i])]
                        else:
                            d3[num].append((ae_x[i],ae_y[i]))
                            
                elif "yeli" in ae_id[i]:
                    ae_spl_x.append(ae_x[i])
                    ae_spl_y.append(ae_y[i])
                
                    
            
            # stack d2 and d3 into arrays            
            d2_ids = []
            d2_values = []
            for key in d2.keys():
                val = d2[key]
                d2_ids.append(key)
                d2_values.append(   [(val[0][0] + val[1][0])/2.0   ,   (val[0][1] + val[1][1])/2.0     ])
            
            d3_ids = []
            d3_values = []
            for key in d3.keys():
                val = d3[key]
                d3_ids.append(key)
                d3_values.append(   [(val[0][0] + val[1][0])/2.0   ,   (val[0][1] + val[1][1])/2.0     ])
            
            d2_values = torch.from_numpy(np.stack([np.array(item) for item in d2_values]))
            d3_values = torch.from_numpy(np.stack([np.array(item) for item in d3_values]))
            
            d2_exp = d2_values.unsqueeze(1).expand(d2_values.shape[0],d3_values.shape[0],2)
            d3_exp = d3_values.unsqueeze(0).expand(d2_values.shape[0],d3_values.shape[0],2)
            
            dist = torch.sqrt(torch.pow(d2_exp - d3_exp , 2).sum(dim = -1))
            
            min_matches = torch.min(dist, dim = 1)[1]
            
            if ADD_PROJ:
                
                try:
                    with open("ae_cache_{}.cpkl".format(direction),"rb") as f:
                        additional_points = pickle.load(f)
                except:
                    # For each d2 lane marker, find the closest d3 lane marker
                    proj_lines = []
                    
                    for i in range(len(min_matches)):
                        j = min_matches[i]
                        pline = [d3_values[j],d2_values[i],d3_ids[j],d2_ids[i]]
                        proj_lines.append(pline)
                    
                    
                    
                    # compute the yellow line spline in state plane coordinates
                    
                    ae_data = np.stack([np.array(ae_spl_x),np.array(ae_spl_y)])
                    ae_data = ae_data[:,np.argsort(ae_data[1,:])]
                    
                    ae_tck, ae_u = interpolate.splprep(ae_data, s=0, per=False)
                    
                    span_dist = np.sqrt((ae_spl_x[0] - ae_spl_x[-1])**2 + (ae_spl_y[0] - ae_spl_y[-1])**2)
                    ae_x_prime, ae_y_prime = interpolate.splev(np.linspace(0, 1, int(span_dist*scale_factor)), ae_tck)
                
                    additional_points = []
                    # for each d2 lane marker, find the intersection between the d2-d3 line and the yellow line spline
                    for p_idx, proj_line in enumerate(proj_lines):
                        print("On proj line {} of {}".format(p_idx,len(proj_lines)))
                        min_dist = np.inf
                        min_point = None
                        line = [proj_line[0][0],proj_line[0][1],proj_line[1][0],proj_line[1][1]]
                        for i in range(len(ae_x_prime)):
                            point = [ae_x_prime[i],ae_y_prime[i]]
                            
                            dist = line_to_point(line, point)
                            if dist < min_dist:
                                min_dist = dist
                                min_point = point
                        if min_dist > max_proj_error:
                            print("Issue")
                        else:
                            name = "{}_{}".format(proj_line[2],proj_line[3])
                            min_point.append(name)
                            additional_points.append(min_point)
                            
                    with open("ae_cache_{}.cpkl".format(direction),"wb") as f:
                        pickle.dump(additional_points,f)
                        
                
                for point in additional_points:
                    ae_x.append(point[0])
                    ae_y.append(point[1])
                    ae_id.append(point[2])
    
    
        # get all cameras
        cam_data_paths = glob.glob(os.path.join(space_dir,"*.cpkl"))
        
        for cam_data_path in cam_data_paths:
            # specify path to camera imagery file
            cam_im_path   = cam_data_path.split(".cpkl")[0] + ".png"
            camera = cam_data_path.split(".cpkl")[0].split("/")[-1]
            
            # load all points
            with open(cam_data_path, "rb") as f:
                im_data = pickle.load(f)
                
            
            # get all non-curve matching points
            point_data = im_data[direction]["points"]
            filtered = filter(lambda x: x[2].split("_")[1] not in ["yeli","yelo","whli","whlo"],point_data)
            im_x  = []
            im_y  = []
            im_id = []
            for item in filtered:
                im_x.append(item[0])
                im_y.append(item[1])
                im_id.append(item[2])
            
            if len(im_x) == 0:
                continue
            
            if ADD_PROJ:
            
                # compute the yellow line spline in image coordinates
                curve_data = im_data[direction]["curves"]
                filtered = filter(lambda x: "yeli" in x[2], curve_data)
                x = []
                y = []
                for item in filtered:
                    x.append(item[0])
                    y.append(item[1])
                data = np.stack([np.array(x),np.array(y)])
                data = data[:,np.argsort(data[0,:])]
                
                tck, u = interpolate.splprep(data, s=0, per=False)
                x_prime, y_prime = interpolate.splev(np.linspace(0, 1, 4000), tck)
                
                if False:
                    im = cv2.imread(cam_im_path)
                    for i in range(len(x_prime)):
                        cv2.circle(im,(int(x_prime[i]),int(y_prime[i])), 2, (255,0,0,),-1)
                        
                    cv2.imshow("frame",im)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    
                # find all d2 and d3 points
            
                # find the intersection of each d2d3 line and the yellow line spline for d2-d3 pairs in image
                d2 = {}
                d3 = {}
                for i in range(len(im_x)):
                    if "d2" in im_id[i]:
                        if im_id[i].split("_")[-1] in ["a","d"]:
                            num = im_id[i].split("_")[-2]
                            if num not in d2.keys():
                                d2[num] = [(im_x[i],im_y[i])]
                            else:
                                d2[num].append((im_x[i],im_y[i]))
                    elif "d3" in im_id[i]:
                        if im_id[i].split("_")[-1] in ["a","d"]:
                            num = im_id[i].split("_")[-2]
                            if num not in d3.keys():
                                d3[num] = [(im_x[i],im_y[i])]
                            else:
                                d3[num].append((im_x[i],im_y[i]))
                    
            
                # stack d2 and d3 into arrays            
                d2_ids = []
                d2_values = []
                for key in d2.keys():
                    d2[key] = [(d2[key][0][0] + d2[key][1][0])/2.0  , (d2[key][0][1] + d2[key][1][1])/2.0 ] 
                
                d3_ids = []
                d3_values = []
                for key in d3.keys():
                    d3[key] = [(d3[key][0][0] + d3[key][1][0])/2.0  , (d3[key][0][1] + d3[key][1][1])/2.0 ] 
                
                additional_im_points = []
                for proj_point in additional_points:
                    
                    
                    d3_id = proj_point[2].split("_")[0]
                    d2_id = proj_point[2].split("_")[1]
                    
                    if d3_id not in d3.keys() or d2_id not in d2.keys():
                        continue
                    
                    im_line = [d3[d3_id][0], d3[d3_id][1], d2[d2_id][0], d2[d2_id][1]]
                    
                    min_dist = np.inf
                    min_point = None
                    for i in range(len(x_prime)):
                        point = [x_prime[i],y_prime[i]]
            
                        dist = line_to_point(im_line, point)
                        if dist < min_dist:
                            min_dist = dist
                            min_point = point
                    if min_dist > 2:
                        print("Issue")
                    else:
                        name = proj_point[2]
                        min_point.append(name)
                        additional_im_points.append(min_point)
                        
                for point in additional_im_points:
                    im_x.append(point[0])
                    im_y.append(point[1])
                    im_id.append(point[2])
                    
            ### Joint
            
            # assemble ordered list of all points visible in both image and space
            
            include_im_x  = []
            include_im_y  = []
            include_im_id = []
            
            include_ae_x  = []
            include_ae_y  = []
            include_ae_id = []
            
            for i in range(len(ae_id)):
                for j in range(len(im_id)):
                    
                    if ae_id[i] == im_id[j]:
                        include_im_x.append(  im_x[j])
                        include_im_y.append(  im_y[j])
                        include_im_id.append(im_id[j])
                        
                        include_ae_x.append(  ae_x[i])
                        include_ae_y.append(  ae_y[i])
                        include_ae_id.append(ae_id[i])
            
            
            
            # compute homography
            vp = im_data[direction]["z_vp"]
            corr_pts = np.stack([np.array(include_im_x),np.array(include_im_y)]).transpose(1,0)
            space_pts = np.stack([np.array(include_ae_x),np.array(include_ae_y)]).transpose(1,0)
            
            
            cor = {}
            #cor["vps"] = vp
            cor["corr_pts"] = corr_pts
            cor["space_pts"] = space_pts
            
            cor["H"],_     = cv2.findHomography(corr_pts,space_pts)
            cor["H_inv"],_ = cv2.findHomography(space_pts,corr_pts)
            
            
            # P is a [3,4] matrix 
            #  column 0 - vanishing point for space x-axis (axis 0) in image coordinates (im_x,im_y,im_scale_factor)
            #  column 1 - vanishing point for space y-axis (axis 1) in image coordinates (im_x,im_y,im_scale_factor)
            #  column 2 - vanishing point for space z-axis (axis 2) in image coordinates (im_x,im_y,im_scale_factor)
            #  column 3 - space origin in image coordinates (im_x,im_y,scale_factor)
            #  columns 0,1 and 3 are identical to the columns of H, 
            #  We simply insert the z-axis column (im_x,im_y,1) as the new column 2
            
            P = np.zeros([3,4])
            P[:,0] = cor["H_inv"][:,0]
            P[:,1] = cor["H_inv"][:,1]
            P[:,3] = cor["H_inv"][:,2] 
            P[:,2] = np.array([vp[0],vp[1],1])  * 10e-09
            cor["P"] = P
    
            self._fit_z_vp(cor,im_data,direction)
            
            cor["state_plane_pts"] = [include_ae_x,include_ae_y,include_ae_id]
            cor_name = "{}_{}".format(camera,direction)
            self.correspondence[cor_name] = cor
  
    def _fit_z_vp(self,cor,im_data,direction):
        P_orig = cor["P"].copy()
        
        max_scale = 10000
        granularity = 1e-12
        upper_bound = max_scale
        lower_bound = -max_scale
        
        # create a grid of 100 evenly spaced entries between upper and lower bound
        C_grid = np.linspace(lower_bound,upper_bound,num = 100,dtype = np.float64)
        step_size = C_grid[1] - C_grid[0]
        iteration = 1
        
        while step_size > granularity:
            
            best_error = np.inf
            best_C = None
            # for each value of P, get average reprojection error
            for C in C_grid:
                
                # scale P
                P = P_orig.copy()
                P[:,2] *= C
                
                
                # search for optimal scaling of z-axis row
                vp_lines = im_data[direction]["z_vp_lines"]
                
                # get bottom point (point # 2)
                points = torch.stack([ torch.tensor([vpl[2] for vpl in vp_lines]), 
                                          torch.tensor([vpl[3] for vpl in vp_lines]) ]).transpose(1,0)
                t_points = torch.stack([ torch.tensor([vpl[0] for vpl in vp_lines]), 
                                          torch.tensor([vpl[1] for vpl in vp_lines]) ]).transpose(1,0)
                heights =  torch.tensor([vpl[4] for vpl in vp_lines]).unsqueeze(1)

                
                # project to space
                
                d = points.shape[0]
                
                # convert points into size [dm,3]
                points = points.view(-1,2).double()
                points = torch.cat((points,torch.ones([points.shape[0],1],device=points.device).double()),1) # add 3rd row
                
                H = torch.from_numpy(cor["H"]).transpose(0,1).to(points.device)
                new_pts = torch.matmul(points,H)
                    
                # divide each point 0th and 1st column by the 2nd column
                new_pts[:,0] = new_pts[:,0] / new_pts[:,2]
                new_pts[:,1] = new_pts[:,1] / new_pts[:,2]
                
                # drop scale factor column
                new_pts = new_pts[:,:2] 
                
                # reshape to [d,m,2]
                new_pts = new_pts.view(d,2)
                
                # add third column for height
                new_pts_shifted  = torch.cat((new_pts,heights.double()),1)
                

                # add fourth column for scale factor
                new_pts_shifted  = torch.cat((new_pts_shifted,torch.ones(heights.shape)),1)
                new_pts_shifted = torch.transpose(new_pts_shifted,0,1).double()

                # project to image
                P = torch.from_numpy(P).double().to(points.device)
                

                new_pts = torch.matmul(P,new_pts_shifted).transpose(0,1)
                
                # divide each point 0th and 1st column by the 2nd column
                new_pts[:,0] = new_pts[:,0] / new_pts[:,2]
                new_pts[:,1] = new_pts[:,1] / new_pts[:,2]
                
                # drop scale factor column
                new_pts = new_pts[:,:2] 
                
                # reshape to [d,m,2]
                repro_top = new_pts.view(d,-1,2).squeeze()
                
                # get error
                
                error = torch.pow((repro_top - t_points),2).sum(dim = 1).sqrt().mean()
                
                # if this is the best so far, store it
                if error < best_error:
                    best_error = error
                    best_C = C
                    
            
            # define new upper, lower  with width 2*step_size centered on best value
            #print("On loop {}: best C so far: {} avg error {}".format(iteration,best_C,best_error))
            lower_bound = best_C - 2*step_size
            upper_bound = best_C + 2*step_size
            C_grid = np.linspace(lower_bound,upper_bound,num = 100,dtype = np.float64)
            step_size = C_grid[1] - C_grid[0]

            #print("New C_grid: {}".format(C_grid.round(4)))
            iteration += 1
        
        
        
        P_new = P_orig.copy()
        P_new[:,2] *= best_C
        cor["P"] = P_new
            
        
        print("Best Error: {}".format(best_error))
        
        
    def _fit_spline(self,space_dir,use_MM_offset = False):
        """
        Spline fitting is done by:
            1. Assemble all points labeled along a yellow line in either direction
            2. Fit a spline to each of EB, WB inside and outside
            3. Sample the spline at fine intervals
            4. Use finite difference method to determine the distance along the spline for each fit point
            5. Refit the splines, this time parameterizing the spline by these distances (u parameter in scipy.splprep)
            6. Sample each spline at fine intervals
            7. Move along one spline and at each point, find the closest point on each other spline
            8. Define a point on the median/ midpoint axis as the average on these 4 splines
            9. Use the set of median points to define a new spline
            10. Use the finite difference method to reparameterize this spline according to distance along it
            11. Optionally, compute a median spline distance offset from mile markers
            12. Optionally, recompute the same spline, this time accounting for the MM offset
            
            space_dir - str - path to directory with csv files of attributes labeled in space coordinates
            use_MM_offset - bool - if True, offset according to I-24 highway mile markers
        """
        samples_per_foot = 10
        splines = {}
        
        for direction in ["EB","WB"]:
            for line_side in ["i","o"]:
                ### State space, do once
                
                ae_x = []
                ae_y = []
                ae_id = []
                
                # 1. Assemble all points labeled along a yellow line in either direction
                for file in os.listdir(space_dir):
                    if direction.lower() not in file:
                        continue
                    
                    # load all points
                    dataframe = pd.read_csv(os.path.join(space_dir,file))
                    try:
                        dataframe = dataframe[dataframe['point_pos'].notnull()]
                        attribute_name = file.split(".csv")[0]
                        feature_idx = dataframe["point_id"].tolist()
                        st_id = [attribute_name + "_" + item for item in feature_idx]
                        
                        st_x = dataframe["st_x"].tolist()
                        st_y = dataframe["st_y"].tolist()
                    
                        ae_x  += st_x
                        ae_y  += st_y
                        ae_id += st_id
                    except:
                        dataframe = dataframe[dataframe['side'].notnull()]
                        attribute_name = file.split(".csv")[0]
                        feature_idx = dataframe["id"].tolist()
                        side        = dataframe["side"].tolist()
                        st_id = [attribute_name + str(side[i]) + "_" + str(feature_idx[i]) for i in range(len(feature_idx))]
                        
                        st_x = dataframe["st_x"].tolist()
                        st_y = dataframe["st_y"].tolist()
                    
                        ae_x  += st_x
                        ae_y  += st_y
                        ae_id += st_id
                
                
                # Find a-d end point of all d2 lane markers
                ae_spl_x = []
                ae_spl_y = []
                ae_spl_u = []  # u parameterizes distance along spline 
                
                
                for i in range(len(ae_x)):
                    
                    if "yel{}".format(line_side) in ae_id[i]:
                        ae_spl_x.append(ae_x[i])
                        ae_spl_y.append(ae_y[i])

                # 2. Fit a spline to each of EB, WB inside and outside
                 
                # compute the yellow line spline in state plane coordinates (sort points by y value since road is mostly north-south)
                ae_data = np.stack([np.array(ae_spl_x),np.array(ae_spl_y)])
                ae_data = ae_data[:,np.argsort(ae_data[1,:],)[::-1]]
            
                # 3. Sample the spline at fine intervals
                # get spline and sample points on spline
                ae_tck, ae_u = interpolate.splprep(ae_data, s=0, per=False)
                span_dist = np.sqrt((ae_spl_x[0] - ae_spl_x[-1])**2 + (ae_spl_y[0] - ae_spl_y[-1])**2)
                ae_x_prime, ae_y_prime = interpolate.splev(np.linspace(0, 1, int(span_dist*samples_per_foot)), ae_tck)
            

                # 4. Use finite difference method to determine the distance along the spline for each fit point
                fd_dist = np.concatenate(  (np.array([0]),  ((ae_x_prime[1:] - ae_x_prime[:-1])**2 + (ae_y_prime[1:] - ae_y_prime[:-1])**2)**0.5),axis = 0) # by convention fd_dist[0] will be 0, so fd_dist[i] = sum(int_dist[0:i])
                integral_dist = np.cumsum(fd_dist)
                
                # for each fit point, find closest point on spline, and assign it the corresponding integral distance
                for p_idx in range(len(ae_spl_x)):
                    px = ae_spl_x[p_idx]
                    py = ae_spl_y[p_idx]
                    
                    dist = ((ae_x_prime - px)**2 + (ae_y_prime - py)**2)**0.5
                    min_dist,min_idx= np.min(dist),np.argmin(dist)
                    ae_spl_u.append(integral_dist[min_idx])
                
                # 5. Refit the splines, this time parameterizing the spline by these distances (u parameter in scipy.splprep)
                #ae_spl_u.reverse()
                
                tck, u = interpolate.splprep(ae_data.astype(float), s=0, u = ae_spl_u)
                splines["{}_{}".format(direction,line_side)] = [tck,u]
                
                
           
        # 6. Sample each spline at fine intervals
        for key in splines:
            tck,u = splines[key]

            span_dist = np.abs(u[0] - u[-1])
            x_prime, y_prime = interpolate.splev(np.linspace(u[0], u[-1], int(3*span_dist)), tck)
            splines[key].append(x_prime)
            splines[key].append(y_prime)
            
        med_spl_x = []
        med_spl_y = []
        
        
        # 7. Move along one spline and at each point, find the closest point on each other spline
        # by default, we'll use EB_o as the base spline
        main_key = "EB_o"
        main_spl = splines[main_key]
        main_x = main_spl[2]
        main_y = main_spl[3]
        
        for p_idx in range(len(main_x)):
            
            points_to_average = [np.array([px,py])]
            
            px,py = main_x[p_idx],main_y[p_idx]
            
            for key in splines:
                if key != main_key:
                    arr_x,arr_y = splines[key][2], splines[key][3]
                    
                    dist = np.sqrt((arr_x - px)**2 + (arr_y - py)**2)
                    min_dist,min_idx= np.min(dist),np.argmin(dist)
                    
                    points_to_average.append( np.array([arr_x[p_idx],arr_y[p_idx]]))
            
            
            med_point = sum(points_to_average)/len(points_to_average)
            
            # 8. Define a point on the median/ midpoint axis as the average on these 4 splines
            med_spl_x.append(med_point[0])
            med_spl_y.append(med_point[1])
        

        
        # 9. Use the set of median points to define a new spline
        med_data = np.stack([np.array(med_spl_x),np.array(med_spl_y)])
        med_tck,med_u = interpolate.splprep(med_data, s=0, per=False)
        
        # 10. Use the finite difference method to reparameterize this spline according to distance along it
        span_dist = np.sqrt((med_spl_x[0] - med_spl_x[-1])**2 + (med_spl_y[0] - med_spl_y[-1])**2)
        med_x_prime, med_y_prime = interpolate.splev(np.linspace(0, 1, int(span_dist*samples_per_foot)), med_tck)
        
        
        med_fd_dist = np.concatenate(  (np.array([0]),  ((med_x_prime[1:] - med_x_prime[:-1])**2 + (med_y_prime[1:] - med_y_prime[:-1])**2)**0.5),axis = 0) # by convention fd_dist[0] will be 0, so fd_dist[i] = sum(int_dist[0:i])
        med_integral_dist = np.cumsum(med_fd_dist)
        
        # for each fit point, find closest point on spline, and assign it the corresponding integral distance
        med_spl_u = []
        for p_idx in range(len(med_spl_x)):
            px = med_spl_x[p_idx]
            py = med_spl_y[p_idx]
            
            dist = ((med_x_prime - px)**2 + (med_y_prime - py)**2)**0.5
            min_dist,min_idx= np.min(dist),np.argmin(dist)
            med_spl_u.append(med_integral_dist[min_idx])
        
        
        final_tck, final_u = interpolate.splprep(ae_data.astype(float), s=0, u = ae_spl_u)
        self.median_tck = final_tck
        self.median_u = final_u
        
        
        if use_MM_offset:
            # 11. Optionally, compute a median spline distance offset from mile markers
            self.MM_offset = self._fit_MM_offset()
        
            # 12. Optionally, recompute the same spline, this time accounting for the MM offset
            ae_spl_u += self.MM_offset
            final_tck, final_u = interpolate.splprep(ae_data.astype(float), s=0, u = ae_spl_u)
            self.median_tck = final_tck
            self.median_u = final_u
    
    
    
    def _fit_MM_offset(self):
        pass
    
    def _cache_spline_points(self,granularity = 0.1):
        pass
    
    def _get_extents(self):
        pass
    
    def generate_mask_images(self):
        pass
    

    
    #%% Conversion Functions
    
    @safe_name
    def im_to_space(self,heights = None):
        pass
    
    def space_to_state(self):
        pass
    
    def state_to_space(self):
        pass
    
    @safe_name
    def space_to_im(self):
        pass
    
    @safe_name
    def im_to_state(self):
        pass
    
    @safe_name
    def state_to_im(self):
        pass
    
    def get_direction(self):
        pass
    
    def class_height(self):
        pass
    
    def height_from_template(self):
        pass
    
    #%% Testing Functions
    
    def test_transformation(self):
        pass
    
    def check_extents(self):
        pass
    
    #%% Plotting Functions
    
    def plot_boxes(self):
        pass
    
    def plot_points(self):
        pass
    
    def plot_homography(self,
                        SPLINE  = False,
                        IM_PTS  = False,
                        FIT_PTS = False,
                        FOV     = False,
                        MASK    = False,
                        Z_AXIS  = False):
        pass
    
    
if __name__ == "__main__":
    im_dir = "/home/derek/Documents/i24/i24_homography/data_real"
    space_dir = "/home/derek/Documents/i24/i24_homography/aerial/to_P24"

    hg = Curvilinear_Homography(space_dir = space_dir, im_dir = im_dir)