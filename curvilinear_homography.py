#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 09:58:31 2022

@author: derek
"""

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
    
    def __init__(self, save_file = None):
        pass
    
    def generate(self,space_path,im_path):
        pass
    
    def _fit_z_vp(self):
        pass
    
    def _fit_spline(self,use_MM_offset = False):
        pass
    
    
    def _cache_spline_points(self,granularity = 0.1):
        pass
    
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
    
    def im_to_state(self):
        pass
    
    def state_to_im(self):
        pass
    
    def get_direction(self):
        pass
    
    def class_height(self):
        pass
    
    def height_from_template(self):
        pass
    
    def test_transformation(self):
        pass
    
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
    
    
    def get_extents(self):
        pass
    
    def generate_mask_images(self):
        pass
    
    def check_extents(self):
        pass