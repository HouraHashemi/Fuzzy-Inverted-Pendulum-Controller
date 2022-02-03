# -*- coding: utf-8 -*-

# python imports
from math import degrees

# pyfuzzy imports
# from fuzzy.storage.fcl.Reader import Reader
import numpy as np
import math
from math import degrees
from numpy import ones,vstack
from numpy.linalg import lstsq
from scipy.integrate import quad
import matplotlib.pyplot as plt

class FuzzyController:
    

    def __init__(self, fcl_path):
        # self.system = Reader().load_from_file(fcl_path)
        self.membership_zone= dict()
        self.membership_functions = dict()

    def _make_input(self, world):
        return dict(
            cp = world.x,
            cv = world.v,
            pa = degrees(world.theta),
            pv = degrees(world.omega)
        )


    def _make_output(self):
        return dict(
            force = 0.
        )


    # ---------------------------------------------------
    # INIFIALIZE 
    # ---------------------------------------------------
    def initialize_membership_zone(self):
        # state:(start,max,end) -- Triangular waves
        pa = {"up_more_right":(0,30,60),
                                "up_right":(30,60,90),
                                "up":(60,90,120),
                                "up_left":(90,120,150),
                                "up_more_left":(120,150,180),
                                "down_more_left":(180,210,240),
                                "down_left":(210,240,270),
                                "down":(240,270,300),
                                "down_right":(270,300,330),
                                "down_more_right":(300,330,360)
                            }

        pv = {"cw_fast":(-200,-200,-100),
                                "cw_slow":(-200,-100,0),
                                "stop":(-100,0,100),
                                "ccw_slow":(0,100,200),
                                "ccw_fast":(100,200,200)
                            }

        cp = {"left_far":(-10,-10,-5),
                                "left_near":(-10,-2.5,0),
                                "stop":(-2.5,0,2.5),
                                "right_near":(0,2.5,10),
                                "right_far":(5,10,10),
                            }

        cv = {"left_fast":(-5,-5,-2.5),
                                "left_slow":(-5,-1,0),
                                "stop":(-1,0,1),
                                "right_slow":(0,1,5),
                                "right_fast":(2.5,5,5)
                            }

        force = {"left_fast":(-100,-80,-60),
                                "left_slow":(-80,-60,0),
                                "stop":(-60,0,60),
                                "right_slow":(0,60,80),
                                "right_fast":(60,80,100)
                            }

        self.membership_zone = dict([
                            ("pa",pa),
                            ("pv",pv),
                            ("cp",cp),
                            ("cv",cv),
                            ("force",force)
                            ])

        for mz in self.membership_zone:
            mzp = self.membership_zone[mz]
            temp = dict()
            for m in mzp:
                st,mx,nd = (mzp[m][0],0.0),(mzp[m][1],1.0),(mzp[m][2],0.0)
                funcs = dict()
                funcs[(st[0],mx[0])] = (self.equation(st,mx,"x"),self.equation(st,mx,"y"))
                funcs[(mx[0],nd[0])] = (self.equation(mx,nd,"x"),self.equation(mx,nd,"y"))
                temp[m] = funcs
            self.membership_functions[mz] = temp 


    

    # ---------------------------------------------------
    # FUZZIFICATION 
    # ---------------------------------------------------
    def fuzzification(self,params):         
        imp = dict()

        for pr in params:
            param_value = params[pr]
            normal_param_value = self.normalize(pr,param_value)
            # equation and zone for this parameter
            mfp = self.membership_functions[pr]
            membership = dict()
            for state in mfp:
                for zone in mfp[state]:
                    # print(pr,normal_param_value, zone)
                    if (zone[0] <= normal_param_value) and (normal_param_value <= zone[1]):
                        func_mu = mfp[state][zone][1]
                        y = func_mu(normal_param_value)
                        membership[state] = y
                    else:
                        membership[state] = 0.0
            imp[pr] = membership
        
        return imp


    def equation(self,(x1,y1),(x2,y2),find):
        if (x2 - x1) == 0: # edgs
            m = 0.0
            b = 0.0
            c = 1.0
        else:
            m = float(y2-y1)/(x2-x1)
            b = y2 - (x2*m)
            c = 0.0

        if find == "y":
            return lambda x: m*x + b + c   
        elif find == "x":
            return lambda y: (y - b)/m + c


    def normalize(self,param,param_value):
        if param == "pa":
            if param_value < 0:
                param_value = param_value % 360
                # param_value = param_value + 360
        elif param == "pv":
                if param_value > 200:
                    param_value = 200
                elif param_value < -200:
                    param_value = -200
        elif param == "cp":
            pass
        elif param == "cv":
            pass

        return param_value


    # ---------------------------------------------------
    # INFERENCE 
    # ---------------------------------------------------
    def inference(self,input_membership_parameters):

        # A*i -> B*i
        mpa = input_membership_parameters["pa"] # mu(x,y)
        mpv = input_membership_parameters["pv"]
        mcp = input_membership_parameters["cp"]
        mcv = input_membership_parameters["cv"]


        # print input_membership_parameters
        # print("pa",mpa)
        # print("pv",mpv)

        left_fast, left_slow, stop, right_slow, right_fast = list(),list(),list(),list(),list()

        output_membership = {"left_fast":0.0,"left_slow":0.0,"stop":0.0,"right_slow":0.0,"right_fast":0.0}

        # RULE 0
        if ((mpa["up"] > 0.0) and (mpv["stop"] > 0.0) or
            (mpa["up_right"] > 0.0) and (mpv["ccw_slow"] > 0.0) or
            (mpa["up_left"] > 0.0) and (mpv["cw_slow"] > 0.0)):
            stop.append(max(min(mpa["up"],mpv["stop"]),min(mpa["up_right"],mpv["ccw_slow"]),
                            min(mpa["up_left"],mpv["cw_slow"])))
            print("rule0 stop")

        # RULE 1,2
        if (mpa["up_more_right"] > 0.0) and (mpv["ccw_slow"] > 0.0):
            right_fast.append(min(mpa["up_more_right"],mpv["ccw_slow"]))
            print("rule1 right_fast")
        
        if (mpa["up_more_right"] > 0.0) and (mpv["cw_slow"] > 0.0):
            right_fast.append(min(mpa["up_more_right"],mpv["cw_slow"]))
            print("rule2 right_fast")
        

        # RULE 3,4
        if (mpa["up_more_left"] > 0.0) and (mpv["cw_slow"] > 0.0):
            left_fast.append(min(mpa["up_more_left"],mpv["cw_slow"]))
            print("rule3 left_fast")
        if (mpa["up_more_left"] > 0.0) and (mpv["ccw_slow"] > 0.0):
            left_fast.append(min(mpa["up_more_left"],mpv["ccw_slow"]))
            print("rule4 left_fast")

        # RULE 5,6
        if (mpa["up_more_right"] > 0.0) and (mpv["ccw_fast"] > 0.0):
            left_slow.append(min(mpa["up_more_right"],mpv["ccw_fast"]))
            print("rule5 left_slow")
        if (mpa["up_more_right"] > 0.0) and (mpv["cw_fast"] > 0.0):
            right_fast.append(min(mpa["up_more_right"],mpv["cw_fast"]))
            print("rule6 right_fast")
       

        # RULE 7,8
        if (mpa["up_more_left"] > 0.0) and (mpv["cw_fast"] > 0.0):
            right_slow.append(min(mpa["up_more_left"],mpv["cw_fast"]))
            print("rule7 right_slow")
        if (mpa["up_more_left"] > 0.0) and (mpv["ccw_fast"] > 0.0):
            left_fast.append(min(mpa["up_more_left"],mpv["ccw_fast"]))
            print("rule8 left_fast")
   

        # RULE 9,10 
        if (mpa["down_more_right"] > 0.0) and (mpv["ccw_slow"] > 0.0):
            right_fast.append(min(mpa["down_more_right"],mpv["ccw_slow"]))
            print("rule9 right_fast")
        if (mpa["down_more_right"] > 0.0) and (mpv["cw_slow"] > 0.0):
            stop.append(min(mpa["down_more_right"],mpv["cw_slow"]))
            print("rule10 stop")           
        

        # RULE 11,12 
        if (mpa["down_more_left"] > 0.0) and (mpv["cw_slow"] > 0.0):
            left_fast.append(min(mpa["down_more_left"],mpv["cw_slow"]))
            print("rule1 left_fast")
        if (mpa["down_more_left"] > 0.0) and (mpv["ccw_slow"] > 0.0):
            stop.append(min(mpa["down_more_left"],mpv["ccw_slow"]))
            print("rule12 stop") 
        

        # RULE 13,14
        if (mpa["down_more_right"] > 0.0) and (mpv["ccw_fast"] > 0.0):
            stop.append(min(mpa["down_more_right"],mpv["ccw_fast"]))
            print("rule13 stop")
        if (mpa["down_more_right"] > 0.0) and (mpv["cw_fast"] > 0.0):
            stop.append(min(mpa["down_more_right"],mpv["cw_fast"]))
            print("rule14 stop") 
        
       

        # RULE 15,16
        if (mpa["down_more_left"] > 0.0) and (mpv["cw_fast"] > 0.0):
            stop.append(min(mpa["down_more_left"],mpv["cw_fast"]))
            print("rule15 stop")
        if (mpa["down_more_left"] > 0.0) and (mpv["ccw_fast"] > 0.0):
            stop.append(min(mpa["down_more_left"],mpv["ccw_fast"]))
            print("rule16 stop") 
        

        # RULE 17,18
        if (mpa["down_right"] > 0.0) and (mpv["ccw_slow"] > 0.0):
            right_fast.append(min(mpa["down_right"],mpv["ccw_slow"]))
            print("rule17 right_fast")
        if (mpa["down_right"] > 0.0) and (mpv["cw_slow"] > 0.0):
            right_fast.append(min(mpa["down_right"],mpv["cw_slow"]))
            print("rule18 right_fast") 
       

        # RULE 19,20
        if (mpa["down_left"] > 0.0) and (mpv["cw_slow"] > 0.0):
            left_fast.append(min(mpa["down_left"],mpv["cw_slow"]))
            print("rule19 left_fast")
        if (mpa["down_left"] > 0.0) and (mpv["ccw_slow"] > 0.0):
            left_fast.append(min(mpa["down_left"],mpv["ccw_slow"]))
            print("rule20 left_fast") 
      

        # RULE 21,22
        if (mpa["down_right"] > 0.0) and (mpv["ccw_fast"] > 0.0):
            stop.append(min(mpa["down_right"],mpv["ccw_fast"]))
            print("rule21 stop")
        if (mpa["down_right"] > 0.0) and (mpv["cw_fast"] > 0.0):
            right_slow.append(min(mpa["down_right"],mpv["cw_fast"]))
            print("rule22 right_slow") 
        

        # RULE 23,24
        if (mpa["down_left"] > 0.0) and (mpv["cw_fast"] > 0.0):
            stop.append(min(mpa["down_left"],mpv["cw_fast"]))
            print("rule23 stop")
        if (mpa["down_left"] > 0.0) and (mpv["ccw_fast"] > 0.0):
            left_slow.append(min(mpa["down_left"],mpv["ccw_fast"]))
            print("rule24 left_slow") 
        

        # RULE 25,26,27,28,29,30
        if (mpa["up_right"] > 0.0) and (mpv["ccw_slow"] > 0.0):
            right_slow.append(min(mpa["up_right"],mpv["ccw_slow"]))
            print("rule25 right_slow")
        if (mpa["up_right"] > 0.0) and (mpv["cw_slow"] > 0.0):
            right_fast.append(min(mpa["up_right"],mpv["cw_slow"]))
            print("rule26 right_fast")
        if (mpa["up_right"] > 0.0) and (mpv["stop"] > 0.0):
            right_fast.append(min(mpa["up_right"],mpv["stop"]))
            print("rule27 right_fast")
        if (mpa["up_left"] > 0.0) and (mpv["cw_slow"] > 0.0):
            left_slow.append(min(mpa["up_left"],mpv["cw_slow"]))
            print("rule28 left_slow") 
        if (mpa["up_left"] > 0.0) and (mpv["ccw_slow"] > 0.0):
            left_fast.append(min(mpa["up_left"],mpv["ccw_slow"]))
            print("rule29 left_fast")
        if (mpa["up_left"] > 0.0) and (mpv["stop"] > 0.0):
            left_fast.append(min(mpa["up_left"],mpv["stop"]))
            print("rule30 left_fast")  
        

        # RULE 31,32,33,34
        if (mpa["up_right"] > 0.0) and (mpv["ccw_fast"] > 0.0):
            left_fast.append(min(mpa["up_right"],mpv["ccw_fast"]))
            print("rule25 left_fast")
        if (mpa["up_right"] > 0.0) and (mpv["cw_fast"] > 0.0):
            right_fast.append(min(mpa["up_right"],mpv["cw_fast"]))
            print("rule26 right_fast")
        if (mpa["up_left"] > 0.0) and (mpv["cw_fast"] > 0.0):
            right_fast.append(min(mpa["up_left"],mpv["cw_fast"]))
            print("rule27 right_fast")
        if (mpa["up_left"] > 0.0) and (mpv["ccw_fast"] > 0.0):
            left_fast.append(min(mpa["up_left"],mpv["ccw_fast"]))
            print("rule28 left_fast") 
        
        
        # RULE 35,36,37
        if (mpa["down"] > 0.0) and (mpv["stop"] > 0.0):
            right_fast.append(min(mpa["down"],mpv["stop"]))
            print("rule35 right_fast")
        if (mpa["down"] > 0.0) and (mpv["cw_fast"] > 0.0):
            stop.append(min(mpa["down"],mpv["cw_fast"]))
            print("rule36 stop")
        if (mpa["down"] > 0.0) and (mpv["ccw_fast"] > 0.0):
            stop.append(min(mpa["down"],mpv["ccw_fast"]))
            print("rule37 stop")
        

        # RULE 38,39,40,41,42
        if (mpa["up"] > 0.0) and (mpv["ccw_slow"] > 0.0):
            left_slow.append(min(mpa["up"],mpv["ccw_slow"]))
            print("rule38 left_slow")
        if (mpa["up"] > 0.0) and (mpv["ccw_fast"] > 0.0):
            left_fast.append(min(mpa["up"],mpv["ccw_fast"]))
            print("rule39 left_fast")
        if (mpa["up"] > 0.0) and (mpv["cw_slow"] > 0.0):
            right_slow.append(min(mpa["up"],mpv["cw_slow"]))
            print("rule40 right_slow")
        if (mpa["up"] > 0.0) and (mpv["cw_fast"] > 0.0):
            right_fast.append(min(mpa["up"],mpv["cw_fast"]))
            print("rule41 right_fast")
        if (mpa["up"] > 0.0) and (mpv["stop"] > 0.0):
            stop.append(min(mpa["up"],mpv["stop"]))
            print("rule42 stop")


        # RULE +
        # if (mpa["up_more_right"] > 0.0) and (mcp["stop"] > 0.0) and (mpv["cw_slow"] > 0.0) and (mcv["left_slow"]):
        #     left_fast.append(min(mpa["up_more_right"],mcp["stop"],mpv["cw_slow"],mcv["left_slow"]))
        #     print("rule42+1 left_fast")


        # if (mpa["down_left"] > 0.0) and (mcp["left_far"] > 0.0) and (mpv["cw_fast"] > 0.0) and (mcv["stop"]):
        #     right_fast.append(min(mpa["down_left"],mcp["left_far"],mpv["cw_fast"],mcv["stop"]))
        #     print("rule42+3 right_slow")


        # if (mpa["down_more_right"] > 0.0) and (mcp["left_far"] > 0.0) and (mpv["cw_fast"] > 0.0) and (mcv["left_slow"]):
        #     right_fast.append(min(mpa["down_more_right"],mcp["left_far"],mpv["cw_fast"],mcv["left_fast"]))
        #     print("rule42+3 right_fast")


        # if (mpa["down"] > 0.0) and (mcp["left_far"] > 0.0) and (mpv["cw_fast"] > 0.0) and (mcv["right_slow"]):
        #     right_fast.append(min(mpa["down"],mcp["left_far"],mpv["cw_fast"],mcv["right_slow"]))
        #     print("rule42+3 right_fast")


        # if (mpa["up_more_left"] > 0.0) and (mcp["left_near"] > 0.0) and (mpv["cw_slow"] > 0.0) and (mcv["left_fast"]):
        #     right_fast.append(min(mpa["down_left"],mcp["left_far"],mpv["cw_fast"],mcv["left_fast"]))
        #     print("rule42+3 right_fast")



        output_membership["left_fast"] = max(left_fast) if len(left_fast) > 0 else 0.0
        output_membership["left_slow"] = max(left_slow) if len(left_slow) > 0 else 0.0
        output_membership["stop"] = max(stop) if len(stop) > 0 else 0.0
        output_membership["right_slow"] = max(right_slow) if len(right_slow) > 0 else 0.0
        output_membership["right_fast"] = max(right_fast) if len(right_fast) > 0 else 0.0


        # # Testing waves ----------------------
        # output_membership["left_fast"] = 0.3
        # output_membership["left_slow"] = 0.6
        # output_membership["stop"] = 0.0
        # output_membership["right_slow"] = 0.7
        # output_membership["right_fast"] = 0.7

        # print(output_membership)
        
        return output_membership


    # ---------------------------------------------------
    # DISCRETE DEFUZZIFICATION
    # ---------------------------------------------------
    def defuzzification(self,params):
        
        # active_params = dict(filter(lambda item: item[1] > 0.0, params.items()))
        # print(active_params)
        if sum(list(params.values())) > 0.0 :

            coordinates = set()
            dots = list(np.linspace(-100,100,500))
            # dots = list(np.arange(-100,100+1,10))
            mff = self.membership_functions["force"]
            states = mff.keys()

            for s in states:
                all_zone = mff[s].keys()
                for zone in all_zone:
                    dots_slice = filter(lambda d: (zone[0] <= d) and (d <= zone[1]), dots)
                    func = mff[s][zone][1] # find y
                    for x in dots_slice:
                        y = func(x)
                        if y > params[s]:
                            y = params[s]
                        coordinates.add((x,y))
            
            coordinates = sorted(list(coordinates))
            # print(coordinates)

            Z = list()
            for dot in dots:
                temp = list()
                for coor in coordinates:
                    if coor[0] == dot:
                        temp.append(coor[1])
                Z.append((dot,max(temp)))    
            # print(Z)
            
            # Plot for Testing waves
            # xx = list()
            # yy = list()
            # for z in Z:
            #     xx.append(z[0])
            #     yy.append(z[1])

            # plt.plot(xx,yy,color='r')
            # plt.show() 
            
            # Approximation
            numerator = list()
            denominator = list()
            # dx = dots[1]-dots[0]
            for z in Z:
                numerator.append(z[1]*z[0])
                denominator.append(z[1])

            # print(numerator,denominator)
            z_star = sum(numerator)/sum(denominator)
            out = z_star
        else:
            out = 0.0
        return out



    def calculate_equation(self,p,x1,y1,x2,y2,find):
        # y = mx + b
        m = float(y2-y1)/(x2-x1)
        b = y2 - (x2*m)
        # print(" y = {:.2f}x + {}".format(m,b))
        if find == 'y':
            xp = p
            yp = (m * p) + b
            return (float(xp),float(yp))
        elif find == 'x':
            xp = (p - b)/m
            yp = p
            return (float(xp),float(yp))
        if yp < 0 :
            print("ERROR")
            error


    # ---------------------------------------------------
    # DEFUZZIFICATION WITH IMPLEMENTATION INTEGRAL 
    # ---------------------------------------------------
    def defuzzification_integral(self,params):
        
        coordinates = set()
        mff = self.membership_zone["force"]

        # Sort by main wave order
        this_order = ["left_fast","left_slow","stop","right_slow","right_fast"]
        ordered_params = list()
        for key in this_order:
            ordered_params.append((key,params[key]))
        active_params =filter(lambda item: item[1] > 0.0, ordered_params)
        
        # Dummy wave -- To count the all waves
        mff["dummy"] = (80,100,110)
        active_params = active_params + [("dummy",0.0)]

        print(active_params)


        
        last_wave = {"state":str(),"overlap":bool(),"st":tuple(),"mx":tuple(),"mu":0.0,
                     "nd":tuple(),"point1":tuple(),"point2":tuple(),"start":bool()}

        # At each level the current wave position is checked and the coordinates of the last wave are confirmed
        for item in active_params:
            state,mu = item[0],item[1]
            state_range = mff[state]
            st,mx,nd = state_range[0],state_range[1],state_range[2]

            # Get (mu(x),x) coordinates for each waves
            point1 = self.calculate_equation(mu,st,0,mx,1,'x')
            point2 = self.calculate_equation(mu,mx,1,nd,0,'x')

            # Check the first wave
            if len(last_wave["state"]) == 0:                
                start = True
                overlap = False

                
            else:
                
                # Overlap is checked               
                if (last_wave["nd"][0] > st):
                    # Find the intersection of two waves 
                    collide = self.collide(last_wave["mx"],last_wave["nd"],(st,0.0),(mx,1.0))
                    
                    # Checked the different intersection modes for each section of the last_wave and current_wave
                    # Intersection above the cross section of two waves
                    if (collide[1] > last_wave["mu"]) and (collide[1] > mu):
                        # Position of the points relative to each other
                        

                        if last_wave["mu"] > mu:
                            # Position of the points relative to each other
                            #---------------------------------------------------
                            #               .(collide)     
                            # .(last_wave point2)               
                            #               .(point1)    
                            #---------------------------------------------------
                            #         .
                            #   ____.
                            #  /     \.____
                            # /            \
                            #===================================================
                            # print("-----_____")
                            new_collide = self.collide(last_wave["mx"],last_wave["nd"],point1,point2)
                            coordinates.add(last_wave["point2"])
                            coordinates.add(new_collide) # change positon point1
                            # If the last_wave was the start wave, confirm the starting point
                            if last_wave["start"] == True:
                                coordinates.add(last_wave["st"])
                            # If the left side of last_wave does not overlap with its last last_wave, confirm its point1 ( i know.. )
                            if last_wave["overlap"] == False:
                                coordinates.add(last_wave["point1"])
                            # The starting point of the next wave can not be confirmed
                            start = False
                            # The  point1 of the next wave can not be confirmed
                            overlap = True

                        elif last_wave["mu"] < mu:
                            # Position of the points relative to each other
                            #---------------------------------------------------
                            #              .(collide)     
                            # .(point1)               
                            #              .(last_wave point2)    
                            #--------------------------------------------------- 
                            #      .
                            #        .____
                            #  ____./     \
                            # /            \
                            #===================================================
                            # print("_____-----")
                            new_collide = self.collide(last_wave["point1"],last_wave["point2"],(st,0.0),(mx,1.0))
                            coordinates.add(new_collide)# change positon point2
                            if last_wave["start"] == True:
                                coordinates.add(last_wave["st"])
                            if last_wave["overlap"] == False:
                                coordinates.add(last_wave["point1"]) 
                            start = False
                            overlap = False

                        elif last_wave["mu"] == mu:
                            # Position of the points relative to each other
                            #---------------------------------------------------
                            #         .(collide)
                            #                 
                            # .(last_wave point2)  .(point1)
                            #---------------------------------------------------   
                            #       .
                            #  .____.____
                            # /           \
                            #===================================================
                            # print("----- -----")
                            if last_wave["start"] == True:
                                coordinates.add(last_wave["st"])
                            if last_wave["overlap"] == False:
                                coordinates.add(last_wave["point1"]) 
                            start = False
                            overlap = False

                    # Intersection under the cross section of two waves
                    else:
                        if (last_wave["point2"][0] <= collide[0]) and (collide[0] <= point1[0]):
                            # Position of the points relative to each other
                            #---------------------------------------------------
                            # .(last_wave point2)   .(point1)
                            # 
                            #         .(collide)     
                            #--------------------------------------------------- 
                            #   ____.   .____
                            #  /     \./     \
                            # /               \
                            #===================================================
                            # print("---\\/---")
                            coordinates.add(collide)
                            coordinates.add(last_wave["point2"])
                            if last_wave["start"] == True:
                                coordinates.add(last_wave["st"])
                            if last_wave["overlap"] == False:
                                coordinates.add(last_wave["point1"])
                            start = False
                            overlap = False

                        elif (last_wave["point2"][0] < collide[0]) and (point1[0] < collide[0]):
                            # Position of the points relative to each other
                            #---------------------------------------------------
                            # .(last_wave point2)      
                            #             .(collide)              
                            # .(point1)    
                            #--------------------------------------------------- 
                            #   ____. 
                            #  /     \.____
                            # /     .      \
                            #===================================================
                            # print("-----_____")
                            new_collide = self.collide(last_wave["mx"],last_wave["nd"],point1,point2)
                            coordinates.add(last_wave["point2"])
                            coordinates.add(new_collide) # change positon point p1
                            if last_wave["start"] == True:
                                coordinates.add(last_wave["st"])
                            if last_wave["overlap"] == False:
                                coordinates.add(last_wave["point1"])
                            start = False
                            overlap = True

                        elif (collide[0] < last_wave["point2"][0]) and (collide[0] < point1[0]):
                            # Position of the points relative to each other
                            #---------------------------------------------------
                            #         .(last_wave point2)      
                            # .(collide)              
                            #         .(point1)    
                            #--------------------------------------------------- 
                            #        .____
                            #  ____./     \
                            # /      .     \
                            #===================================================
                            # print("_____-----")
                            new_collide = self.collide(last_wave["point1"],last_wave["point2"],(st,0.0),(mx,1.0))
                            coordinates.add(new_collide) # change positon point p2
                            if last_wave["start"] == True:
                                coordinates.add(last_wave["st"])
                            if last_wave["overlap"] == False:
                                coordinates.add(last_wave["point1"])
                            start = False
                            overlap = False

                # Overlap is checked               
                else:
                    #       
                    #  ____    ____
                    # /    \  /    \
                    #===================================================
                    # print("No Overlap")
                    coordinates.add(last_wave["point2"])
                    coordinates.add(last_wave["nd"])
                    if last_wave["start"] == True:
                        coordinates.add(last_wave["st"])
                    if last_wave["overlap"] == False:
                        coordinates.add(last_wave["point1"])
                    start = True
                    overlap = False

            # Update last_wave
            last_wave["st"] = (st,0.0)
            last_wave["mx"] = (mx,1.0)
            last_wave["nd"] = (nd,0.0)
            last_wave["point1"] = point1
            last_wave["point2"] = point2
            last_wave["state"] = state 
            last_wave["start"] = start 
            last_wave["overlap"] = overlap
            last_wave["mu"] = mu      
        

        coordinates = sorted(list(coordinates))
        print(coordinates)

        # # Draws the confirmed points for testing
        # xx = list()
        # yy = list()
        # for c in coordinates:
        #     xx.append(c[0])
        #     yy.append(c[1])

        # plt.plot(xx,yy,color='r')
        # plt.show() 
        
        # Remove dummy wave from orginal membership_zone
        del mff["dummy"]
        
        # center-of-gravity method
        out = self.center_of_gravity(coordinates)    
        return out


    # Given two points on each line segment
    def collide(self,(x1,y1),(x2,y2),(x3,y3),(x4,y4)):      
        t = (((x1-x3)*(y3-y4))-((y1-y3)*(x3-x4)))/(((x1-x2)*(y3-y4))-((y1-y2)*(x3-x4)))
        # u = (((x1-x3)*(y1-y2))-((y1-y3)*(x1-x2)))/(((x1-x2)*(y3-y4))-((y1-y2)*(x3-x4)))
        px = x1 + t*(x2-x1)
        py = y1 + t*(y2-y1)
        return (px,py)


    def center_of_gravity(self,coors):

        def numerator((x1,y1),(x2,y2)):   
            m = float(y2-y1)/(x2-x1)
            b = float(y2 - (x2*m))
            # print("N: y = {}/{}x^2 + {}x".format(y2-y1,x2-x1,b))
            neq = lambda x: (m*x*x) + b*x
            return neq
        
        def denominator((x1,y1),(x2,y2)):
            m = float(y2-y1)/(x2-x1)
            b = y2 - (x2*m)
            # print("D: y = {}/{}x + {}".format(y2-y1,x2-x1,b))
            deq = lambda x: (m*x) + b   
            return deq

        numerator_sum = 0.0
        denominator_sum = 0.0


        for i in range(len(coors)-1):
            
            func = numerator(coors[i],coors[i+1])
            slice_numerator_integrand = quad(func, coors[i][0], coors[i+1][0])

            func = denominator(coors[i],coors[i+1])
            slice_denominator_integrand = quad(func, coors[i][0], coors[i+1][0])

            numerator_sum += slice_numerator_integrand[0]
            denominator_sum += slice_denominator_integrand[0]
            # print (slice_numerator_integrand[0],slice_denominator_integrand[0])

        # print(numerator_sum,denominator_sum)
        if denominator_sum == 0.0:
            out = 0.0
        else:
            out = numerator_sum / denominator_sum
        return out 

        

    # =========================================================
    def decide(self, world):
        
        output = self._make_output()
        parameters = self._make_input(world)
        self.initialize_membership_zone()
        input_membership_parameters = self.fuzzification(parameters)
        output_membership_parameters = self.inference(input_membership_parameters)
        # output['force'] = self.defuzzification_integral(output_membership_parameters)
        output['force'] = self.defuzzification(output_membership_parameters)
        
        return output['force']


    # def decide(self, world):
    #     output = self._make_output()
    #     self.system.calculate(self._make_input(world), output)
    #     return output['force']