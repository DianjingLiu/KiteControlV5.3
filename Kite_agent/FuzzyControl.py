import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl

import matplotlib.pyplot as plt
from pdb import set_trace

def build_fuzzy_variables():
    # Sparse universe makes calculations faster, without sacrifice accuracy.
    # Only the critical points are included here; making it higher resolution is
    # unnecessary.
    universe_theta = np.array([-0.15, -0.1, -0.05, 0, 0.05, 0.1])
    universe_vtheta= np.array([-0.125, -0.075, -0.025, 0.025, 0.075, 0.125]) 
    universe_phi   = np.array([-0.2, -0.1, -0.05, 0.05, 0.1, 0.2])
    universe_vphi  = np.array([-0.125, -0.075, -0.025, 0.025, 0.075, 0.125])
    universe_output= np.arange(-1, 1.1, 1/6)
    # Create the three fuzzy variables - four inputs, one output
    theta = ctrl.Antecedent(universe_theta , 'theta')
    vtheta= ctrl.Antecedent(universe_vtheta, 'vtheta')
    phi   = ctrl.Antecedent(universe_phi   , 'phi')
    vphi  = ctrl.Antecedent(universe_vphi  , 'vphi')
    output= ctrl.Consequent(universe_output, 'output')
    
    # Set membership function
    # fuzz.trapmf(support_x,[0,0,20,30])
    sup = theta.universe
    theta['low']    = fuzz.trapmf(theta.universe, [sup[0], sup[0], sup[1], sup[2]])
    theta['normal'] = fuzz.trapmf(theta.universe, [sup[1], sup[2], sup[3], sup[4]])
    theta['high']   = fuzz.trapmf(theta.universe, [sup[3], sup[4], sup[5], sup[5]])
    
    sup = vtheta.universe
    vtheta['falling'] = fuzz.trapmf(vtheta.universe, [sup[0], sup[0], sup[1], sup[2]])
    vtheta['stable']  = fuzz.trapmf(vtheta.universe, [sup[1], sup[2], sup[3], sup[4]])
    vtheta['rising']  = fuzz.trapmf(vtheta.universe, [sup[3], sup[4], sup[5], sup[5]])
    
    sup = phi.universe
    phi['left']   = fuzz.trapmf(phi.universe, [sup[0], sup[0], sup[1], sup[2]])
    phi['center'] = fuzz.trapmf(phi.universe, [sup[1], sup[2], sup[3], sup[4]])
    phi['right']  = fuzz.trapmf(phi.universe, [sup[3], sup[4], sup[5], sup[5]])
    
    sup = vphi.universe
    vphi['goleft']  = fuzz.trapmf(vphi.universe, [sup[0], sup[0], sup[1], sup[2]])
    vphi['stable']  = fuzz.trapmf(vphi.universe, [sup[1], sup[2], sup[3], sup[4]])
    vphi['goright'] = fuzz.trapmf(vphi.universe, [sup[3], sup[4], sup[5], sup[5]])

    output.automf(names=['lefthard','left','leftslow','hold','rightslow','right','righthard'])
    return theta, vtheta, phi, vphi, output
    
    
def build_rule1(theta, vtheta, phi, vphi, output):
    rule1 = ctrl.Rule(vtheta['stable'] & vphi['stable'], 
        output['hold'], 'rule1')
    
    rule2 = ctrl.Rule(theta['low'] & vphi['goleft'] & vtheta['falling'], 
        output['righthard'], 'rule2')
    
    rule3 = ctrl.Rule(theta['low'] & vphi['goright'] & vtheta['falling'], 
        output['lefthard'], 'rule3')
    
    rule4 = ctrl.Rule(theta['low'] & vtheta['rising'], 
        output['hold'], 'rule4')
    
    rule5 = ctrl.Rule(phi['left'] & vphi['goleft'], 
        output['right'], 'rule5')
    
    rule6 = ctrl.Rule(phi['left'] & vtheta['rising'], 
        output['rightslow'], 'rule6')
    
    rule7 = ctrl.Rule(phi['left'] & vphi['goright'], 
        output['hold'], 'rule7')
    
    rule8 = ctrl.Rule(phi['left'] & vphi['goright'] & theta['low'], 
        output['left'], 'rule8')
    
    rule9 = ctrl.Rule(phi['right'] & vphi['goright'], 
        output['left'], 'rule9')
    
    rule10= ctrl.Rule(phi['right'] & vtheta['rising'], 
        output['leftslow'], 'rule10')
    
    rule11= ctrl.Rule(phi['right'] & vphi['goleft'], 
        output['hold'], 'rule11')
    
    rule12= ctrl.Rule(phi['right'] & vphi['goleft'] & vtheta['falling'], 
        output['right'], 'rule12')
    
    rule13= ctrl.Rule(phi['center'] & vphi['goleft'] & vtheta['falling'], 
        output['rightslow'], 'rule13') #????
    
    rule14= ctrl.Rule(phi['center'] & vphi['goright'] & vtheta['falling'], 
        output['leftslow'], 'rule14')
    
    rule15= ctrl.Rule(theta['high'] & vphi['goright'] & vtheta['rising'], 
        output['right'], 'rule15')
    
    rule16= ctrl.Rule(theta['high'] & vphi['goleft'] & vtheta['rising'], 
        output['left'], 'rule16')
    return [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, 
            rule9, rule10, rule11, rule12, rule13, rule14, rule15, rule16]

def build_rule2(theta, vtheta, phi, vphi, output):
    rule1 = ctrl.Rule(phi['left'] & vphi['goleft'], output['right'], 'rule1')
    rule2 = ctrl.Rule(phi['right'] & vphi['goright'], output['left'], 'rule2')
    rule3 = ctrl.Rule(phi['center'] |
        (phi['left'] & ~vphi['goleft']) |
        (phi['right'] & ~vphi['goright']) ,
         output['hold'], 'rule3')
    # save when heading down
    rule4 = ctrl.Rule(theta['low'] & vtheta['falling'] & phi['left'], output['righthard'], 'rule4')
    rule5 = ctrl.Rule(theta['low'] & vtheta['falling'] & phi['right'], output['lefthard'], 'rule5')
    rule6 = ctrl.Rule(theta['low'] & vtheta['falling'] & phi['center'], output['lefthard'], 'rule6')
    return [rule1, rule2, rule3, rule4, rule5, rule6]

def build_fuzzy_system(rules):
    system = ctrl.ControlSystem(rules=rules)
    fz_agent = ctrl.ControlSystemSimulation(system)
    return fz_agent

# input: state: n pieces of [x, y, phi, l1, l2, v1, v2, T1, T2, J]
# output: joystick
class fuzzy_agent():
    def __init__(self):
        theta, vtheta, phi, vphi, output = build_fuzzy_variables()
        rules1 = build_rule1(theta, vtheta, phi, vphi, output)
        self.agent1 = build_fuzzy_system(rules1)
    
        rules2 = build_rule2(theta, vtheta, phi, vphi, output)
        self.agent2 = build_fuzzy_system(rules2)

        self.theta = theta
        self.vtheta = vtheta
        self.phi = phi
        self.vphi = vphi
        self.output = output

        self.scale_phi = 0.5#0.6
        self.scale_vphi = 0.25#1
        self.dist_l = 520
        self.dist_h = 350
        self.scale_vtheta = 1

        self.ProgramRunSpeed = 0.04
        # each piece of record is [x, y, phi, l1, l2, v1, v2, T1, T2, J]
        self.record_len = 7
        
    def control(self, state):
        in_theta, in_vtheta, in_phi, in_vphi = self.process(state)
        print(in_theta, in_vtheta, in_phi, in_vphi)
        return self.fuzz_control(in_theta, in_vtheta, in_phi, in_vphi)

    # get theta, vtheta, phi, vphi from state input
    def process(self, state):
        record = state.reshape((-1,self.record_len))
        
        x = record[:,0]
        y = record[:,1]
        theta, phi = self.calc_angle(x, y)

        #vtheta = (theta - theta1) / self.ProgramRunSpeed
        #vphi = 0.2 * (phi - phi1) / self.ProgramRunSpeed
        vtheta = np.mean(np.diff(theta)) / self.ProgramRunSpeed * self.scale_vtheta
        vphi = np.mean(np.diff(phi)) / self.ProgramRunSpeed * self.scale_vphi / self.scale_phi
        return float(theta[-1]), float(vtheta), float(phi[-1]), float(vphi)
    def calc_angle(self, x, y):
        x = x * 1000 + 700
        y = y * 1000 + 700
        x0 = 1504 *0.5
        y0 = 1496 *0.5
        dist = np.array((x - x0)**2 + (y - y0)**2, dtype=np.float32)
        dist = np.sqrt(dist)
        dx = (x-x0)/dist
        dy = (y-y0)/dist
        phi = np.arctan2(dx,dy) * self.scale_phi
        #theta = -1/1600 * dist + 0.1 + 3.0/16
        theta = (self.theta.universe[-1] - self.theta.universe[0])/(self.dist_h - self.dist_l) * (dist - self.dist_l) + self.theta.universe[0]
        #print(dist)
        return theta, phi

    def fuzz_control(self, in_theta, in_vtheta, in_phi, in_vphi):
        in_theta = np.clip(in_theta, self.theta.universe[0], self.theta.universe[-1])
        in_vtheta= np.clip(in_vtheta,self.vtheta.universe[0], self.vtheta.universe[-1])
        in_phi   = np.clip(in_phi,   self.phi.universe[0], self.phi.universe[-1])
        in_vphi  = np.clip(in_vphi,  self.vphi.universe[0], self.vphi.universe[-1])
        try:
            self.agent1.input['theta']  = in_theta
            self.agent1.input['phi']    = in_phi
            self.agent1.input['vtheta'] = in_vtheta
            self.agent1.input['vphi']   = in_vphi
            self.agent1.compute()
            joystick = self.agent1.output['output']
            #self.output.view(sim=self.agent1)
        except ValueError:
            print('Rules 1 are not activated.')
            self.agent2.input['theta']  = in_theta
            self.agent2.input['phi']    = in_phi
            self.agent2.input['vtheta'] = in_vtheta
            self.agent2.input['vphi']   = in_vphi
            self.agent2.compute()
            joystick = self.agent2.output['output']
            #self.output.view(sim=self.agent2)
        return joystick


if __name__ == '__main__':  
    in_theta = -0.15
    in_phi = -0.1
    in_vtheta = -0.1#-0.1
    in_vphi = -0.8#-0.05
    fz_agent = fuzzy_agent()
    joystick = fz_agent.fuzz_control(in_theta, in_vtheta, in_phi, in_vphi)
    print(joystick)
    #set_trace()

    """
    theta, vtheta, phi, vphi, output = build_fuzzy_variables()
    rules1 = build_rule1(theta, vtheta, phi, vphi, output)
    agent1 = build_fuzzy_system(rules1)

    rules2 = build_rule2(theta, vtheta, phi, vphi, output)
    agent2 = build_fuzzy_system(rules2)

    try:
        agent1.input['theta']  = in_theta
        agent1.input['phi']    = in_phi
        agent1.input['vtheta'] = in_vtheta
        agent1.input['vphi']   = in_vphi
        agent1.compute()
        joystick = agent1.output['output']

        output.view(sim=agent1)
        print(joystick)
        set_trace()
    except ValueError:
        print('Rules 1 are not activated.')
        agent2.input['theta']  = in_theta
        agent2.input['phi']    = in_phi
        agent2.input['vtheta'] = in_vtheta
        agent2.input['vphi']   = in_vphi
        agent2.compute()
        joystick = agent2.output['output']

        output.view(sim=agent2)
        print(joystick)
        set_trace()
    """

