#
#!/usr/bin/env python3
#
# Author: Gabriele Morandi, November 2023
#
# HMC and Wilson Flow for Yang-Mills SU(3) theory
#
import gpt as g
import sys, os
import numpy as np 
import bison


g.default.set_verbose(False)
g.default.set_verbose("step_size", False)


params = g.params("run_params.txt")

#################################
### General global parameters ###
#################################
L1, L2, L3, T = params["Lattice"]
beta       = params["beta"]
tau        = params["tau"]           # length of MD trajectory
Ns         = params["Ns"]            # integration steps of OMF4 for HMC
Ntherm     = params["Ntherm"]        # thermalization trajectories
therm_step = params["therm_step"]

# Note: Ntherm = 100 from empirical analysis, but it needs to be verified a posteriori,
#       once you have a quite long run, that Ntherm ~ 4/5 tau_int (the longest of the run). 

Ntraj   = params["Ntraj"]         # trajectories for data production
MC_step = params["MC_measure_step"]
WF_evol = params["WF_evolutions"]
WF_step = params["WF_measure_step"]
eps_WF  = params["eps_WF"]
seed    = params["seed"]


grid = g.grid([L1, L2, L3, T], g.double)
rng = g.random(seed)

U = g.qcd.gauge.unit(grid)
# rng.normal_element(U)
rng.normal_element(U)

# links conjugate momenta
pi = g.group.cartesian(U)


#########################################
### General informations for Log file ###
#########################################
g.message(f"  Lattice  =  {grid.fdimensions}")

a0 = g.qcd.scalar.action.mass_term() # action for conjugate momenta
a1 = g.qcd.gauge.action.wilson(beta) # Wilson action

g.message()
g.message("  Actions: ")
g.message(f"   -- {a0.__name__}")
g.message(f"   -- {a1.__name__} with coupling beta = {beta}")


####################################
### HMC setup: MD and Metropolis ###
####################################

# molecular dynamics
sympl = g.algorithms.integrator.symplectic

ip = sympl.update_p(pi, lambda: a1.gradient(U, U))
iq = sympl.update_q(U,  lambda: a0.gradient(pi, pi))

# integrator
mdint = sympl.OMF4(Ns, ip, iq)
g.message()
g.message(f"  Integration scheme:\n{mdint}")

# metropolis
metro = g.algorithms.markov.metropolis(rng)

# MD units
g.message(f"  tau = {tau} MD units")



def HMC(tau, pi):
    rng.normal_element(pi)
    accrej = metro(U)
    S_i = a1(U)
    H_i = a0(pi) + S_i
    mdint(tau) # MD evolution
    S_f = a1(U) 
    H_f = a0(pi) + S_f 
    return [accrej(H_f, H_i), H_f - H_i]
    #return [accrej(H_f, H_i), H_i, H_f, S_i, S_f]



######################
### Thermalization ###
######################
g.message()
g.message()
g.message(f"----------------------------------------------------------------------------------------")
g.message(f"------------------------------ Starting thermalization... ------------------------------")
g.message(f"----------------------------------------------------------------------------------------")
g.message()
g.message()

timer = g.timer("HMC")
for i in range(therm_step):
    h    = []
    plaq = []
    for j in range(Ntherm // therm_step):
        timer("therm traj")
        h    += [HMC(tau, pi)]
        plaq += [g.qcd.gauge.plaquette(U)]
        timer()
    h = np.array(h) 
    plaq = np.array(plaq)
    dt_traj = timer.time['therm traj'].dt_sum / ((Ntherm//therm_step)*(i+1))
    g.message(f"  ---- {(i+1)*(Ntherm//therm_step)/Ntherm*100:.0f}% of Thermalization completed  :  avg. time per traj. = {dt_traj:.2f} secs, {dt_traj / 60:.2f} mins ----  ")
    g.message(f"       < Plaq > = {np.mean(plaq):.4f},  Acceptance = {np.mean(h[:, 0])*100:.0f}%,  < |dH| > = {np.mean(np.abs(h[:, 1])):.3e}  ")
    g.message()
g.message(f"  Thermalization time: {timer.time['therm traj'].dt_sum:.2f} secs ({timer.time['therm traj'].dt_sum / 60:.2f} mins, {timer.time['therm traj'].dt_sum / 3600:.2f} hrs), ")
g.message() 

traj_time = timer.time['therm traj'].dt_sum / Ntherm



####################################
## Wilson Flow & Data Production ###
####################################
g.message()
g.message(f"--------------------------------------------------------------------------------------")
g.message(f"---------------------------------- Data Production: ----------------------------------")
g.message(f"--------------------------------------------------------------------------------------")
g.message()
g.message()
g.message(f"  We compute in total {Ntraj} trajectories:  ")
g.message()
g.message()


history = [] # for entire MC history, contains dH and Acc/Rej. 
data_Plaq  = []
data_Energy  = [] 
for i in range(1, Ntraj+1):
    timer("data traj")
    history += [HMC(tau, pi)] # HMC evolves U field
    timer()
    dt_traj = timer.time['data traj'].dt_last
    g.message(f"  ------ MD trajectory {i} | time = {dt_traj:.2f} secs ({dt_traj/60:.2f} mins)  ------  ")
    g.message(f"         Acc/Rej = {history[-1][0]},  dH = {history[-1][1]:.2e}  ")
    g.message()
    if (i % MC_step == 0):
        timer("WF & measurement")
        g.message(f"      ---- | Measurement at MC time {i} | ----")
        g.message()
        WF_Plaq   = [g.qcd.gauge.plaquette(U)]
        WF_Energy = [g.qcd.gauge.energy_density(U).real] # data saved also at WF time = 0. 
        V = U.copy()
        # Wilson flow:
        for n in range(1, WF_evol+1): # WF evolves V, which is a copy of U 
            g.message(f"         -- WF evolution with eps = {eps_WF} --")
            V = g.qcd.gauge.smear.wilson_flow(V, epsilon=eps_WF) # ~ 3 secs for each eps_WF
            if (n % WF_step == 0):
                # measurement at given flow time: t = n * eps_WF
                g.message()
                g.message(f"            -- | Measurement at flow time t = {n * eps_WF:.2f} | --")
                WF_Plaq   += [g.qcd.gauge.plaquette(V)]
                WF_Energy += [g.qcd.gauge.energy_density(V).real]
                g.message(f"            --   Plaq(V) = {WF_Plaq[-1]:.5f},  E(V) = {WF_Energy[-1]:.4f} --")
                g.message()
        timer()
        dt_WF = timer.time['WF & measurement'].dt_last
        g.message(f"           total time WF & measurements = {dt_WF:.2f} secs ({dt_WF/60:.2f} mins) ")
        g.message()

        data_Plaq.append(WF_Plaq)
        data_Energy.append(WF_Energy)

tot = timer.time['data traj'].dt_sum + timer.time['WF & measurement'].dt_sum
g.message()
g.message(f"  Data production total time = {tot:.2f} secs ({tot/60:.2f} mins, {tot/3600:.2f} hrs)")
g.message()

WF_time = timer.time['WF & measurement'].dt_sum / (Ntraj // MC_step)
history      = np.array(history)
data_Plaq    = np.array(data_Plaq)
data_Energy  = np.array(data_Energy)

obs = {
    'Acc/Rej & dH'       :       history, \
    'WF Energy density'  :       data_Energy, \
    'WF Plaquette'       :       data_Plaq
}



# g.qcd.gauge.energy_density(U, field=True)            # Energy density Clover
# g.qcd.gauge.topological_charge(U, field=True)        # Usual def Top. Charge (~ eps_mu,nu,rho,sigma tr{F_mu,nu F_rho,sigma})
# g.qcd.gauge.topological_charge_5LI(U, field=True)    # O(a^4) improved def. Top. Charge


# g.message()
# if g.rank() == 0:
#     file = bison.FILE(f"../../Data_Analysis/Pure_gauge/WF_beta{int(beta*1e2)}_lat{L1}^3x{T}_Ntraj{Ntraj}-{MC_step}_WFevol{WF_evol}-{WF_step}_seed{int(seed)}.dat", mode="w")
#     file.write('beta coupling', beta)
#     file.write('Lattice size: L1', L1)
#     file.write('Lattice size: L2', L2)
#     file.write('Lattice size: L3', L3)
#     file.write('Lattice size: T', T)
#     file.write('Lattice volume', grid.fsites)
#     file.write('Number of steps of OMF4 integrator', Ns)
#     file.write('Length of each MD trajectory', tau)
#     file.write('Thermalization trajectories', Ntherm)
#     file.write('MD trajectories', Ntraj)
#     file.write('MC measure step', MC_step)
#     file.write('epsilon WF', eps_WF)
#     file.write('WF evolutions from t = 0', WF_evol)
#     file.write('WF measure step', WF_step)
#     file.write('History & Measurments', obs)
#     file.write('Avg. time in secs single traj.', traj_time)
#     file.write(f'Avg. time in secs WF(eps={eps_WF}, evol={WF_evol}) + Measurements', WF_time)
#     file.close()
# g.message()



