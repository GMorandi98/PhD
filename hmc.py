#!/usr/bin/env python3
#
# Authors: Christoph Lehner, Mattia Bruno 2021
#
#
import gpt as g
import sys, os
import numpy
import matplotlib.pyplot as plt
from pslog import LOG
import bison

run = 2
name = f"run{run}"
seed = f'my_random {name}'
start = 36
end = (start + 1) + 10
dtrj = 4

# trajectories = (end-start)* dtrj

##########

log = None
if g.rank()==0:
    log = LOG(f"{name}_log")
    log.make_title('DWF Mobius Nf=2 HMC', os.getlogin())
    log.start_capture()
    
g.message("Simulation program of the Nf=2 theory with HMC algorithm")

grid = g.grid([16,16,16,32], g.double)
rng = g.random(seed)

U = g.load(f'cnfgs/ckpoint_lat.{start}')
# load

# conjugate momenta
mom = g.group.cartesian(U)

# Log
g.message(f"Lattice = {grid.fdimensions}")
g.message("Actions:")
# action for conj. momenta
a0 = g.qcd.scalar.action.mass_term()
g.message(f" - {a0.__name__}")

# action
beta = 2.13
a1 = g.qcd.gauge.action.iwasaki(beta)
g.message(f" - {a1.__name__}")

# Mobius fermions
# target 0.0362
mass=[0.0362, 0.1, 1.0]
M5=[1.8]*len(mass)
Ls=[24]*len(mass)
b=[1.5]*len(mass)
c=[0.5]*len(mass)
npf=len(mass)-1

DWF = []
g.message(f" - Hasenbush Mobius DWF actions")
for i in range(len(mass)):
    DWF.append(g.qcd.fermion.mobius(U, M5=M5[i], mass=mass[i], Ls=Ls[i], b=b[i], c=c[i],
                                    boundary_phases=[1,1,1,-1]))
    g.message(f"   - m={mass[i]}, M5={M5[i]}, Ls={Ls[i]}, b={b[i]}, c={c[i]}")


inv = g.algorithms.inverter
pc = g.qcd.fermion.preconditioner
g.default.set_verbose("cg_convergence", False)
g.default.set_verbose("cg", True)

solver0 = inv.cg({"eps": 1e-10, "maxiter": 2048})
solver1 = inv.cg({"eps": 1e-8, "maxiter": 2048})
solver2 = inv.cg({"eps": 1e-6, "maxiter": 2048})

fact = g.qcd.pseudofermion.action.two_flavor_ratio_evenodd_schur
a2 = []
a2frc = []
fields = []

for i in range(npf):
    g.message(f" - Determinant 2flavor ratio {mass[i]} / {mass[i+1]}; pseudo-fermion {i}")
    a2.append(fact([DWF[i], DWF[i+1]], solver0))
    a2frc.append(fact([DWF[i], DWF[i+1]], solver1))
    fields.append(U+[g.vspincolor(DWF[0].F_grid_eo)])


def hamiltonian(draw):
    h = 0
    if draw:
        rng.normal_element(mom)
        h += a0(mom)
        h += a1(U)
        for i in range(npf):
            h += a2[i].draw(fields[i], rng)
    else:
        h += a0(mom)
        h += a1(U)
        for i in range(npf):
            h += a2[i](fields[i])
    return h

# molecular dynamics

sympl = g.algorithms.integrator.symplectic
sympl.set_verbose(True)
#dbg = g.algorithms.integrator.symplectic.debug_force
dbg = sympl.log()

ip1 = sympl.update_p(mom, dbg(lambda: a1.gradient(U, U), 'gauge'))
ip2 = []
for i in range(npf):
    ip2 += [sympl.update_p(mom, dbg(lambda i=i: a2frc[i].gradient(fields[i], U), f'pseudo-fermion {i}'))]
#ip2sl = sympl.update_p(mom, lambda: a2sloppy.gradient(fields, U))
iq = sympl.update_q(U, lambda: a0.gradient(mom, mom))
#ip2_fg = sympl.update_p_force_gradient(U, iq, mom, ip2, ip2sl)

# integrator
#mdint = sympl.OMF2_force_gradient(7, ip2, sympl.OMF4(1, ip1, iq), ip2_fg)
mdint = sympl.OMF2(8, ip2[0], 
                   sympl.OMF2(1, ip2[1:], 
                              sympl.OMF4(1, ip1, iq)))
# mdint = sympl.OMF2(1, ip2, sympl.OMF4(1, ip1, iq))

g.message(f"Integration scheme:\n{mdint}")

# metropolis
metro = g.algorithms.markov.metropolis(rng)

# MD units
tau = 1.0 #1.4
g.message(f"tau = {tau} MD units")


def hmc(tau, mom):
    accrej = metro(U)
    h0 = hamiltonian(True)
    mdint(tau)
    h1 = hamiltonian(False)
    return [accrej(h1, h0), h1 - h0]


# production
history = []
plaq = []
for i in range(start+1,end):
    dbg.reset()
    for _ in range(dtrj):
        history += [hmc(tau, mom)]
    plaq += [g.qcd.gauge.plaquette(U)]
    g.message(f"Trajectory {i * dtrj}, {history[-1]}, {plaq[-1]:g}")
    g.save(f"cnfgs/ckpoint_lat.{i * dtrj}", U, g.format.nersc())
    if g.rank()==0:
        for key in dbg.grad:
            plt.figure()
            plt.title(key)
            plt.plot(dbg.get(key))
            log.pyplot_figure(plt)
        log.flush()


history = numpy.array(history)
plaq = numpy.array(plaq)
             
if g.rank()==0:
    plt.figure()
    plt.title('|dH|')
    plt.plot(numpy.abs(history[:,1]))
    plt.yscale('log')
    log.pyplot_figure(plt)

    plt.figure()
    plt.title('accept')
    plt.plot(histotry[:,0])
    log.pyplot_figure(plt)

g.message(f"Acceptance rate = {numpy.mean(history[:,0]):.2f}")
g.message(f"<|dH|> = {numpy.mean(numpy.abs(history[:,1])):.4e}")

if g.rank()==0:
    plt.figure()
    plt.title('Plaquette')
    plt.plot(plaq)
    log.pyplot_figure(plt)

g.message(f"<plaq>   = {numpy.mean(plaq)}")

g.message("Terminating program")

if g.rank()==0:
    log.end_capture()
    log.save(pdf=True)

    xax = numpy.arange((start+1)*dtrj,end*dtrj)*tau
    bison.save(f'{name}.mc.dat', {
        'accept': numpy.c_[xax, history[:,0]],
        'dH': numpy.c_[xax, history[:,1]],
        'plaq': numpy.c_[numpy.arange(start+1,end)*tau, plaq],
    })
