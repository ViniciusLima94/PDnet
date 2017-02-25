# Potjans & Diesmann cortical microcircuit model implemented on Brian2.

from brian2 import *
import numpy as np
#set_device('genn')

matplotlib.use('Agg')

# Seting the simulation to run with openmp
set_device('cpp_standalone', directory='PD')
prefs.devices.cpp_standalone.openmp_threads = 4

###############################################################################
# Simulation parameters
###############################################################################
defaultclock.dt = 0.1*ms    # Timestep of numerical integration method
tsim = 1*second             # Time of simulation

##############################################################################
# Network parameters
###############################################################################
N = 80000 # Total population
# Fraction of neurons in each layer
#       2/3e   2/3i   4e     4i     5e     5i     6e     6i
frac = [.2680, .0756, .2840, .0710, .0628, .0138, .1866, .0382 ]

#Number of neurons per layer
n_layer = [N*f for f in frac]
n_layer = [int(round(n_pop)) for n_pop in n_layer]
# Reescale factor
rf = 80000.0 / N
nn_cum = cumsum(n_layer)
# Background number per layer
bg_layer = [1600, 1500 ,2100, 1900, 2000, 1900, 2900, 2100]
bg_layer = [bg_pop/rf for bg_pop in bg_layer]   # Reescaling
# Prob. connection table

table = [[0.1009,  0.1689, 0.0437, 0.0818, 0.0323, 0.,     0.0076, 0.    ],
        [0.1346,   0.1371, 0.0316, 0.0515, 0.0755, 0.,     0.0042, 0.    ],
        [0.0077,   0.0059, 0.0497, 0.135,  0.0067, 0.0003, 0.0453, 0.    ],
        [0.0691,   0.0029, 0.0794, 0.1597, 0.0033, 0.,     0.1057, 0.    ],
        [0.1004,   0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.    ],
        [0.0548,   0.0269, 0.0257, 0.0022, 0.06,   0.3158, 0.0086, 0.    ],
        [0.0156,   0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
        [0.0364,   0.001,  0.0034, 0.0005, 0.0277, 0.008,  0.0658, 0.144 ]]

###############################################################################
# Synapses parameters
###############################################################################
d_ex = 1.5*ms      # Excitatory delay
std_d_ex = 0.75*ms # Std. Excitatory delay
d_in = 0.80*ms      # Inhibitory delay
std_d_in = 0.4*ms  # Std. Inhibitory delay

w_ex = rf*0.3512*mV     # Excitatory weight
std_w_ex = rf*0.0352*mV  # Standard deviation weigth
g = 4.0                 # Inhibitory weight balance

###############################################################################
# Neuron model parameters
###############################################################################
# Initial potential
v0 = -65*mV         # Initial membrane potential
std_v0 = 10*mV      # Standard deviation for initial values of v0
tau_m   = 10.0*ms   # Membrane time constant
tau_ref = 2.0*ms    # Absolute refractory period
tau_syn = 0.5*ms    # Post-synaptic current time constant

eqs = '''
	dv/dt = (k*(v-v_r)*(v-v_th) - u + I) / Cm : volt (unless refractory)
  du/dt = a*(b*(v-v_r)-u) : amp
	I : amp
  Cm : farad     
  v_r : volt   
  v_th : volt
  k : amp/volt**2
  v_p : volt
  a : hertz
  b : amp/volt
  c : volt
  d : amp

'''
reset = '''
	v = c
	u = u + d
'''
###############################################################################
# Creating neurons
###############################################################################
neurons = NeuronGroup(N, eqs, threshold='v>v_p', reset=reset, method='rk4', refractory=tau_ref)
#neurons.v = 'v0 + std_v0*randn()'
neurons.I = 0.0*pA

p = [] # Stores NeuronGroups, one for each population

for r in range(0, 8):
	if r == 0:
		p.append(neurons[:nn_cum[0]])
		p[r].Cm = 100.0*pF
        	p[r].v_r = -60.0*mV
        	p[r].v_th = -40.0*mV
        	p[r].v_p = 35*mV
        	p[r].k = 0.7*pA/(mV**2)
          	p[r].a = 0.03*kHz
          	p[r].b = -2.0*pA/mV
          	p[r].c = -50*mV 
          	p[r].d = 100*pA
		p[r].v = '(-60 + 10*randn())*mV'
	else:
		p.append(neurons[nn_cum[r-1]:nn_cum[r]])
          	if r%2 == 0:
         		p[r].Cm = 100.0*pF
                	p[r].v_r = -60.0*mV
                	p[r].v_th = -40.0*mV
                	p[r].v_p = 35*mV
                	p[r].k = 0.7*pA/(mV**2)
                        p[r].a = 0.03*kHz
                        p[r].b = -2.0*pA/mV
                        p[r].c = -50*mV
                        p[r].d = 100*pA
                        p[r].v = '(-60 + 10*randn())*mV'
		else:
            		p[r].Cm = 20.0*pF
                	p[r].v_r = -55.0*mV
                	p[r].v_th = -40.0*mV
                	p[r].v_p = 25*mV
                	p[r].k = 1.0*pA/(mV**2)
                	p[r].a = 0.2*kHz
                	p[r].b = 8.0*pA/mV
                	p[r].c = -55*mV 
                	p[r].d = 200*pA         
			p[r].v = '(-55 + 10*randn())*mV'
###############################################################################
# Creating synapse connections
###############################################################################
con = [] # Stores connections
for c in range(0, 8):
    for r in range(0, 8):
        # Excitatory layer
        if (c % 2) == 0:
            if c == 2 and r == 0:
                con.append( Synapses(p[c],p[r], """ w:volt """,on_pre = 'v_post += 2*w') )
                con[-1].connect(condition='i!=j', p=table[r][c])
                con[-1].w = 'clip((w_ex + std_w_ex*randn()),w_ex*0.1, w_ex*10.0)'
            else:
                con.append( Synapses(p[c],p[r], """ w:volt """,on_pre = 'v_post += w') )
                con[-1].connect(condition='i!=j', p=table[r][c])
                con[-1].w = 'clip((w_ex + std_w_ex*randn()),w_ex*0.1, w_ex*10.0)'
            con[-1].delay = 'clip(d_ex + std_d_ex*randn(),0,d_ex*10)'
        # Inhibitory layer
        else:
            con.append( Synapses(p[c],p[r], """ w:volt """,on_pre = 'v_post -= g*w') )
            con[-1].connect(condition='i!=j', p=table[r][c])
            con[-1].w = 'clip((w_ex + std_w_ex*randn()),w_ex*0.1, w_ex*10.0)'
            con[-1].delay = 'clip(d_in + std_d_in*randn(),0,d_in*10)'

###############################################################################
# Creating poissonian background inputs
###############################################################################
bg_in  = []
for r in range(0, 8):
	bg_in.append( PoissonInput(p[r], 'v', bg_layer[r], 8*Hz, weight = w_ex ) )

###############################################################################
# Creating spike monitors
###############################################################################
spikemon = []
for r in range(0, 8):
	spikemon.append( SpikeMonitor( p[r]) )

smon_net = SpikeMonitor(neurons)

###############################################################################
# Measuring frequencies by layer
###############################################################################
f_23e = spikemon[0].count;
f_23i = spikemon[1].count;
f_4e  = spikemon[2].count;
f_4i  = spikemon[3].count;
f_5e  = spikemon[4].count;
f_5i  = spikemon[5].count;
f_6e  = spikemon[6].count;
f_6i  = spikemon[7].count;

###############################################################################
# Running the simulation
###############################################################################
net = Network(collect())
net.add(neurons,p, con, spikemon, bg_in)    # Adding objects to the simulation
net.run(tsim, report='stdout')

###############################################################################
# Raster plot
###############################################################################
plot(smon_net.t/ms, smon_net.i,'.k', markersize=0.5)
xlabel('Time (ms)')
ylabel('Neuron index');
ylim(0,sum(n_layer))
xlim(500,1000)
plt.gca().invert_yaxis()
savefig('raster_PDnet.png',dpi=300)
close()

###############################################################################
# Calculating the average frequency by layer
###############################################################################
freqs = []
freqs.append(mean(f_23e)/tsim)
freqs.append(mean(f_23i)/tsim)
freqs.append(mean(f_4e)/tsim)
freqs.append(mean(f_4i)/tsim)
freqs.append(mean(f_5e)/tsim)
freqs.append(mean(f_5i)/tsim)
freqs.append(mean(f_6e)/tsim)
freqs.append(mean(f_6i)/tsim)

# Ploting frequencies by layer
ind = np.arange(8)
bar(ind, freqs)
savefig('freqs_PDnet.png', dpi=300)
