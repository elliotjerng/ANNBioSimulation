### Synaptic sign switching simulation in a co-releasing neural circuit model
#### Shun Li | Elliot Jerng
Modeling synaptic plasticity in a circuit involving a rare population of glutamate/GABA co-relasing neurons:
- Endopeduncular Nucleus(EP) to Lateral Habenula(LHb) to Dopamine(DAN)

Sign-Switching Task Background:
- In biology, optogenetic stimulation at LHb paired with reward then punishment leads to synaptic sign switching at EP.
- In artificial neural networks, an "optogenetic" stimulus/input (ex. MNIST label 4) paired with target value of 1 (reward) then 0 (punishment) leads to weight sign switching at EP layer.

Goal:
- Compare sign-fixed networks (Dale's Law) to our co-releasing circuit to explore advantages of co-release mechanism in synaptic sign switching.

synaptic_plasticity_model.ipynb:

- model synaptic sign switching using sign-switching task
- compare Dale's Law network vs. co-release circuit

neuron_constraint.ipynb:

- sequentially reduce the number of updatable weights
- compare learning in Dale's Law network vs. co-release circuit as we reduce the number of active weights.

