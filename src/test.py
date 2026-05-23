import numpy as np

hf = np.load(r"d:\gitproject\Tact_gen\outputs\agent_run\heightfield_agent.npy")
hf = np.clip(hf, 0, 1)

print("min:", hf.min())
print("max:", hf.max())
print("range:", hf.max() - hf.min())
print("std:", hf.std())

labels = np.floor(hf * 10).astype(int)
labels = np.clip(labels, 0, 9)

print("unique terrace labels:", np.unique(labels))
print("num labels:", len(np.unique(labels)))