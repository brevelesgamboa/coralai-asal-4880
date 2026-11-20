import pickle
import numpy as np

pop = pickle.load(open("illumination_results/test4/pop.pkl", "rb"))
data = pickle.load(open("illumination_results/test4/data.pkl", "rb"))
#print(pop)
#print(data)
#print(pop['params'].shape, data['loss'][-5:])

import matplotlib.pyplot as plt

fig, axs = plt.subplots(8, 8, figsize=(16, 16))
for ax, img in zip(axs.flat, pop['rgb'][:64]):
    ax.imshow(img); ax.axis('off')

#plt.show()

'''i = 0  # which individual
img = pop['rgb'][i]
plt.imshow(img)
plt.axis('off')
plt.show()

plt.show()'''

