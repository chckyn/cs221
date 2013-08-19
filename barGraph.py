
from matplotlib import pyplot as plt
import numpy as np
from collections import defaultdict

width = 0.15

errors = [['784-3000', (0.175, 0.1638)],
          ['784-3000-1500', (0.125, 0.1438)],
          ['784-3000-1500-750', (0.158, 0.1461)],
          ['784-3000-1500-750-100', (0.147, 0.1294)],
          ['784-3000-1500-750-100-30', (0.181, 0.1846)],
          ['784-3000-1500-750-100-30-10', (0.307, 0.2901)],
#          ['784-196', (0.167, 0.1549)],
#          ['784-196-64', (0.169, 0.1529)],
#          ['784-196-64-25', (0.228, 0.2228)],
#          ['784-196-64-25-6', (0.432, 0.4163)],
         ] 

fig, ax = plt.subplots()
ind = np.arange(0, 3*len(errors)*width-2*width, 3*width)

axes = []
for i, e in enumerate(errors):
    axes += ax.bar(ind[i], e[1][0], width, color='r')
    axes += ax.bar(ind[i]+width, e[1][1], width, color='y')

ax.set_ylabel('Error\n', fontsize=20)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(14) 
ax.set_xticks(ind+width)
ax.set_xticklabels( ['\n'+chr(i) for i in range(ord('A'),ord('A')+len(errors))] , fontsize=14)

textstr = ""
l = ord('A')
for i in range(len(errors)):
    textstr += '\n '+chr(l)+' : '+errors[i][0]+'   '
    l+=1

props = dict(boxstyle='round', facecolor='white', alpha=1)
ax.text(1.09, 0.8, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

ax.legend([axes[0], axes[1]], ['training error', 'test error'], bbox_to_anchor=(1.5,1))

fig = plt.gcf()
fig.set_facecolor('white')
plt.tick_params(\
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
)
plt.tick_params(\
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left='off',        # ticks along the bottom edge are off
    right='off',         # ticks along the top edge are off
)
plt.subplots_adjust(right=0.6)
plt.show()
