import numpy as np
from scipy.interpolate import interp1d

x = np.array([
   [30., 0.],
   [19., 0.],
   [18., 0.0001],
   [17., 0.0005],
   [16., 0.001],
   [15., 0.003],
   [14., 0.005],
   [13.75, 0.007],
   [13.5, 0.009 ],
   [13.25, 0.011 ],
   [13.0, 0.014 ],
   [12.75, 0.017 ],
   [12.5, 0.021 ],
   [12.25, 0.026 ],
   [12.0, 0.032 ],
   [11.75, 0.039],
   [11.5, 0.048 ],
   [11.25, 0.058 ],
   [11.0, 0.07 ],
   [10.75, 0.085 ],
   [10.5, 0.102 ],
   [10.25, 0.121 ],
   [10.0, 0.143 ],
   [9.75, 0.168 ],
   [9.5, 0.197 ],
   [9.25, 0.23 ],
   [9.0, 0.268 ],
   [8.75, 0.312 ],
   [8.5, 0.362 ],
   [8.25, 0.42 ],
   [8.0, 0.487 ],
   [7.75, 0.563],
   [7.5, 0.65 ],
   [7.25, 0.747 ],
   [7.0, 0.851 ],
   [6.75, 0.951 ],
   [6.5, 0.999 ],
   [6.25, 1.0 ],
   [6.0, 1.0 ],
   [5.75, 1.0 ],
   [5.5, 1.0 ],
   [0.,1.]])
   
x_interp = interp1d(x[:,0],x[:,1],kind='cubic',bounds_error=False,fill_value=0.)

Xhi = lambda z: max(min(float(x_interp(z)),1),0)