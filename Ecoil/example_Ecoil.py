from PFBNet_Ecoil import *

# cold_data example
cold_data = np.transpose(np.matrix(pd.read_csv('data/ecoil_cold_data.txt', sep='\t')))
cold = np.zeros((3,1484,8))
for i in range(3):
    cold[i,:,:] = cold_data[:,i*8:(i+1)*8]

vv = ecoil_main(cold, 8, 2, 0.45, 163, 1000)
df = pd.DataFrame(vv)
df.to_csv('result/cold.csv',index=False)

# heat_data example
heat_data = np.transpose(np.matrix(pd.read_csv('data/ecoil_heat_data.txt', sep='\t')))
heat = np.zeros((3,1484,8))
for i in range(3):
    heat[i,:,:] = heat_data[:,i*8:(i+1)*8]

vv = ecoil_main(heat, 8, 2, 0.45, 163, 1000)
df = pd.DataFrame(vv)
df.to_csv('result/heat.csv',index=False)

# oxi_data example
oxi_data = np.transpose(np.matrix(pd.read_csv('data/ecoil_oxi_data.txt', sep='\t')))
oxi = np.zeros((3,1484,11))
for i in range(3):
    oxi[i,:,:] = oxi_data[:,i*11:(i+1)*11]

vv = ecoil_main(oxi, 11, 2, 0.45, 163, 1000)
df = pd.DataFrame(vv)
df.to_csv('result/oxi.csv',index=False)


