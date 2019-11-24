from dream4.iXGBNet_dream4 import *
import pandas as pd

for i in range(1, 6):
    # Read data
    file_tm = "data/timeseries_data/insilico_size100_{}_timeseries.csv".format(i)
    tm = pd.read_csv(file_tm).to_numpy()
    file_ko = "data/knockout_data/insilico_size100_{}_knockout.csv".format(i)
    ko = pd.read_csv(file_ko).to_numpy()

    # Compute weights of gene regulatory network
    vv = main(tm, 10, 2, 0.45, 1000, ko)

    # Export result
    df = pd.DataFrame(vv)
    df.to_csv("result/dream4_d{}.csv".format(i), index=False)