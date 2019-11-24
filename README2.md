# iXGBNet: an integrative XGBoost-based method for inferring gene regulatory networks
iXGBNet is a method based on the XGBoost model and is able to integrate the information from the other data, such as the knockout data, to refine the inferred GRN. Moreover, iXGBNet utilizes a new way to tackle the time-series expression data by considering the accumulation impact of the gene expressions at previous time points. Using the benchmark datasets from DREAM4 and DREAM5 challenges, we show that iXGBNet achieves better performance than other state-of-the-art methods.
## Dependency
Xgboost Version=0.6 [Reference Link](https://xgboost.readthedocs.io/en/latest/build.html "悬停显示")
### Pip install
    Python version=3.7
    xgboost Version>=0.6
    Pandas>=0.19.x
    Numpy>=1.12.x
Now, we give a code example in two data sets, dream4 and dream5, respectively.    
## Dream4 example
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
### Dream4 parameters
    data_tm: Time-series experimental data.
    sample_num: Number of time-series experimental's samples.
    k: Previous k time points.
    alpha: Decay factor.
    iter_num: Number of iterations in XGBoost model.
    data_ko: Knockout experimental data.       
## Dream5 example
    from dream5.iXGBNet_dream5 import *
    import pandas as pd
    for i in [1,3]:
        # Read data
        file_ss = "data/steadystate_data/dream5_net{}_steadystate.csv".format(i)
        data_ss = pd.read_csv(file_ss).to_numpy()
        
        # Compute weights of gene regulatory network
        vv = main(data_ss, 1000)
        
        # Export result
        df = pd.DataFrame(vv)
        df.to_csv("result/dream5_d{}.csv".format(i), index=False)
### Dream5 parameters
    data_ss: Steady-state experimental data.
    iter_num: Number of iterations in XGBoost model.
