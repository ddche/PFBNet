# PFBNet: Prior fused boosting method for gene regulatory network inference
we present a novel method, namely prior fused boosting network inference method (PFBNet), to infer GRNs from time-series expression data by using the non-linear model of Boosting and the prior information (e.g., the knockout data) fusion scheme. The experiments on the benchmark datasets from DREAM challenge as well as the E.coli datasets show that PFBNet achieves significantly better performance than other state-of-the-art methods (HiDi, iRafNet and BiXGBoost).  
## Dependency
Xgboost Version=0.6 [Reference Link](https://xgboost.readthedocs.io/en/latest/build.html "悬停显示")
### Pip install
    Python version=3.7
    xgboost Version>=0.6
    Pandas>=0.19.x
    Numpy>=1.12.x
Now, we give a code example in two data sets, dream4 and Ecoil, respectively.    
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
