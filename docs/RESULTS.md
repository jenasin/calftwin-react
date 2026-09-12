# Computed results

Generated from `outputs/experiments`. All values are means over 50 seeds.


## Welfare burden (model units, lower is better)

| scenario     |   shadow_only |   threshold |   bounded_react_noforecast |   bounded_react_noverify |   bounded_react |   full_info |
|:-------------|--------------:|------------:|---------------------------:|-------------------------:|----------------:|------------:|
| normal       |         0.02  |       0.012 |                      0.017 |                    0.02  |           0.02  |       0.001 |
| heat         |         8.258 |       0.274 |                      0.366 |                    0.064 |           0.071 |       0.107 |
| water_block  |         7.371 |       3.183 |                      1.512 |                    1.93  |           1.601 |       0.328 |
| sensor_fault |         6.394 |       0.885 |                      0.506 |                    0.021 |           0.022 |       0.185 |
| fever        |        11.444 |      11.025 |                      3.487 |                    3.812 |           3.464 |       3.737 |
| fever_hot    |        17.685 |      10.159 |                      3.992 |                    3.098 |           2.977 |       4.163 |
| combined     |        17.679 |       3.22  |                      1.561 |                    1.29  |           0.925 |       0.669 |


## Total score = burden + intervention + information cost

| scenario     |   shadow_only |   threshold |   bounded_react_noforecast |   bounded_react_noverify |   bounded_react |   full_info |
|:-------------|--------------:|------------:|---------------------------:|-------------------------:|----------------:|------------:|
| normal       |         0.02  |       0.033 |                      0.136 |                    0.02  |           0.059 |       0.522 |
| heat         |         8.258 |       0.839 |                      1.406 |                    0.714 |           0.748 |       0.823 |
| water_block  |         7.371 |       3.396 |                      2.696 |                    2.309 |           2.008 |       0.957 |
| sensor_fault |         6.394 |       1.284 |                      1.614 |                    0.617 |           0.669 |       0.82  |
| fever        |        11.444 |      11.105 |                      5.41  |                    4.571 |           4.371 |       4.997 |
| fever_hot    |        17.685 |      10.839 |                      5.674 |                    4.35  |           4.328 |       5.553 |
| combined     |        17.679 |       3.809 |                      3.114 |                    2.514 |           2.304 |       1.58  |


## Confirmatory measurements purchased per day

| scenario     |   shadow_only |   threshold |   bounded_react_noforecast |   bounded_react_noverify |   bounded_react |   full_info |
|:-------------|--------------:|------------:|---------------------------:|-------------------------:|----------------:|------------:|
| normal       |             0 |           0 |                       0.52 |                        0 |            0.32 |           0 |
| heat         |             0 |           0 |                       1.08 |                        0 |            0.26 |           0 |
| water_block  |             0 |           0 |                       2.34 |                        0 |            0.52 |           0 |
| sensor_fault |             0 |           0 |                       1.96 |                        0 |            0.52 |           0 |
| fever        |             0 |           0 |                       4.66 |                        0 |            1.22 |           0 |
| fever_hot    |             0 |           0 |                       1.78 |                        0 |            0.84 |           0 |
| combined     |             0 |           0 |                       2.78 |                        0 |            2.28 |           0 |


## Cooling run-time (h per day)

| scenario     |   shadow_only |   threshold |   bounded_react_noforecast |   bounded_react_noverify |   bounded_react |   full_info |
|:-------------|--------------:|------------:|---------------------------:|-------------------------:|----------------:|------------:|
| normal       |             0 |       0.085 |                      0.94  |                     0    |            0    |       8.685 |
| heat         |             0 |       9.285 |                     15.17  |                    10.84 |           10.76 |      11.94  |
| water_block  |             0 |       2.785 |                     13.205 |                     4.48 |            4.2  |       8.82  |
| sensor_fault |             0 |       6.62  |                     14.55  |                     9.38 |            9.56 |      10.595 |
| fever        |             0 |       1.165 |                     13.57  |                     3.48 |            3.52 |      11.82  |
| fever_hot    |             0 |      11.065 |                     15.13  |                    11.7  |           11.68 |      14.01  |
| combined     |             0 |       9.12  |                     15.57  |                    15.1  |           15.02 |      13.475 |


## Detection latency (h)

| scenario     |   threshold |   bounded_react_noforecast |   bounded_react_noverify |   bounded_react |   full_info |
|:-------------|------------:|---------------------------:|-------------------------:|----------------:|------------:|
| normal       |       2.25  |                      1.25  |                  nan     |         nan     |       0     |
| heat         |       0.33  |                      0.842 |                    0.333 |           0.333 |       0     |
| water_block  |       0.695 |                      0.735 |                    0.698 |           0.703 |       0     |
| sensor_fault |       0.96  |                      1.031 |                    0.227 |           0.2   |       0     |
| fever        |       3.089 |                      0.56  |                    0.6   |           0.48  |       0.345 |
| fever_hot    |       0.505 |                      0.715 |                    0.62  |           0.54  |       0.17  |
| combined     |       0.505 |                      0.684 |                    0     |           0     |       0     |


## State-estimation quality (open loop)

| scenario     |   rmse_core_c |   bias_core_c |   coverage95_core |   mean_core_sd |   rmse_deficit |
|:-------------|--------------:|--------------:|------------------:|---------------:|---------------:|
| normal       |        0.1397 |        0.0261 |            0.9369 |         0.1357 |         0.0015 |
| heat         |        0.1079 |        0.0097 |            0.9779 |         0.1295 |         0.0022 |
| water_block  |        0.0902 |        0.0318 |            0.9973 |         0.1306 |         0.0076 |
| sensor_fault |        0.1189 |        0.007  |            0.9673 |         0.1318 |         0.0021 |
| fever        |        0.1775 |       -0.0063 |            0.915  |         0.1458 |         0.0019 |
| fever_hot    |        0.1397 |       -0.029  |            0.936  |         0.1351 |         0.0025 |
| combined     |        0.1694 |        0.0327 |            0.905  |         0.1371 |         0.0085 |
