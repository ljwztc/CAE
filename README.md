# CAE
## CLASSIFICATION MODEL FOR TIME-SERIES
### INTRODUCE
    This is the second version of this project. The first version is lost. So, we take immediate action to updata this project to github, which is a wise behavior.
    We, taking the stock for an example, put forward an approach to classify the time-series. 
### STEP
#### FIRST
    we change the 5-channel data to the same-formula candlesticks chart. 
#### SECOND
    we build a  **convolutional autoencoder** and feed it with this candlesticks chart. With the mature network, we can take the candlesticks chart as input, extracting the feature of time-series in the FC-layer inside the network.
#### LAST
    we prepare to train a classifier to take the different data into two categories. This step is **under consideration now**.
                                                                                                                                                                            2019.3.21
