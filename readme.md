# Filtering strategies for RSSI signal prediction

Implementation of all filtering strategies described in [1] to filter a noisy RSSI signal. 

This is:
* Grey filter
* Fourier Transform filter
* Kalman filter
* Particles filter

Although [1] refers to a RSSI signal, this implementation can be runned with any time series.

## Getting Started

### Clone repository

    ~ $ git clone https://github.com/philipiv/rssi-filtering-kalman.git
    ~ $ cd rssi-filtering-kalman

### Project requirements 

It is strongly advised you work in a virtual environment.\
First step is to create one and install all necessary project requirements.
       
    ~/rssi-filtering-kalman $ virtualenv env --python=python3
    ~/rssi-filtering-kalman $ source env/bin/activate
    ~/rssi-filtering-kalman $ pip install -r requirements.txt

## Execution

    ~/rssi-filtering-kalman $ cd scripts
    ~/rssi-filtering-kalman/scripts $ python main.py [--file /path/to/file]

Optionaly, you can set the path to a file containing your data, default path is _../data/sample.csv_.

For example:

    ~/rssi-filtering-kalman/scripts $ python strategy.py --file ../data/sample.csv


## Results

After execution, the script output is a Figure containing original signal and output to all filters.

When executed with the sample data the output looks like this:

![image](https://github.com/philipiv/rssi-filtering-kalman-grey-fourier-particles-bellavista2006/blob/master/sample_output.png)



## References

[1] P. Bellavista, A. Corradi and C. Giannelli, "Evaluating Filtering Strategies for Decentralized Handover Prediction in the Wireless Internet," 11th IEEE Symposium on Computers and Communications (ISCC'06), Cagliari, Italy, 2006, pp. 167-174.
