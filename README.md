# Waveform-based Classification of Dentate Spikes (WFbC)
This repository contains the scripts for detecting and classifying dentate spikes according to the following publication:

Santiago, Rodrigo MM, et al. "Waveform-based classification of dentate spikes." _bioRxiv_ (2023): 2023-10. [https://doi.org/10.1101/2023.10.24.563826](https://doi.org/10.1101/2023.10.24.563826)


The Python code is available at "dentatespike.py".

To use it, place it in the same directory as your Jupyter Notebook or simply map it as follows:
```sh
import sys
sys.path.append('/path') # replace 'path' with the directory where the file is located.
import dentatespike as ds
```

See Jupyter Notebooks for usage examples.

DS waveform data
---
The original data used in this study, as well as the access links, are described in the Materials and Methods section. The derived data identified as the waveforms of DSs detected in the channel located at the hilus of each animal are available at [https://data.mrc.ox.ac.uk/waveform](https://data.mrc.ox.ac.uk/waveform) (DOI: [10.5287/ora-wrbzrbwpk](http:/doi.org/10.5287/ora-wrbzrbwpk)).

For instructions on how to read the data, please refer to "Waveform data.ipynb" notebook.

Citing
---
To cite this code, please refer to DOI [10.5281/zenodo.10080866](http://doi.org/10.5281/zenodo.10080866).
