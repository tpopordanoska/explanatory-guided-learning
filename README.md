# Explanatory guided learning

This is the code that accompanies the paper “Machine Guides, Human Supervises: Interactive Learning with Global Explanations”


### Experiments
The `requirements.txt` contains the Python dependencies and they can be installed using:

```
pip install -r requirements.txt
```

To run the experiments, use the `main.py` script. Type `python main.py --help` for the list of options.
For instance, to run 10 folds of 100 iterations for the synthetic experiment with XGL(rules) and the competitors use:
```
python main.py --experiments synthetic --strategies al_density_weighted al_least_confident random sq_random rules
```

The code will save all results in the `results` directory in pickle format.

### Plots
To draw the plots, use the `draw.py` script. Type `python draw.py --help` for the list of options.
For example, to draw the plots for several pickle files run:
```
python draw.py --folder <name_of_folder_containing_pickles> 
```
