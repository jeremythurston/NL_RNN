# NL_RNN

NL RNN is a recurring neural network (RNN) that learns the evolution of optical pulses through nonlinear media. NL RNN is trained with data sets generated using [PyNLO](https://github.com/pyNLO/PyNLO). This project is heavily inspired by [Salmela, et al., Nature Machine Intelligence, 2021](https://doi.org/10.1038/s42256-021-00297-z).


## Contact

Jeremy Thurston, Physics Ph.D. student at JILA (jeremy.thurston@colorado.edu)


## Version Requirements
NL RNN was written using the following versions:
* Python 3.9.1
* TensorFlow 2.9.1
* Keras 2.9.0


## Usage
Generation of a trainable dataset can be done with the command
```sh
python3 .\generate_data.py
```
To make a variable parameter, simply loop over the array, such as
```python
for fwhm in np.linspace(100, 200, 64):
    # Run NLSE simulation
    # Save data
```
This script will generate a full list of ```.csv``` files inside ```\NL_RNN\testing_data\```. Each ```.csv``` will be of the form
```
{
    'fwhm':      # pulse duration in ps (float),
    'epp':       # energy per pulse in J (float),
    'AW':        # input pulse spectral profile (pts x 1),
    'AT':        # input pulse temporal profile (pts x 1),
    'pulse_AW':  # evolution of spectrum (pts x steps),
    'pulse_AT':  # evolution of temporal profile (pts x steps)
}
```

To loop over hyperparameters to find the optimal model with the [KerasTuner](https://keras.io/keras_tuner/) module, run
```sh
python3 .\main.py
```
Once the optimal model is saved, 

tbd...

To visualize the model training using [TensorBoard](https://www.tensorflow.org/tensorboard), run
```sh
tensorboard --logdir=logs/
```
Then open http://localhost:6006/ on your favorite browser.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## License
NL RNN is licensed under the [MIT](https://choosealicense.com/licenses/mit/) license.
