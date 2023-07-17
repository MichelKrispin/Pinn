# Physics-Informed Neural Networks for reconstructing flow velocity fields

The bachelor thesis aims to guard a neural networks learning process by providing it with physical knowledge about the data, which it should take into consideration.
It tries to connect knowledge in the fields of physics with the advances in machine learning algorithms.
When trying to learn from data which captured a phenomenon that follows a physical law, e.g. fluid data, we can infer that the solution to this data will also include some variant of a physical law.
This does not mean, that the parameters or constants in these equations are known beforehand.
Only that we made an educated guess about the structure of the data.

The application, that this thesis looks at, is the prediction of the speed of some fluid at different locations and points in time described by the linear advection-diffusion equation (`C_t & = D \cdot C_{xx} - v \cdot C_x`).
In more detail, if there are images, i.e. measurements, of a fluid at different time steps inside a specific positional range, the goal is to feed these into a neural network that is then able to approximate the speed of this fluid for a specific time at all locations.
This will be of practical use, if an injection fluid in a human body is observed, that will move through organic tissue.
As the speed of the fluid moving through it can give further information about the health of the tissue, this application might help in the diagnosis of a disease.
