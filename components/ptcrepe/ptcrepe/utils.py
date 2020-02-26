import torch
import numpy as np
import os


def download_weights(model_capacitiy):
    try:
        from urllib.request import urlretrieve
    except ImportError:
        from urllib import urlretrieve

    weight_file = "crepe-{}.pth".format(model_capacitiy)
    base_url = "https://github.com/sweetcocoa/crepe-pytorch/raw/models/"

    # in all other cases, decompress the weights file if necessary
    package_dir = os.path.dirname(os.path.realpath(__file__))
    weight_path = os.path.join(package_dir, weight_file)
    if not os.path.isfile(weight_path):
        print("Downloading weight file {} from {} ...".format(weight_path, base_url + weight_file))
        urlretrieve(base_url + weight_file, weight_path)


def to_local_average_cents(salience, center=None):
    """
    find the weighted average cents near the argmax bin
    """

    if not hasattr(to_local_average_cents, "cents_mapping"):
        # the bin number-to-cents mapping
        to_local_average_cents.mapping = (
            torch.tensor(np.linspace(0, 7180, 360)) + 1997.3794084376191
        )

    if isinstance(salience, np.ndarray):
        salience = torch.from_numpy(salience)

    if salience.ndim == 1:
        if center is None:
            center = int(torch.argmax(salience))
        start = max(0, center - 4)
        end = min(len(salience), center + 5)
        salience = salience[start:end]
        product_sum = torch.sum(salience * to_local_average_cents.mapping[start:end])
        weight_sum = torch.sum(salience)
        return product_sum / weight_sum
    if salience.ndim == 2:
        return torch.tensor(
            [to_local_average_cents(salience[i, :]) for i in range(salience.shape[0])]
        )

    raise Exception("label should be either 1d or 2d Tensor")


def to_viterbi_cents(salience):
    """
    Find the Viterbi path using a transition prior that induces pitch
    continuity.

    * Note : This is NOT implemented with pytorch.
    """
    from hmmlearn import hmm

    # uniform prior on the starting pitch
    starting = np.ones(360) / 360

    # transition probabilities inducing continuous pitch
    xx, yy = np.meshgrid(range(360), range(360))
    transition = np.maximum(12 - abs(xx - yy), 0)
    transition = transition / np.sum(transition, axis=1)[:, None]

    # emission probability = fixed probability for self, evenly distribute the
    # others
    self_emission = 0.1
    emission = np.eye(360) * self_emission + np.ones(shape=(360, 360)) * ((1 - self_emission) / 360)

    # fix the model parameters because we are not optimizing the model
    model = hmm.MultinomialHMM(360, starting, transition)
    model.startprob_, model.transmat_, model.emissionprob_ = starting, transition, emission

    # find the Viterbi path
    observations = np.argmax(salience, axis=1)
    path = model.predict(observations.reshape(-1, 1), [len(observations)])

    return np.array(
        [to_local_average_cents(salience[i, :], path[i]) for i in range(len(observations))]
    )


def to_freq(activation, viterbi=False):
    if viterbi:
        cents = to_viterbi_cents(activation.detach().numpy())
        cents = torch.tensor(cents)
    else:
        cents = to_local_average_cents(activation)

    frequency = 10 * 2 ** (cents / 1200)
    frequency[torch.isnan(frequency)] = 0
    return frequency
