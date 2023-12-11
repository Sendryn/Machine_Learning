import numpy as np
import matplotlib.pyplot as plt


def normal(x: np.ndarray, sigma: float, mu: float) -> np.ndarray:
    # Part 1.1
    return (np.exp(-((np.power((x-mu),2))/(2*np.power(sigma,2))))/(np.power(2*np.pi*np.power(sigma,2),0.5)))

def plot_normal(sigma: float, mu:float, x_start: float, x_end: float):
    # Part 1.2
    xs = np.linspace(x_start, x_end, num=500)
    plt.plot(xs, normal(xs, sigma, mu))
    
def _plot_three_normals():
    # Part 1.2
    plt.figure()
    plot_normal(0.5, 0, -5, 5)
    plot_normal(0.25, 1, -5, 5)
    plot_normal(1, 1.5, -5, 5)
    plt.savefig("1_2_1.png")
    plt.show()
_plot_three_normals()

def normal_mixture(x: np.ndarray, sigmas: list, mus: list, weights: list):
    # Part 2.1
    sum = 0
    for i,mu in enumerate(mus):
        sum = sum + (weights[i]/np.power(2*np.pi*np.power(sigmas[i],2),0.5)) * np.exp(-np.power(x-mu,2)/(2*np.power(sigmas[i],2)))
    return sum

def _compare_components_and_mixture(sigmas: list, mus: list, weights: list, x_start: float, x_end: float):
    # Part 2.2
    x_range = np.linspace(x_start, x_end, 500)
    for i, sigma in enumerate(sigmas):
        plt.plot(x_range, normal(x_range, sigma, mus[i]))
    plt.plot(np.linspace(x_start, x_end, 500), normal_mixture(np.linspace(x_start, x_end, 500), sigmas, mus, weights))
    plt.savefig("2_2_1.png")
    plt.show()

def sample_gaussian_mixture(sigmas: list, mus: list, weights: list, n_samples: int = 500):
    # Part 3.1
    samples_per_component = np.random.multinomial(n_samples, weights)
    sampled_values = []
    for component_idx, num_samples in enumerate(samples_per_component):
        samples_from_component = np.random.normal(loc=mus[component_idx], scale=sigmas[component_idx], size=num_samples)
        sampled_values.extend(samples_from_component)
    
    sampled_values = np.array(sampled_values)
    
    return sampled_values

def _plot_mixture_and_samples(sigmas: list, mus: list, weights: list, n_samples: int = 500):
    # Part 3.2
    plt.figure(figsize=(30, 6))
    samples = sample_gaussian_mixture(sigmas, mus, weights, n_samples)
    plt.subplot(141)
    plt.hist(samples, 10, density=True)
    plt.plot(np.linspace(-5, 5, n_samples), normal_mixture(np.linspace(-5, 5, n_samples), sigmas, mus, weights))
    plt.show()
    plt.subplot(142)
    plt.hist(samples, 100, density=True)
    plt.plot(np.linspace(-5, 5, n_samples), normal_mixture(np.linspace(-5, 5, n_samples), sigmas, mus, weights))
    plt.savefig("3_2_1.png")
    plt.show()
    plt.subplot(143)
    plt.hist(samples, 500, density=True)
    plt.plot(np.linspace(-5, 5, n_samples), normal_mixture(np.linspace(-5, 5, n_samples), sigmas, mus, weights))
    plt.show()
    plt.subplot(144)
    plt.hist(samples, 1000, density=True)
    plt.plot(np.linspace(-5, 5, n_samples), normal_mixture(np.linspace(-5, 5, n_samples), sigmas, mus, weights))
    plt.show()

if __name__ == '__main__':
    # select your function to test here and do `python3 template.py`
    normal(0, 1, 0) #= 0.3989422804014327
    normal(3, 1, 5) #= 0.05399096651318806
    normal(np.array([-1,0,1]), 1, 0) #= array([0.24197072, 0.39894228, 0.24197072])
    plot_normal(0.5, 0, -2, 2)
    normal_mixture(np.linspace(-5, 5, 5), [0.5, 0.25, 1], [0, 1, 1.5], [1/3, 1/3, 1/3])
    normal_mixture(np.linspace(-2, 2, 4), [0.5], [0], [1])
    _compare_components_and_mixture([0,-0.5,1.5], [0.5,1.5,0.25],0,3)
    sample_gaussian_mixture([0.1, 1], [-1, 1], [0.9, 0.1], 3) #= [-0.95352122 -0.88278249 -1.11422348]
    sample_gaussian_mixture([0.1, 1, 1.5], [1, -1, 5], [0.1, 0.1, 0.8], 10)
    _plot_mixture_and_samples([0.3,0.5,1],[0,-1,1.5],[0.2,0.3,0.5])