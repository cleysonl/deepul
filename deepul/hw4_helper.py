from .utils import *

######################
##### Question 2 #####
######################



def visualize_q2_data():
    data_dir = get_data_dir(2)
    train_data, test_data = load_pickled_data(join(data_dir, 'shapes.pkl'))
    name = 'Shape'

    idxs = np.random.choice(len(train_data), replace=False, size=(100,))
    images = train_data[idxs] * 255
    show_samples(images, title=f'{name} Samples')

def q2_save_results(fn):
    data_dir = get_data_dir(2)
    train_data, test_data = load_pickled_data(join(data_dir, 'shapes.pkl'))

    train_losses, test_losses, samples = fn(train_data, test_data)
    samples = np.clip(samples.astype('float') * 2.0, 0, 1.9999)
    floored_samples = np.floor(samples)

    print(f'Final Test Loss: {test_losses[-1]:.4f}')
    save_training_plot(train_losses, test_losses, f'Q2 Dataset Train Plot',
                       f'results/q2_train_plot.png')
    show_samples(samples * 255.0 / 2.0, f'results/q2_samples.png')
    show_samples(floored_samples * 255.0, f'results/q2_flooredsamples.png', title='Samples with Flooring')

######################
##### Question 3 #####
######################

def visualize_q3_data():
    data_dir = get_data_dir(2)
    train_data, test_data = load_pickled_data(join(data_dir, 'celeb.pkl'))
    name = 'CelebA'

    idxs = np.random.choice(len(train_data), replace=False, size=(100,))
    images = train_data[idxs].astype(np.float32) / 3.0 * 255.0
    show_samples(images, title=f'{name} Samples')

def get_q3_data():
    data_dir = get_data_dir(2)
    train_data, test_data = load_pickled_data(join(data_dir, 'celeb.pkl'))
    return train_data, test_data


def q3_save_results(fn, part):
    data_dir = get_data_dir(2)
    train_data, test_data = load_pickled_data(join(data_dir, 'celeb.pkl'))

    train_losses, test_losses, samples, interpolations = fn(train_data, test_data)
    samples = samples.astype('float')
    interpolations = interpolations.astype('float')

    print(f'Final Test Loss: {test_losses[-1]:.4f}')
    save_training_plot(train_losses, test_losses, f'Q3 Dataset Train Plot',
                       f'results/q3_{part}_train_plot.png')
    show_samples(samples * 255.0, f'results/q3_{part}_samples.png')
    show_samples(interpolations * 255.0, f'results/q3_{part}_interpolations.png', nrow=6, title='Interpolations')


