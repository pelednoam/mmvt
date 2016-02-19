try:
    import cPickle as pickle
except:
    import pickle


def save_color_map(color_map='jet'):
    import matplotlib.pyplot as plt
    cm = plt.get_cmap(color_map)
    with open('color_map_{}.pkl'.format(color_map), 'wb') as fp:
        # protocol=2 so we'll be able to load in python 2.7
        pickle.dump(cm, fp)

if __name__ == '__main__':
    save_color_map()