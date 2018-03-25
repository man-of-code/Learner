
import glob
import os
import csv
import matplotlib.pyplot as plt
import numpy as np


def movingaverage(y, window_size):

    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(y, window, 'same')


def readable_output(filename):
    readable = ''
   
    f_parts = filename.split('-')

    if f_parts[0] == 'learn_data':
        readable += 'distance: '
    else:
        readable += 'loss: '

    readable += f_parts[1] + ', ' + f_parts[2] + ' | '
    readable += f_parts[3] + ' | '
    readable += f_parts[4].split('.')[0]

    return readable


def plot_file(filename, type='loss'):
    with open(f, 'r') as csvfile:
        reader = csv.reader(csvfile)
        
        y = []
        for row in reader:
            if type == 'loss':
                y.append(float(row[0]))
            else:
                y.append(float(row[1]))

       
        if len(y) == 0:
            return

        print(readable_output(f))

      
        if type == 'loss':
            window = 100
        else:
            window = 10
        y_av = movingaverage(y, window)

        
        arr = np.array(y_av)
        if type == 'loss':
            print("%f\t%f\n" % (arr.min(), arr.mean()))
        else:
            print("%f\t%f\n" % (arr.max(), arr.mean()))

        
        plt.clf()  
        plt.title(f)
        
        if type == 'loss':
            plt.plot(y_av[:-50])
            plt.ylabel('Loss')
            plt.ylim(0, 5000)
            plt.xlim(0, 250000)
        else:
            plt.plot(y_av[:-5])
            plt.ylabel('Score')
            plt.ylim(0, 600)

        plt.savefig(f + '.png', bbox_inches='tight')


if __name__ == "__main__":
    
    os.chdir("Data")

    for f in glob.glob("learn*.csv"):
        plot_file(f, 'learn')

    for f in glob.glob("loss*.csv"):
	    plot_file(f, 'loss')