import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import csv

# https://note.nkmk.me/python-list-index/
def _index(li, x, default=None):
    return li.index(x) if x in li else default

def _csv_key2col( csv_filename, keys ):
    line = None
    with open(csv_filename, newline="") as f:
        reader = csv.reader(f)
        line = next(reader)

    if( line == None ):
        return None

    refs = [ ref.strip() for ref in line ]

    cols = []
    for key in keys:
        cols.append( _index( refs, key ) )

    return cols

def plot_csv(csv_filename, x_col, y_cols, labels, img_filename, title=None, xlabel=None, ylabel=None ):
    cols = _csv_key2col( csv_filename, [x_col]+y_cols )
    data = np.loadtxt( csv_filename, dtype="float", delimiter=",", skiprows=1, usecols=cols )

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    for i, label in enumerate(labels):
        ax.plot( data[:,0], data[:,i+1], label=label )

    plt.legend(loc='best')

    if( title is not None ):
        plt.title(title)

    if( xlabel is not None ):
        plt.xlabel(xlabel)

    if( ylabel is not None ):
        plt.ylabel(ylabel)

    plt.savefig(img_filename)
    plt.close()

def plot_csvs( csv_filenames, x_col, y_col, img_filename, title=None, ylim=None, xlabel=None, ylabel=None ):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for csv_filename in csv_filenames:
        try:
            cols = _csv_key2col( csv_filename, [x_col,y_col] )
            data = np.loadtxt( csv_filename, dtype="float", delimiter=",", skiprows=1, usecols=cols )
            if( data.shape[0] > 2 ):
                label = os.path.splitext(os.path.basename(csv_filename))[0]
                ax.plot( data[:,0], data[:,1], label=label )
        except:
            pass

    plt.legend(loc='best')

    if( title is not None ):
        plt.title(title)

    if( xlabel is not None ):
        plt.xlabel(xlabel)

    if( ylabel is not None ):
        plt.ylabel(ylabel)

    if( ylim is not None ):
        plt.ylim( ylim[0], ylim[1] )

    plt.savefig(img_filename)
    plt.close()

#https://www.sejuku.net/blog/63568
def _listdir_dir(path):
    dirlist = []
    for f in os.listdir(path):
      if os.path.isdir(os.path.join(path, f)):
        dirlist.append(f)
    dirlist.sort()
    return dirlist

if( __name__ == '__main__' ):
    import argparse
    import glob

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--log_dir', default='.' )
    parser.add_argument('--keys', nargs='*', default=None )
    parser.add_argument('--name', default='summary' )
    args = parser.parse_args()

    dirs = _listdir_dir( args.log_dir )
    csv_files = []

    for dir in dirs:
        s = os.path.join( dir, '*.csv' )
        s = os.path.join( args.log_dir, s )
        for csv_file in glob.glob( s ):
            csv_files.append(csv_file)
            name = os.path.splitext(os.path.basename(csv_file))[0]
            for key in args.keys:
                img_filename = os.path.join(dir, '{}_{}.png'.format(name,key) )
                img_filename = os.path.join(args.log_dir, img_filename )
                plot_csv( csv_file, 'epoch', [key], [key], img_filename, xlabel='epoch', ylabel=key )

    for key in args.keys:
        img_filename = os.path.join( args.log_dir, '{}_{}.png'.format(args.name,key) )
        plot_csvs( csv_files, 'epoch', key, img_filename, xlabel='epoch', ylabel=key )
