import os
import time
import copy
from datetime import datetime, timedelta, timezone

import glob
import yaml
import torch

from .yaml_summary import *
from .plot_csv import *


#https://www.sejuku.net/blog/63568
def _listdir_dir(path):
    dirlist = []
    for f in os.listdir(path):
      if os.path.isdir(os.path.join(path, f)):
        dirlist.append(f)

    dirlist.sort()
    return dirlist

def time_str():
    JST = timezone(timedelta(hours=+9), 'JST')
    return datetime.now(JST).strftime("%Y/%m/%d %H:%M:%S")

def time_str_for_file():
    JST = timezone(timedelta(hours=+9), 'JST')
    return datetime.now(JST).strftime("%Y%m%d-%H%M%S")

class SimpleLogger:
    def __init__( self, root_dir='.', nolog=False,
                   yaml_name='config.yaml', csv_name=None, state_name='state.pth',
                   timestamp=True, disp=True, log_name=None ):
        self.__nolog = nolog
        self.disp = disp
        self.root_dir = root_dir
        self.log_name = log_name
        self.yaml_name = yaml_name
        self.state_name = state_name
        self.timestamp = timestamp
        self.keys = None

        if( csv_name is None ):
            self.csv_name = self.log_name.rstrip('/')+'.csv'
        else:
            self.csv_name = csv_name

        self.__dir_name = self.mkdir(log_name)

    def mkdir( self, dir ):
        if( self.is_logged ):
            if( dir is None ):
                while( True ):
                    dir_name = os.path.join( self.root_dir, time_str_for_file() )
                    if( not os.path.exists( dir_name ) ):
                        break
                    time.sleep(0.5)
            else:
                dir_name = os.path.join( self.root_dir, dir )
            os.makedirs( dir_name, exist_ok=True )
        else:
            dir_name = None

        return dir_name

    def clear_log( self ):
        for file in glob.glob( self.dir_name+'/*' ):
            os.remove(file)

    @property
    def dir_name( self ):
        return self.__dir_name

    def get_csv_filename( self ):
        return os.path.join(self.dir_name, self.csv_name)

    def get_yaml_filename( self ):
        return os.path.join(self.dir_name, self.yaml_name)

    @property
    def is_logged( self ):
        return not self.__nolog

    def write_state_dict( self, state_dict ):
        if( self.is_logged ):
            torch.save( state_dict, os.path.join(self.dir_name, self.state_name) )

    def read_state_dict( self ):
        state = None
        if( self.is_logged ):
            state_filename = os.path.join( self.dir_name, self.state_name )
            if( os.path.exists( state_filename ) ):
                state = torch.load( state_filename )
        return state

    def exists_yaml( self ):
        yaml_filename = os.path.join( self.dir_name, self.yaml_name )
        return self.is_logged and os.path.exists( yaml_filename )

    def write_yaml( self, dict_yaml ):
        if( self.is_logged ):
            with open( os.path.join(self.dir_name, self.yaml_name), "w") as fout:
                    yaml.dump(dict_yaml, fout, default_flow_style=False)

    def read_yaml( self ):
        cfg = None
        if( self.is_logged ):
            yaml_filename = os.path.join( self.dir_name, self.yaml_name )
            if( os.path.exists( yaml_filename ) ):
                with open( yaml_filename, 'r') as yin:
                    cfg = yaml.load(yin, Loader=yaml.SafeLoader)
        return cfg

    def init_csv( self, keys ):
        self.keys = copy.copy(keys)

        line = ''
        if( self.timestamp ):
            line += 'timestamp, '
        for key in self.keys:
            line += key + ', '
        line = line[:-2]

        if( self.is_logged ):
            filename = os.path.join(self.dir_name, self.csv_name)
            if( not os.path.exists( filename ) ):
                with open( filename, "w") as fout:
                    fout.write( line+'\n' )

        if( self.disp ):
            print( line )

    def append_csv( self, dict_data ):
        line = ''
        if( self.timestamp ):
            line += time_str() + ', '

        for key in self.keys:
            if( key in dict_data.keys() ):
                line += str(dict_data[key])
            line += ', '
        line = line[:-2]

        if( self.is_logged ):
            with open( os.path.join(self.dir_name, self.csv_name), "a") as fout:
                fout.write( line+'\n' )
        if( self.disp ):
            print( line )

    def yaml_summary(self, summary_csv='summary.csv'):
        if( self.is_logged ):
            yaml_summary( csv_file=summary_csv, keys=self.keys[1:], root_dir=self.root_dir, yaml_name=self.yaml_name, folder=True )

    def plot_csv(self):
        if( self.is_logged ):
            for key in self.keys[1:]:
                img_filename = os.path.join(self.dir_name, '{}_{}.png'.format(self.log_name,key) )
                csv_filename = os.path.join(self.dir_name, self.csv_name)
                plot_csv( csv_filename, 'epoch', [key], [key], img_filename, xlabel='epoch', ylabel=key )

    def plot_csvs(self, name='summary'):
        if( self.is_logged ):
            dirs = _listdir_dir( self.root_dir )
            csv_filenames = []
            for dir in dirs:
                s = os.path.join( dir, '*.csv' )
                s = os.path.join( self.root_dir, s )
                csv_filenames += glob.glob( s )

            for key in self.keys[1:]:
                img_filename = os.path.join( self.root_dir, '{}_{}.png'.format(name,key) )
                plot_csvs( csv_filenames, 'epoch', key, img_filename, xlabel='epoch', ylabel=key )

if( __name__ == '__main__' ):
    sl = SimpleLogger()
    keys = ['epoch' , 'acc']
    sl.write_yaml( keys )
    sl.init_csv( keys )
    log = {}
    log['epoch'] = 0
    log['acc'] = 0.9

    sl.append_csv( log )

