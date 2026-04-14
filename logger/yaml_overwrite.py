import os
import glob
import yaml
import shutil
from datetime import datetime, timedelta, timezone

def time_str_for_file():
    JST = timezone(timedelta(hours=+9), 'JST')
    return datetime.now(JST).strftime("%Y%m%d-%H%M%S")

def yaml_overwrite( key, value, root_dir='.', yaml_name='config.yaml', folder=True ):
    filenames = glob.glob( os.path.join( root_dir, f'**/{yaml_name}' ), recursive=True)
    filenames.sort()

    for filename in filenames:
        with open( filename, 'r') as yin:
            cfg = yaml.load(yin, Loader=yaml.SafeLoader)

        if( key in cfg.keys() ):
            dst = filename+'.'+time_str_for_file()
            shutil.copyfile( filename, dst )
            cfg[key] = value
            with open( filename, "w") as fout:
                    yaml.dump(cfg, fout, default_flow_style=False)
            print( dst )


if( __name__ == '__main__' ):
    import argparse

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--log_dir', default='.' )
    parser.add_argument('--yaml_name', default='config.yaml' )

    parser.add_argument('--key', default=None )
    parser.add_argument('--type', default='str' )
    parser.add_argument('--val', nargs='*', default=None )

    args = vars( parser.parse_args() )

    value = [ eval( '{}(v)'.format(args['type']) ) for v in args['val'] ]
    if( len(value) == 1 ):
        value = value[0]

    yaml_overwrite( key=args['key'],
                    value=value,
                    root_dir=args['log_dir'],
                    yaml_name=args['yaml_name'],
                    folder=True )
