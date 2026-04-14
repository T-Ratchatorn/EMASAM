import os
import glob
import yaml

def yaml_summary( csv_file, keys=[], root_dir='.', yaml_name='config.yaml', folder=True ):
    csv_file = os.path.join( root_dir, csv_file )
    filenames = glob.glob( os.path.join( root_dir, f'**/{yaml_name}' ), recursive=True)
    filenames.sort()
    with open( csv_file, 'w' ) as fout:
        line = 'folder, ' if folder else ''

        for key in keys:
            line += key + ', '
        line = line[:-2]
        fout.write( line+'\n' )

        for filename in filenames:
            line = os.path.dirname(filename)+', ' if folder else ''
            with open( filename, 'r') as yin:
                kwargs = yaml.load(yin, Loader=yaml.SafeLoader)
            for key in keys:
                if( key in kwargs.keys() ):
                    if( kwargs[key] is None ):
                        line += '-, '
                    elif( isinstance( kwargs[key], bool ) ):
                        if( kwargs[key] ):
                            line += key+', '
                        else:
                            line += '-, '
                    elif( isinstance( kwargs[key], list ) ):
                        line += str(kwargs[key]).replace(',','|')+', '
                    else:
                        line += str(kwargs[key])+', '

                else:
                    line += ', '
            line = line[:-2]
            fout.write( line+'\n' )

if( __name__ == '__main__' ):
    import argparse

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--csv_name', default='summary.csv' )
    parser.add_argument('--log_dir', default='.' )
    parser.add_argument('--yaml_name', default='config.yaml' )

    parser.add_argument('--keys', nargs='*', default=None )

    args = vars( parser.parse_args() )

    yaml_summary( csv_file=args['csv_name'],
                  keys=args['keys'],
                  root_dir=args['log_dir'],
                  yaml_name=args['yaml_name'],
                  folder=True )
