import pandas as pd
import sys

def load_large_dta(fname):
    reader = pd.read_stata(fname, iterator=True)
    df = pd.DataFrame()

    try:
        chunk = reader.get_chunk(100*1000)
        while len(chunk) > 0:
            df = df.append(chunk, ignore_index=True)
            chunk = reader.get_chunk(100*1000)
            print '.',
            sys.stdout.flush()
    except (StopIteration, KeyboardInterrupt):
        pass

    print '\nloaded {} rows'.format(len(df))
    print(df)
    return df

if __name__ == '__main__':
	load_large_dta(sys.argv[1])