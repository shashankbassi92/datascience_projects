import pandas as pd
from time import time


def make_submission(name, data_frame):
    """
    Parameters:
    -----------
        name:               string, your name
        data_frame:         pandas DataFrame [5999, 2], customer ids and 
                                predicted probabilities on the test set
    """
    cnames = data_frame.columns.values
    if not isinstance(data_frame, pd.DataFrame):
        raise ValueError('Expecting a pandas DataFrame for argument 2')
    elif cnames[0] != 'customer_id':
        raise ValueError('Column name 0 incorrect, expecting customer_id, received ' + cnames[0])

    t0 = time()
    filename = name + '_predictions.csv'
    data_frame.to_csv(filename, index=False)
    return '{fname} written in {t:.{rd}f} ms'.format(fname=filename, t=(time() - t0) / 1000., rd=3)
