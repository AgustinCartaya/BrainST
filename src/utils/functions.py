import argparse


def dict_to_args(dict_to_convert, deep_conversion=False):
    """converts a dictionary to an argparse.Namespace object to acces the values as attributes with dot notation""" 
    if not isinstance(dict_to_convert, dict):
        raise ValueError("El argumento debe ser un diccionario.")

    args = argparse.Namespace()
    if not deep_conversion:
        for k, v in dict_to_convert.items():
            setattr(args, k, v)   
    else:
        for k, v in dict_to_convert.items():
            if isinstance(v, dict):
                setattr(args, k, dict_to_args(v, deep_conversion=True))
            else:
                setattr(args, k, v)  
    return args