def bool_arr_type(var): return var
def str_arr_type(var): return var
def int_arr_type(var): return var


def parse_parser(parser):
    in_args = vars(parser.parse_args())
    args = {}
    for val in parser._option_string_actions.values():
        if val.type is bool:
            args[val.dest] = is_true(in_args[val.dest])
        elif val.type is str_arr_type:
            args[val.dest] = get_args_list(in_args, val.dest)
        elif val.type is bool_arr_type:
            args[val.dest] = get_args_list(in_args, val.dest, is_true)
        elif val.type is int_arr_type:
            args[val.dest] = get_args_list(in_args, val.dest, int)
        elif val.dest in in_args:
            args[val.dest] = in_args[val.dest]
    return args


def get_args_list(args, key, var_type=None):
    if ',' in args[key]:
        ret = args[key].split(',')
    elif len(args[key]) == 0:
        ret = []
    else:
        ret = [args[key]]
    if var_type:
        ret = list(map(var_type, ret))
    return ret


def is_true(val):
    if isinstance(val, str):
        if val.lower() == 'true':
            return True
        elif val.lower() == 'false':
            return False
        elif val.isnumeric():
            return bool(int(val))
        else:
            raise Exception('Wrong value for boolean variable')
    else:
        return bool(val)
