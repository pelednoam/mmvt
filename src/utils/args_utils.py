import traceback

def bool_arr_type(var): return var
def str_arr_type(var): return var
def int_arr_type(var): return var
def float_arr_type(var): return var


def add_arguments(parser, arguments):
    for argument in arguments:
        parser.add_argument(argument)


def parse_parser(parser, argv=None):
    if not argv is None and not isinstance(argv, list):
        argv = create_arr_args(argv)
    if argv is None:
        in_args = vars(parser.parse_args())
    else:
        in_args = vars(parser.parse_args(argv))
    args = {}
    for val in parser._option_string_actions.values():
        # if val.type is None and val.dest in in_args:
        #     val.type = str
        #     print(val.dest, in_args[val.dest], val.type)
        # if val.type is str:
        #     args[val.dest] = in_args[val.dest].replace('_', ' ')
        if val.type is bool or val.type is is_true:
            args[val.dest] = is_true(in_args[val.dest])
        elif val.type is str_arr_type:
            args[val.dest] = get_args_list(in_args, val.dest, str, val.default)
        elif val.type is bool_arr_type:
            args[val.dest] = get_args_list(in_args, val.dest, is_true, val.default)
        elif val.type is int_arr_type:
            args[val.dest] = get_args_list(in_args, val.dest, int, val.default)
        elif val.type is float_arr_type:
            args[val.dest] = get_args_list(in_args, val.dest, float, val.default)
        elif val.dest in in_args:
            if type(in_args[val.dest]) is str:
                args[val.dest] = in_args[val.dest].replace("'", '')
            else:
                args[val.dest] = in_args[val.dest]
    return args


def get_args_list(args, key, var_type, default_val=''):
    if args[key] is None or len(args[key]) == 0:
        return default_val
    # Remove '"' if any and replace '_' with ' '
    args[key] = args[key].replace("'", '')
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
        if val.lower() in ['true', 'yes', 'y']:
            return True
        elif val.lower() in ['false', 'no', 'n']:
            return False
        elif is_int(val):
            return bool(int(val))
        else:
            raise Exception('Wrong value for boolean variable ("{}")'.format(val))
    else:
        return bool(val)


def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def is_true_or_none(val):
    try:
        return is_true(val)
    except:
        return None


def float_or_none(val):
    try:
        return float(val)
    except:
        return None


def str_arr_to_markers(args, field_name):
    if args[field_name] and len(args[field_name]) > 0:
        if len(args[field_name]) % 2 != 0:
            raise Exception("{} is list of tuples, like: '-1,marker1,0,marker2'".format(field_name))
        ret = []
        for ind in range(0, len(args[field_name]), 2):
            time, marker = args[field_name][ind], args[field_name][ind + 1]
            ret.append((float(time), marker.replace('_', ' ')))
        return ret


def create_arr_args(args):
    call_arr = []
    for arg, value in args.items():
        if isinstance(value, list):
            value = ','.join(map(str, value))
        call_arr.append('--{}'.format(arg))
        call_arr.append(str(value))
    return call_arr
