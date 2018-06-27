from time import strftime, gmtime


class TerminalColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_print_data():
    return strftime("%Y-%m-%d %H:%M:%S", gmtime())


def print_normal(s):
    print(TerminalColors.OKBLUE + "[" + get_print_data() + "] INFO : " + str(s) + TerminalColors.ENDC)


def print_warning(s):
    print(TerminalColors.WARNING + "[" + get_print_data() + "] WARN : " + str(s) + TerminalColors.ENDC)


def print_error(s):
    print(TerminalColors.FAIL + "[" + get_print_data() + "] ERROR : " + str(s) + TerminalColors.ENDC)
