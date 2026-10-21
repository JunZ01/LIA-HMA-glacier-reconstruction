import sys
from .experiment import register,prepare,run
{'register':register,'prepare':prepare,'run':run}[sys.argv[1]]()
