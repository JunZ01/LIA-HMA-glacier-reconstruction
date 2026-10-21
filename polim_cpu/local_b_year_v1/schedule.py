"""Pure time and conditional scheduling, also used by no-model checks."""
def times(case):
    if case in ('D_old','D_new'):return [1482.5]
    if case=='N':return [1553.,1553.25,1553.5,1553.75]
    raise ValueError(case)
def can_enter_annual(local,old,new,pair):
    return bool(local and old=='completed' and new=='completed' and pair)
def can_enter_C(status):return status=='completed'
