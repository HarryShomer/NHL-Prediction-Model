"""
These are all the constant used in the marcels calculations
"""


# Weight given to a year
# 0 = that year, 1 is year b3 ....
marcel_weights = {
    "F": {
        'gs60': [.62, .22, .16],
        'toi/gp': [.9, .1, 0]
    },
    "D": {
        'gs60': [.6, .25, .15],
        'toi/gp': [.85, .15, 0]
    }
}

# Regression constant x/(x+c)
reg_consts = {
    "F": {"gs60": 540, "toi/gp": 370},
    "D": {"gs60": 657, "toi/gp": 710},
}

# What we are regressing towards
reg_avgs = {
    # Where to regress to
    "F": {"gs60": 1.5, "toi/gp": 14.93},
    "D": {"gs60": .95, "toi/gp": 19.7},
}