import numpy as np

# Check if given slope and intercept is within set boundary
# input: list of slope, intercept
# return: number of (slope, intercept) pairs within boundary
def score(slope, intercept):
    cnt = 0
    gt = np.genfromtxt('line_gt.csv', delimiter=',')
    slope_max, slope_min, int_max, int_min = gt
    for i in range(len(slope)):
        s = slope[i]
        ic = intercept[i]
        if (slope_min[i] < s < slope_max[i]) and (int_min[i] < ic < int_max[i]):
            cnt+=1
    return cnt