def cosine_sim(x,y,eps = 1):
    return ((x*y).sum()+eps)/(x.norm()*y.norm()+eps)

def dot_sim(x,y):
    return ((x*y).sum())

def proj_sim(x,y):
    return ((x*y).sum())/x.norm()

def mse(x,y):
    return -((x-y)**2).mean()

def iou(x,y):
    return (x*y).sum()/(x+y-x*y).sum()