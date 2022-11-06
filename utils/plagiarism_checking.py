import os, sys
from os import path

from utils.pianoroll import PianoRoll, PianoRollDataset
from tqdm import tqdm

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

def sort_by_similarity(x,ys : 'list[PianoRoll]' ,sim_function = mse):
    scores={} 
    x = x.to_tensor()/128
    for y in tqdm(ys):
        scores[y]=sim_function(x,y.to_tensor(0,x.shape[0],padding=True)/128).item()
    ys = sorted(ys,key=lambda y: scores[y])
    scores = sorted(scores.values())
    return ys, scores, x

if __name__ == '__main__': 

    # keys
    data_dir = '/home/eri24816/pianoroll'
    keys = PianoRollDataset(data_dir,32,32).get_all_piano_rolls()

    # query
    query_file = sys.argv[1]
    query = PianoRoll.from_midi(query_file).slice(32*15,32*16)
    
    # save query
    p = query_file.replace('.mid','similar')+'/query.mid'
    os.makedirs(path.dirname(p),exist_ok=True)
    query.to_midi(p)

    keys.append(query) # add query to keys to check similarity function
    
    # sort by similarity
    ys, scores, x = sort_by_similarity(query,keys,mse)
    print(scores[:10],scores[-10:])

    # save 10 most similar
    for i, y in enumerate(list(reversed(ys))[:10]):
        p = query_file.replace('.mid','similar')+f'/{i}.mid'
        os.makedirs(path.dirname(p),exist_ok=True)
        y.to_midi(p)



    print('done')