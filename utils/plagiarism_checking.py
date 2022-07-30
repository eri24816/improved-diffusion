from utils.pianoroll import PianoRoll, PianoRollDataset
from tqdm import tqdm
def cosine_sim(x,y,eps = 1):
    return ((x*y).sum()+eps)/(x.norm()*y.norm()+eps)

def dot_sim(x,y):
    return ((x*y).sum())

def proj_sim(x,y):
    return ((x*y).sum())/x.norm()


def sort_by_similarity(x: PianoRoll,ys : 'list[PianoRoll]' ,sim_function = cosine_sim):
    scores={} 
    x = x.to_tensor(0,64,True)
    for y in tqdm(ys):
        scores[y]=sim_function(x,y.to_tensor(0,64,padding=True))
    ys = sorted(ys,key=lambda y: scores[y])
    scores = sorted(scores.values())
    return ys, scores

def check(target_pr, ds):
    ys, scores = sort_by_similarity(target_pr,ds)



if __name__ == '__main__':
    data_dir = '/screamlab/home/eri24816/pianoroll_dataset/data/dataset_1/pianoroll'
    ds = PianoRollDataset(data_dir)

    sample_dir = 'log/2bb/samples/ema_0.9999_1430000/63.mid'
    target_pr = PianoRoll.from_midi(sample_dir)
    check(target_pr,ds)