import os, sys
#sys.path.append('/screamlab/home/eri24816/improved-diffusion')
from os import path 
from utils.pianoroll import PianoRoll, PianoRollDataset
from tqdm import tqdm

from metrics import iou, mse

def sort_by_similarity(x,ys : 'list[PianoRoll]' ,sim_function = mse):
    scores={} 
    x = x.to_tensor()/128
    for y in tqdm(ys):
        scores[y]=sim_function(x,y.to_tensor(0,x.shape[0],padding=True)/128).item()
    ys = sorted(ys,key=lambda y: scores[y])
    scores = sorted(scores.values())
    return ys, scores, x

from typing import List
def get_sim_map(q,k,sim_function):
    m = []
    for k_bar in k.split(32):
        m.append(sim_function(q,k_bar))
    return m

def get_sim_maps(q,ks,sim_function = iou):
    all_sim_maps : List[List] = [] # [key_song, query_pos, key_pos]
    for k in tqdm(ks):
        sim_maps : List[List] = [] # [query_pos, key_pos]
        k = k.to_tensor()/128
        for q_bar in q.split(32):
            sim_maps.append(get_sim_map(q_bar,k,sim_function))
        all_sim_maps.append(sim_maps)
    return all_sim_maps

def rank_similiar_parts(all_sim_maps, q_start=0, q_end=None):
    # rank similarity maps
    scores = []
    for i_key,sim_maps in enumerate(all_sim_maps):
        sim_maps = sim_maps[q_start:q_end]
        q_len = len(sim_maps)
        k_len = len(sim_maps[0])
        for q_pos_on_k in range(k_len - q_len): # move query window over key
            score = 0
            for q_pos in range(q_len):
                score += sim_maps[q_pos][q_pos_on_k+q_pos] # sum similarity of query bars

            scores.append((i_key,q_pos_on_k,score.item()/q_len))
    scores = sorted(scores,key=lambda x: x[2],reverse=True)
    return scores

def process_query(q_file,ks,sim_function = iou):
    global all_sim_maps
    # load query
    q = PianoRoll.from_midi(q_file).to_tensor(0,16*32,True)/128

    # get bar to bar similarity maps. [key_song, query_pos, key_pos]
    all_sim_maps = get_sim_maps(q,ks,sim_function)

    # prepare save path
    from os import path
    import json
    q_id = path.basename(q_file).replace('.mid','')
    q_dir = path.dirname(q_file)
    save_dir = path.join(q_dir,'similar',q_id)
    os.makedirs(save_dir,exist_ok=True)

    # prepare metadata
    metadata_path = path.join(save_dir,'metadata.json')

    files_metadata = {}

    for q_start in range(0,16,4): # move query window over query (4 bars at a time)
        q_end = q_start + 4

        # rank similar keys
        scores = rank_similiar_parts(all_sim_maps, q_start=q_start, q_end=q_end)

        # save query
        q_path = path.join(save_dir,f'{q_start}_query.mid')
        PianoRoll.from_tensor(q*128).slice(q_start*32,q_end*32).to_midi(q_path)
        files_metadata[path.basename(q_path)]={
            'title':f'Query {q_id}',
            'start':q_start,
            'end':q_end,
            'score':None
        }

        for rank,(i_key, q_pos_on_k, score) in enumerate(scores[:10]):
            #print(i_key, q_pos_on_k, score,ks[i_key].metadata.name)
            k = ks[i_key]
            k_path = path.join(save_dir,f'{q_start}_{rank}.mid')
            k.slice(q_pos_on_k*32,(q_pos_on_k+q_end-q_start)*32).to_midi(k_path)

            files_metadata[path.basename(k_path)]={
                'title':k.metadata.name,
                'start':q_pos_on_k,
                'end':q_pos_on_k + q_end - q_start,
                'score':score
            }
    
    # save metadata
    metadata = {'files':files_metadata}
    with open(metadata_path,'w') as f:
        json.dump(metadata,f)
    
    # update global metadata
    global_metadata_path = path.join(q_dir,'similar','metadata.json')
    if not path.exists(global_metadata_path):
        global_metadata = {'queries':[]}
    else:
        with open(global_metadata_path) as f:
            global_metadata = json.load(f)
    global_metadata['queries'].append(q_id)
    with open(global_metadata_path,'w') as f:
        json.dump(global_metadata,f)

if __name__ == '__main__': 
    pass
    data_dir = '/screamlab/home/eri24816/pianoroll_dataset/data/dataset_1/pianoroll/'
    songs = PianoRollDataset(data_dir,32,32,metadata_file='/screamlab/home/eri24816/pianoroll_dataset/data/dataset_1/metadata.csv')
    len(songs.pianorolls)

    for i in range(0,32):
        q_file = f'../log/ema_0.9999_2700000/{i}.mid'
        ks = songs.pianorolls
        process_query(q_file,ks,sim_function=iou)