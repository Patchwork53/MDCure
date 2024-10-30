import torch
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import random
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd 

def align(snippet_idx, snippet_emb, all_embs, all_sents_in_cluster, sim_threshold=0.8):
    """
    Helper to align given snippet with all other snippets in the context documents and select the highest-alignment distinct snippet to pair with given snippet
    """
    # remove reference snippet from full list of snippets & embeddings
    del all_sents_in_cluster[snippet_idx]

    # align reference snippet with other snippets
    cos_sims = util.cos_sim(snippet_emb, all_embs)[0].cpu().numpy()

    # identify index of highest-alignment snippet that is not too similar (and not same snippet)
    mask = cos_sims < sim_threshold
    try:
        max_sim_idx = np.nonzero(mask)[0][np.argmax(cos_sims[mask])]
    except: # if no cosine similarities are less than threshold
        max_sim_idx = -1
    
    # if length of second snippet not good
    if len(all_sents_in_cluster[max_sim_idx][0].split())<=10:
        return -1

    return max_sim_idx


def generate_and_save_snippet_pairs(
    input_dir,
    output_dir,
):
    """
    Function to save pairs of related snippets across all document clusters
    """
    # instantiate sentence embedding model
    emb_model = SentenceTransformer('sentence-transformers/msmarco-distilbert-cos-v5')
            
    # access all newshead file paths
    all_files = [
        f
        for f in os.listdir(input_dir)
        if f.endswith(".pt")
    ]
    all_files = all_files[::-1]
    count = 0   
    
    # consider each data split (train/valid/test)
    for file_idx, file_name in enumerate(tqdm(all_files)):
        print(file_name)

        # load the data
        all_clusters = torch.load(os.path.join(input_dir, file_name))
        all_source_docs = []
        all_answers = []
        all_cluster_ids = []

        for cluster in all_clusters:
            print("on cluster",count)
            
            # access all segments separated by \n in the dataset and embed them
            all_sents_in_cluster = [(sentence, emb_model.encode(sentence), document) for document in cluster for sentence in document.split('\n')]
            all_embs = emb_model.encode(list(zip(*all_sents_in_cluster))[0])

            selected_sent_pairs = []

            # select specified number of snippet pairs for instruction generation
            for i in range(len(all_sents_in_cluster)//2):
                
                # randomly select a snippet index
                snippet1_idx = random.choice(range(len(all_sents_in_cluster)))
                snippet1 = all_sents_in_cluster[snippet1_idx]
                snippet1_emb = all_embs[snippet1_idx]

                # don't use snippets that are too short (length <10 words)
                if len(snippet1[0].split())<=10:
                    pass

                else:
                    # identify highest-alignment snippet that is not too similar
                    all_embs = np.delete(all_embs, snippet1_idx, axis=0)
                    snippet2_idx = align(snippet1_idx, snippet1_emb, all_embs, all_sents_in_cluster, sim_threshold=0.7)
                    
                    # if there is no such snippet then do nothing
                    if snippet2_idx == -1:
                        pass
                    # otherwise, add the pair to the snippet pair pool
                    else:
                        snippet2 = all_sents_in_cluster[snippet2_idx]

                        selected_sent_pairs.append([(snippet1[0],snippet1[2]), (snippet2[0], snippet2[2])])

                        # remove the second snippet from the pool
                        del all_sents_in_cluster[snippet2_idx]
                        all_embs = np.delete(all_embs, snippet2_idx, axis=0)

            # save snippet pairs to list
            for [(snippet1, doc1),(snippet2, doc2)] in selected_sent_pairs:
                if doc1 != doc2:
                    all_source_docs.append([doc1, doc2])
                    all_answers.append([snippet1, snippet2])
                    all_cluster_ids.append(count)
                
            count += 1

        print("finished selecting snippets for", file_name)
        
        # ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # save results to CSV
        data = {"articles": all_source_docs, "answer": all_answers, "cluster_id": all_cluster_ids}
        snippet_data = pd.DataFrame(data)
        snippet_data_csv_name = os.path.join(output_dir,file_name.replace('.pt', '.csv'))
        snippet_data.to_csv(snippet_data_csv_name, index=False)
        print("Finished saving snippet pairs to %s"%(snippet_data_csv_name))

    print("done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./0_source_data_preparation/base_articles",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./0_source_data_preparation/base_snippet_pairs",
    )
    
    args = parser.parse_args()
    print(args)

    current_date_time = str(datetime.now().strftime("%Y-%m-%d_hr%H-min%M"))
    
    generate_and_save_snippet_pairs(
        args.input_dir,
        args.output_dir,
    )






