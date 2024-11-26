import array
import gzip
import json
import os
import string
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer

folder = './beauty'
bert_path = './sentence-bert/all-mpnet-base-v2'
bert_model = SentenceTransformer(bert_path)
core = 5

print("----------Build dict----------")
item_dict, reverse_item_dict = {}, {}
with open(folder + 'item_list.txt', 'r') as f:
    for line in f:
        item, id = line.split()
        item_dict[item] = int(id)
    f.close()

print("----------Search meta data----------")
jsons = {}
tag_num = {}
with open(folder + 'meta.json', 'r') as f:
    for line in f:
        a = json.loads(line)
        if a['asin'] in item_dict:
            jsons[item_dict[a['asin']]] = []
            if 'categories' in a:
                for tags in a['categories']:   # categories--2014   category--2018
                    for tag in tags:            # for tag in tags--2014
                        jsons[item_dict[a['asin']]].append(tag)
                        if tag not in tag_num:
                            tag_num[tag] = 1
                        else:
                            tag_num[tag] += 1
            if 'brand' in a:
                jsons[item_dict[a['asin']]].append(a['brand'])
                if a['brand'] not in tag_num:
                    tag_num[a['brand']] = 1
                else:
                    tag_num[a['brand']] += 1
            if 'title' in a:
                jsons[item_dict[a['asin']]].append(a['title'])
            if 'description' in a:
                jsons[item_dict[a['asin']]].append(a['description'])   # append-2014   extend--2018
    f.close()

# print(len(item_dict))
# print(len(jsons))
# assert len(item_dict) == len(jsons)

item_tag_adj = {}
tag_dict = {}
for asin in jsons.keys():
    i = 0
    item_tag_adj[asin] = []
    while i < len(jsons[asin]):
        if jsons[asin][i] in tag_num and tag_num[jsons[asin][i]] > len(jsons) * 0.25:
            del jsons[asin][i]
            i -= 1
        else:
            if jsons[asin][i] not in tag_dict:
                tag_dict[jsons[asin][i]] = len(tag_dict)
            item_tag_adj[asin].append(tag_dict[jsons[asin][i]])
        i += 1

text = []
for i in range(len(item_dict)):
    string = ''
    if i in jsons:
        for info in jsons[i]:
            string += info + ' '
    else:string = 'None'
    text.append(string.replace('\n', ' '))

print("----------Text Features----------")
sentence_embeddings = bert_model.encode(text)
assert sentence_embeddings.shape[0] == len(item_dict)
np.save(os.path.join(folder, 'item_text_feat.npy'), sentence_embeddings)
with open(os.path.join(folder, 'item_tag_adj.json'), 'w') as f:
    json.dump(item_tag_adj, f)
    f.close()
