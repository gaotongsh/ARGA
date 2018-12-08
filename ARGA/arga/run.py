import settings

from clustering import Clustering_Runner
from link_prediction import Link_pred_Runner

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

dataname = '698'       # 'cora' or 'citeseer' or 'pubmed'
model = 'arga_ae'          # 'arga_ae' or 'arga_vae'
task = 'clustering'         # 'clustering' or 'link_prediction'

# dataname = [0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980]
dataname = [3980]

for d in dataname:
    s = settings.get_settings(str(d), model, task)
    if task == 'clustering':
        runner = Clustering_Runner(s)
    else:
        runner = Link_pred_Runner(s)

    runner.erun()

