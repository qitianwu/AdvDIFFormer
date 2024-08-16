import argparse
import os
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import AllChem

import json
import seaborn as sns


def draw(graph):
    # cl1 = list(sns.hls_palette(20, h=.5))
    # cl2 = list(sns.hls_palette(20, s=.4, h=.5))
    cl1 = list(sns.hls_palette(15, l=0.6, s=0.8))
    cl2 = list(sns.hls_palette(15, l=0.6, s=0.8))
    # colors = [
    #     '#b35806', '#f1a340', '#fee0b6', '#d8daeb', '#998ec3', '#542788',
    #     '#b2182b', '#ef8a62', '#fddbc7', '#d1e5f0', '#67a9cf', '#2166ac'
    # ]
    # cl1 = cl2 = list(sns.color_palette(colors))
    colors = [
        '#d7191c', '#fdae61', '#E9E308', '#abdda4', '#2b83ba', '#d01c8b',
        '#7b3294', '#a6611a', '#2166ac', '#fddbc7', '#d1e5f0', '#67a9cf'
    ]
    cl1 = cl2 = list(sns.color_palette(colors))

    s = graph['smiles']
    m = Chem.MolFromSmiles(s)
    rdDepictor.Compute2DCoords(m)
    aa = []
    ele = []
    atms = {}
    for nodes in graph['nodes']:
        j = int(nodes['cg'])
        k = int(nodes['id'])
        z = str(nodes['element'])
        ele.append(z)
        aa.append(k)
        value = {k: cl1[j]}
        atms.update(value)
    bl = []
    edges = {}
    a = (graph['nodes'])
    for g in range(m.GetNumBonds()):
        begin = m.GetBonds()[g].GetBeginAtomIdx()
        end = m.GetBonds()[g].GetEndAtomIdx()
        if a[begin]['cg'] == a[end]['cg']:
            bl.append(g)
            bond_value = {g: cl2[a[begin]['cg']]}
            edges.update(bond_value)
    drawer = rdMolDraw2D.MolDraw2DSVG(300, 200)
    drawer.DrawMolecule(
        m, highlightAtoms=aa, highlightBonds=bl,
        highlightAtomColors=atms, highlightBondColors=edges
    )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace('svg:', '')
    return svg


def update_nodes(graph, labels):
    block = {}
    labels.sort(key=lambda x: min(x))
    for idx, px in enumerate(labels):
        block.update({x: idx for x in px})
    new_nodes = []
    for x in graph['nodes']:
        x['cg'] = block[x['id']]
        new_nodes.append(x)
    graph['nodes'] = new_nodes
    return graph


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    backs = ['gat', 'gcn', 'ours-series', 'difformer', 'graphgps']
    max_clus_num = 0

    for x in os.listdir(os.path.join(args.input_dir, 'gat')):
        if not x.endswith('.json'):
            continue
        with open(os.path.join(args.input_dir, 'gat', x)) as Fin:
            INFO = json.load(Fin)

        out_dir = os.path.join(args.output_dir, x)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        INFO = update_nodes(INFO, INFO['cgnodes'])
        max_clus_num = max(max_clus_num, len(INFO['cgnodes']))
        gt_img = draw(INFO)
        with open(os.path.join(out_dir, 'gt.svg'), 'w') as Fout:
            Fout.write(gt_img)

        INFO = update_nodes(INFO, INFO['pred'])
        max_clus_num = max(max_clus_num, len(INFO['pred']))
        gat_img = draw(INFO)
        out_file = os.path.join(out_dir, f'gat_ami_{INFO["AMI"]}.svg')
        with open(out_file, 'w') as Fout:
            Fout.write(gat_img)

        for back in backs[1:]:
            with open(os.path.join(args.input_dir, back, x)) as Fin:
                INFO = json.load(Fin)
            out_file = os.path.join(out_dir, f'{back}_ami_{INFO["AMI"]}.svg')
            INFO = update_nodes(INFO, INFO['pred'])
            max_clus_num = max(max_clus_num, len(INFO['pred']))
            with open(out_file, 'w') as Fout:
                Fout.write(draw(INFO))

    print('[INFO] max cluster:', max_clus_num)
