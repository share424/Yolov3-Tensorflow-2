import json
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input", help="mscoco annotation file")
    parser.add_argument("output", help="output file")
    args = parser.parse_args()

    print('Reading dataset...')
    with open(args.input, "r") as file:
        dataset = json.load(file)

    images = {}
    for image in tqdm(dataset['images'], 'Indexing...'):
        images[image['id']] = image
    
    annotations = {}
    for ant in tqdm(dataset['annotations'], "Parsing..."):
        id = ant['image_id']
        name = '%012d.jpg' % id
        cat = ant['category_id']
        bbox = ant['bbox']

        # gatau kenapa category di mscoco ada yg bolong id nya
        if cat >= 1 and cat <= 11:
            cat = cat - 1
        elif cat >= 13 and cat <= 25:
            cat = cat - 2
        elif cat >= 27 and cat <= 28:
            cat = cat - 3
        elif cat >= 31 and cat <= 44:
            cat = cat - 5
        elif cat >= 46 and cat <= 65:
            cat = cat - 6
        elif cat == 67:
            cat = cat - 7
        elif cat == 70:
            cat = cat - 9
        elif cat >= 72 and cat <= 82:
            cat = cat - 10
        elif cat >= 84 and cat <= 90:
            cat = cat - 11

        # get image actual size
        img = images[id]
        w, h = img['width'], img['height']

        # save in format x, y, w, h, c
        if(name not in annotations):
            annotations[img['file_name']] = [[bbox[0]/w, bbox[1]/h, bbox[2]/w, bbox[3]/h, cat]]
        else:
            annotations[img['file_name']].append([bbox[0]/w, bbox[1]/h, bbox[2]/w, bbox[3]/h, cat])
    
    print("Saving")
    with open(args.output, "w") as file:
        json.dump(annotations, file)

    if('categories' in dataset):
        categories = []
        for cat in tqdm(dataset['categories'], 'Parsing categories'):
            categories.append(cat['name'])
        print('Saving categories')
        with open('categories.txt', 'w') as file:
            file.write('\n'.join(categories))

    
