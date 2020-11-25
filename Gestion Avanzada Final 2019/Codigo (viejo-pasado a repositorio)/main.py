import os, sys
import argparse
import requests
import psycopg2
import torch
import numpy as np 

from config import config
from img2vec_pytorch import Img2Vec
from PIL import Image

def convertArray(numpyArray):
    newArray = []
    for elem in numpyArray:
        newArray.append(elem)
    return newArray

def tensorToString(embeddingsVector):
    numpyVector = np.array(embeddingsVector.unsqueeze(0).tolist())
        
    stringVector = str(convertArray(numpyVector.flatten()))
        
    stringVector = '{' + stringVector[1:]
    stringVector = stringVector[:-1] + '}'

    return stringVector


def loadPivots(path):

    params = config() 
    print('Connecting to the PostgreSQL database...')
    conn = psycopg2.connect(**params)
    cur = conn.cursor() 
    
    dirs = os.listdir(path)
    i = 0
    for item in dirs:
        i = i + 1
        image = Image.open(path+item)
        embeddingsVector = extractVector(resizeImage(image))
        
        stringVector = tensorToString(embeddingsVector) 
        cur.execute('INSERT INTO pivotes (id,vector,nivel) VALUES (%s,%s,%s);',[i,stringVector,i])

    conn.commit()
    cur.close()
    if conn is not None:
        conn.close()
        print('Database connection closed.')

def loadImages(path):

    params = config() 
    print('Connecting to the PostgreSQL database...')
    conn = psycopg2.connect(**params)
    cur = conn.cursor() 
    
    dirs = os.listdir(path)
    for item in dirs:
        image = Image.open(path+item)
        embeddingsVector = extractVector(resizeImage(image))
        
        stringVector = tensorToString(embeddingsVector) 
        cur.execute('INSERT INTO images (name,vector) VALUES (%s,%s);',[item,stringVector])

    conn.commit()
    cur.close()
    if conn is not None:
        conn.close()
        print('Database connection closed.')

def readImage(path):
    return Image.open(path)

def getNCloseImages(imagePath,n):
        
    params = config() 
    print('Connecting to the PostgreSQL database...')
    conn = psycopg2.connect(**params)
    cur = conn.cursor() 
    
    image = Image.open(imagePath)
    embeddingsVector = extractVector(resizeImage(image))
    
    stringVector = tensorToString(embeddingsVector)

    cur.execute('SELECT queryimage(%s,%s);',[stringVector,n])

    resultImages = cur.fetchall() 

    print(resultImages)
    
    cur.close()
    if conn is not None:
        conn.close()
        print('Database connection closed.')

def cleanImage(imagePath):
    response = requests.post(
        'https://api.remove.bg/v1.0/removebg',
        files={'image_file': open(imagePath, 'rb')},
        data={'size': 'auto'},
        headers={'X-Api-Key': 'xXGpv8ncn1vC7eXyZJ8RUP3K'},
    )
    if response.status_code == requests.codes.ok:
        base=os.path.basename(imagePath)
        with open(os.path.splitext(base)[0]+'.png', 'wb') as out:
            out.write(response.content)
    else:
        print("Error:", response.status_code, response.text)

def resizeImage(im):
    final_size = 224
    size = im.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    im = im.resize(new_image_size, Image.ANTIALIAS)

    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))

    return new_im

def extractVector(image):
    img2vec = Img2Vec(cuda=False)
    vec = img2vec.get_vec(image, tensor=True)
    return vec

def main():
    parser = argparse.ArgumentParser(description='Tool for ')

    parser.add_argument("--path", type=str, required=True, help="path of image or folder")
    parser.add_argument("--action", choices=["queryImage", "loadImages","loadPivots"], required=True, type=str, help="action to apply")
    parser.add_argument("--range", type=int, default=5, help="quantity of closest images (default = 5)")

    args = parser.parse_args()

    if args.action == 'queryImage':
        #cleanImage(args.path)
     
        base = os.path.basename(args.path)
        imageFullPath = os.path.abspath(os.path.splitext(base)[0] + '.png')
            
        listOfImages = getNCloseImages(imageFullPath,args.range)

        print(listOfImages)

    elif args.action == 'loadImages':
        loadImages(args.path)

    elif args.action == 'loadPivots':
        loadPivots(args.path)

if __name__ == '__main__':
    main()
