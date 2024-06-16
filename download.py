# Brian Lesko
# 6/16/2024
# Download images from Flickr using the Flickr API

import flickrapi
import requests
import skimage.io
from io import BytesIO
import os
from mysecrets import api_key, api_secret

flickr = flickrapi.FlickrAPI(api_key, api_secret)
keywords = ['thumbs down']

# Ensure the images directory exists
for keyword in keywords:
    if not os.path.exists('images'):
        os.makedirs(f'{keyword}')

num_images = 3000
for keyword in keywords:
    photos = flickr.walk(text=keyword, tag_mode='all', tags=keyword, extras='url_c', sort='relevance', per_page=100)
    itrain = 0  # initialize counter
    for photo in photos:
        if itrain >= num_images:
            break 
        url = photo.get('url_c')
        if url:
            try:
                response = requests.get(url)
                img = skimage.io.imread(BytesIO(response.content))
                # img = skimage.transform.resize(img, (64, 64)) # Optional: resize the image
                file_name = f'{keyword}/{keyword}_{itrain}.jpg'
                skimage.io.imsave(file_name, img)
                itrain += 1
            except Exception as e:
                print(f'Failed to process image {url}: {str(e)}')

# Count how many images we got
for keyword in keywords:
    keyword_images = [f for f in os.listdir(f'{keyword}') if f.startswith(keyword)]
    print(f'Number of {keyword} images:', len(keyword_images))

