import os
import json
import time
import random
import logging
import urllib.request
import urllib.error
from urllib.parse import urlparse, quote
import math

from multiprocessing import Pool
from user_agent import generate_user_agent
from selenium import webdriver
from selenium.webdriver.common.keys import Keys



def get_image_links(main_keyword, supplemented_keywords, link_file_path):
    print("*****************get_image_links**************************")
    #Call get_image_links with args:  Szczecin {'name': 'Dworzec Główny budynek', 'id': 1, 'pictures': 30} ./data/link_files/

    chromedriver="/usr/local/bin/chromedriver"
    options = webdriver.ChromeOptions()
    options.add_argument('--no-sandbox')
    options.add_argument("--headless")
    try:
        driver = webdriver.Chrome(chromedriver, options=options)
    except Exception as e:
        print("chromedriver not found (use the '--chromedriver' argument to specify the path to the executable)"
                "or google chrome browser is not installed on your machine (exception: %s)" % e)
        sys.exit()

    number_img_urls = 0    

    for i in range(len(supplemented_keywords)):
        img_urls = set()
        search_query = quote(main_keyword + ' ' + supplemented_keywords[i]["name"])
        #num_requested = supplemented_keywords[i]["pictures"]
        num_requested = 50
        #num_requested = 1
        # number_of_scrolls * 400 images will be opened in the browser
        number_of_scrolls = int(num_requested / 400) + 1 
        #Images: tbm=isch; source=lnms: origin of the search website, application, browser extension, Local Network Management Systemetc.
        #parameters explained https://www.reddit.com/r/explainlikeimfive/comments/2ecozy/eli5_when_you_search_for_something_on_google_the/
        #When you search for images, TBM=isch, you can also use the following TBS values:
        #https://stenevang.wordpress.com/2013/02/22/google-advanced-power-search-url-request-parameters/
        #images_type = quote('isz:l')
        #url = "https://www.google.com/search?q="+search_query+"&source=lnms&tbm=isch&tbs="+images_type
        url = "https://www.google.pl/search?q="+search_query+"&source=lnms&tbm=isch"
        driver.get(url)
        for _ in range(number_of_scrolls):
            for __ in range(10):
                # multiple scrolls needed to show all 400 images
                driver.execute_script("window.scrollBy(0, 1000000)")
                time.sleep(2)
            # to load next 400 images
            time.sleep(1)
            try:
                driver.find_element_by_xpath("//input[@value='Show more results']").click()
            except Exception as e:
                print("Process-{0} reach the end of page or get the maximum number of requested images".format(main_keyword))
                break

        # imges = driver.find_elements_by_xpath('//div[@class="rg_meta"]') # not working anymore
        # imges = driver.find_elements_by_xpath('//div[contains(@class,"rg_meta")]') # not working anymore
        
        thumbs = driver.find_elements_by_xpath('//a[@class="wXeWr islib nfEiy mM5pbd"]')

        #print(len(thumbs))
        
        """
        #random thumbs
        thumbsLength=len(thumbs)
        randomThumbsIndex=[]
        
        for _ in range(num_requested):
            r=random.randint(0,thumbsLength-1)
            while (r in randomThumbsIndex):
                r=random.randint(0,thumbsLength-1)
            randomThumbsIndex.append(r)
        """
        #for thumb in thumbs:
        #for index in randomThumbsIndex:
        for index in range(num_requested):
            #print(index)
            thumb = thumbs[index]
            try:
                thumb.click()
                time.sleep(1)
            except e:
                print("Error clicking one thumbnail")

            url_elements = driver.find_elements_by_xpath('//img[@class="n3VNCb"]')
            for url_element in url_elements:
                try:
                    url = url_element.get_attribute('src')
                    #time.sleep(1)
                except e:
                    print("Error getting one url")

                if url.startswith('http') and not url.startswith('https://encrypted-tbn0.gstatic.com'):
                    img_urls.add(url)
                    #print("Found image url: " + url)
        print('Process-{0} add keyword {1} , got {2} image urls so far'.format(main_keyword, supplemented_keywords[i]["name"], len(img_urls)))
        number_img_urls += len(img_urls)
        store_links_in_file(link_file_path=link_file_path + str(supplemented_keywords[i]["id"]),img_urls=img_urls)
    
    print('Process-{0} totally get {1} images'.format(main_keyword, number_img_urls))
    #driver.quit()
    
    

def store_links_in_file(link_file_path,img_urls):
    with open(link_file_path, 'w') as wf:
        for url in img_urls:
            wf.write(url +'\n')
    #print('Store all the links in file {0}'.format(link_file_path))      
    

def download_images(main_keyword, supplemented_keywords, base_link_file_path, download_dir, log_dir):

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = log_dir + 'download_selenium_{0}.log'.format(main_keyword)
    logging.basicConfig(level=logging.DEBUG, filename=log_file, filemode="a+", format="%(asctime)-15s %(levelname)-8s  %(message)s")        
    
    for i in range(len(supplemented_keywords)):
        link_file_path = base_link_file_path + str(supplemented_keywords[i]["id"])
        print('Start downloading with link file {0}..........'.format(link_file_path))
        img_dir = download_dir + main_keyword + '/' + str(supplemented_keywords[i]["id"]) + '/'
        count = 0
        headers = {}
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        # start to download images
        with open(link_file_path, 'r') as rf:
            for link in rf:
                try:
                    print("hola1")
                    o = urlparse(link)
                    ref = o.scheme + '://' + o.hostname
                    #ref = 'https://www.google.com'
                    ua = generate_user_agent()
                    headers['User-Agent'] = ua
                    headers['referer'] = ref
                    #print('\n{0}\n{1}\n{2}'.format(link.strip(), ref, ua))
                    req = urllib.request.Request(link.strip(), headers = headers)
                    response = urllib.request.urlopen(req)
                    data = response.read()
                    file_path = img_dir + '{0}.jpg'.format(count)
                    with open(file_path,'wb') as wf:
                        wf.write(data)
                    print('Process-{0} download image {1}/{2}.jpg'.format(main_keyword, str(supplemented_keywords[i]["id"]), count))
                    count += 1
                    if count % 10 == 0:
                        print('Process-{0} is sleeping'.format(main_keyword))
                        time.sleep(5)

                except urllib.error.URLError as e:
                    print('URLError')
                    logging.error('URLError while downloading image {0}reason:{1}'.format(link, e.reason))
                    continue
                except urllib.error.HTTPError as e:
                    print('HTTPError')
                    logging.error('HTTPError while downloading image {0}http code {1}, reason:{2}'.format(link, e.code, e.reason))
                    continue
                except Exception as e:
                    print('Unexpected Error')
                    logging.error('Unexpeted error while downloading image {0}error type:{1}, args:{2}'.format(link, type(e), e.args))
                    continue        
        



if __name__ == "__main__":

    download_dir = './data/'
    link_files_dir = './data/link_files/'
    log_dir = './logs/'
    for d in [download_dir, link_files_dir, log_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    with open('landmarks.json') as f:
        landmarks_json = json.load(f)

    keyword = landmarks_json['city']
    supplemented_keywords = landmarks_json['landmarks']
    
    groups = math.ceil(len(supplemented_keywords) / 10)
    
    ###################################
    # get image links and store in file
    ###################################    
    
    startIndex = 0
    endIndex = 10
    
    t1 = time.time()
    p = Pool()
    print('Waiting for all subprocesses done...')
    for i in range(groups):
        p.apply_async(get_image_links, args=(keyword,supplemented_keywords[startIndex:endIndex],link_files_dir))
        startIndex += 10
        endIndex += 10
    p.close()
    p.join()
    print("Pool took: ", time.time()-t1)
    print('All subprocesses done.')       
    
    
    ###################################
    # download images with link file
    ###################################    
    
    startIndex = 0
    endIndex = 10
    
    t1 = time.time()
    p = Pool()
    print('Waiting for all subprocesses done...')
    for i in range(groups):
        p.apply_async(download_images, args=(keyword,supplemented_keywords[startIndex:endIndex],link_files_dir,download_dir,log_dir))
        startIndex += 10
        endIndex += 10
    p.close()
    p.join()
    print("Pool took: ", time.time()-t1)
    print('All subprocesses done.')