# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------

# Finds all HTTP URLs found in the NimbusML repository
# Converts all valid HTTP links to HTTPS
# Usage: python3 findHttpURLs.py [PATH_TO_NimbusML_REPOSITORY]
# Output: Report_AlterableUrls_FindHttpURLs.csv, [Report_NonAlterableUrls_FindHttpURLs.csv, Report_InvalidUrls_FindHttpURLs.csv]

# Required non-standard pip library: urlextract

import sys
import os
import requests
import csv
import collections
from urlextract import URLExtract

def addToDictionary(dict, key, value):
    if key not in dict:
        dict[key] = [value]
    else:
        if value not in dict[key]:
            dict[key].append(value)
    return dict

def findHttpUrls(searchRootDirectory):
    alterableUrlsStore = {}
    nonAlterableUrlsStore = {}
    invalidUrlsStore = {}
    extractor = URLExtract()
    lengthOfOriginalRootPath = -1
    for root, _, files in os.walk(searchRootDirectory, onerror=None):
        if lengthOfOriginalRootPath == -1:
             lengthOfOriginalRootPath = len(root)
        for filename in files:
            absoluteFilePath = os.path.join(root, filename)
            relativeFilePath = '.' + absoluteFilePath[lengthOfOriginalRootPath:]
            try:
                with open(absoluteFilePath, "rb") as f:
                    data = f.read()
                    try:
                        data = data.decode("utf-8")  
                    except Exception as e:
                        print("Unable to decodefile: {} in UTF-8 Encoding.".format(relativeFilePath)) 
                        print(str(e))
                        continue
                    currentUrlList = extractor.find_urls(data)
                    currentUrlList = [url for url in currentUrlList if url[:5] == "http:"]
                    for selectedUrl in currentUrlList:
                        try:
                            request = requests.get(selectedUrl)
                            if request.status_code == 200:
                                changedSelectedUrl = selectedUrl.replace("http", "https")
                                try:
                                    newRequest = requests.get(changedSelectedUrl)
                                    if newRequest.status_code == 200:
                                        alterableUrlsStore = addToDictionary(alterableUrlsStore, relativeFilePath, selectedUrl)
                                    else:
                                        nonAlterableUrlsStore = addToDictionary(nonAlterableUrlsStore, relativeFilePath, selectedUrl)
                                except:
                                    nonAlterableUrlsStore = addToDictionary(nonAlterableUrlsStore, relativeFilePath, selectedUrl)
                            else:
                                invalidUrlsStore = addToDictionary(invalidUrlsStore, relativeFilePath, selectedUrl)
                        except ConnectionError:
                            invalidUrlsStore = addToDictionary(invalidUrlsStore, relativeFilePath, selectedUrl)
            except (IOError, OSError):
                pass
    makeReports(alterableUrlsStore, nonAlterableUrlsStore, invalidUrlsStore)

def makeReports(alterableUrlsStore, nonAlterableUrlsStore, invalidUrlsStore):
    with open('Report_AlterableUrls_FindHttpURLs.csv', mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["url", "relativeFilepath"])
        for fileKey in alterableUrlsStore:
            for url in alterableUrlsStore[fileKey]:
                writer.writerow([url, fileKey])
    with open('Report_NonAlterableUrls_FindHttpURLs.csv', mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["url", "relativeFilepath"])
        for fileKey in nonAlterableUrlsStore:
            for url in nonAlterableUrlsStore[fileKey]:
                writer.writerow([url, fileKey])
    with open('Report_InvalidUrls_FindHttpURLs.csv', mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["url", "relativeFilepath"])
        for fileKey in invalidUrlsStore:
            for url in invalidUrlsStore[fileKey]:
                writer.writerow([url, fileKey])
    return 

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 findHttpURLs.py [PATH_TO_NimbusML_REPOSITORY]")
        exit(1)
    findHttpUrls(sys.argv[1])
    
if __name__ == "__main__":
    main()