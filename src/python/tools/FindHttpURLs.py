# Finds all HTTP URLs found in the NimbusML repository
# Converts all valid HTTP links to HTTPS
# Usage: python3 FindHttpURLs.py [PATH_TO_NimbusML_REPOSITORY]
# Output: Report_AlterableUrls_FindHttpURLs.csv, [Report_NonAlterableUrls_FindHttpURLs.csv, Report_InvalidUrls_FindHttpURLs.csv]

# Required non-standard pip library: urlextract

import sys
import os
import requests
import csv
from urlextract import URLExtract

def findHttpUrls(searchRootDirectory):
    extractor = URLExtract()
    for root, _, files in os.walk(searchRootDirectory, onerror=None): 
        for filename in files: 
            filePath = os.path.abspath(os.path.join(root, filename)) 
            try:
                with open(filePath, "rb") as f:
                    data = f.read()
                    try:
                        data = data.decode("utf-8")  
                    except ValueError: 
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
                                        alterableUrlsStore.append([filePath, selectedUrl])
                                    else:
                                        nonAlterableUrlsStore.append([filePath, selectedUrl])
                                except:
                                    nonAlterableUrlsStore.append([filePath, selectedUrl])
                            else:
                                invalidUrlsStore.append([filePath, selectedUrl])
                        except ConnectionError:
                            invalidUrlsStore.append([filePath, selectedUrl])
            except (IOError, OSError):
                pass
    return

def makeReports():
    fieldnames = ['filepath', 'url']
    if makeAlterableUrlsReport:
        with open('Report_AlterableUrls_FindHttpURLs.csv', mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["filepath","url"])
            for pair in alterableUrlsStore:
                writer.writerow([pair[0], pair[1]])
    if makeNonAlterableUrlsStore:
        with open('Report_NonAlterableUrls_FindHttpURLs.csv', mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["filepath","url"])
            for pair in alterableUrlsStore:
                writer.writerow([pair[0], pair[1]])
    if makeInvalidUrlsStore:
        with open('Report_InvalidUrls_FindHttpURLs.csv', mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["filepath","url"])
            for pair in alterableUrlsStore:
                writer.writerow([pair[0], pair[1]])
    return 

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 FindHttpURLs.py [PATH_TO_NimbusML_REPOSITORY]")
        exit(1)
    findHttpUrls(sys.argv[1])
    makeReports()

alterableUrlsStore = []
invalidUrlsStore = []
nonAlterableUrlsStore = []
makeAlterableUrlsReport = True
makeNonAlterableUrlsStore = False
makeInvalidUrlsStore = False
if __name__ == "__main__":
    main()