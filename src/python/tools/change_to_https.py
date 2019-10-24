# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------

# Converts all valid HTTP links to HTTPS, where the fed
# HTTP links are found in alterable_urls.csv, which
# is generated by find_http_urls.py
# Usage: python3 change_to_https.py urls.csv path_to_repo

import sys
import os
import csv

def changeUrls(pathToReportCsv, pathToRootDirectory):
    with open(pathToReportCsv, newline='') as csvFile:
        csv_reader = csv.reader(csvFile, delimiter='\t')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                #URL: row[0]
                #relativePath: row[1]
                print(row[1])
                absolutePath = pathToRootDirectory+row[1]
                fullText = open(absolutePath).read()
                fullText = fullText.replace(row[0], row[0].replace('http', 'https'))
                f = open(absolutePath, 'w')
                f.write(fullText)
                f.close()
                print("Altered {} in file: {}".format(row[0], absolutePath))
                line_count += 1
        print(f'Processed {line_count} URLs.')

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 change_to_https.py [PATH_TO_alterable_urls.csv] [PATH_TO_ORIGINAL_NIMBUSML_DIRECTORY]")
        exit(1)
    changeUrls(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
