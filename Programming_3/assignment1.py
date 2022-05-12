from unittest import result
from Bio import Entrez
import multiprocessing.dummy as mp
import random
import sys

from xml.dom import minidom
import os 
  

Entrez.api_key = 'c01cb5f0a7a067a2fe5e57a845b52934c708'
Entrez.email  = "stylianosmavrianos@gmail.com"
 


def print_abstract(pmid):
    handle = Entrez.efetch(db='pubmed', id=pmid, retmode='text', rettype='XML')
    read = handle.read()
    return read


def asd(x):
    return x




def find_PMID(pmid):
    ids = []
    record = Entrez.read(Entrez.elink(dbfrom="pubmed", id=pmid))
    
    for link in record[0]["LinkSetDb"][5]["Link"]:
        ids.append(link["Id"])
    return ids


def review_ids(pmid):
    record = Entrez.read(Entrez.elink(dbfrom="pubmed", id=pmid))
    for linksetdb in record[0]["LinkSetDb"]:
        print(linksetdb["DbTo"], linksetdb["LinkName"], len(linksetdb["Link"]))


def write(abstracts, names):
    for i in range(len(abstracts)):
        save_path_file = names[i] + ".xml"
        with open(save_path_file, "w") as f:
                f.write(str(abstracts[i]))



def main():
    # call: python assignment1.py 33669327
    pmid = sys.argv[1]

    # cpus = mp.cpu_count()

    ids = find_PMID(pmid)
    ten_ids = ids[0:10]


    
    with mp.Pool(4) as pool:
        result = pool.map(print_abstract, ten_ids)
    
    write(result, ten_ids)
 


if __name__ == "__main__":
    main()
