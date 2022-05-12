from Bio import Entrez
import multiprocessing.dummy as mp
import multiprocessing as mp1
import sys

Entrez.api_key = 'c01cb5f0a7a067a2fe5e57a845b52934c708'
Entrez.email  = "stylianosmavrianos@gmail.com"



def print_abstract(pmid):
    handle = Entrez.efetch(db='pubmed', id=pmid, retmode='text', rettype='XML')
    read = handle.read()
    return read

def find_PMID(pmid):

    ids = []
    index = find_index(pmid)
    record = Entrez.read(Entrez.elink(dbfrom="pubmed", id=pmid))

    for link in record[0]["LinkSetDb"][index]["Link"]:
        ids.append(link["Id"])
    return ids

def find_index(pmid):
    dd = review_ids(pmid)
    for i in range(len(dd.keys())):
        if list(dd.keys())[i] =='pubmed_pubmed_refs':
            return i 



def review_ids(pmid):
    identidier = {}
    record = Entrez.read(Entrez.elink(dbfrom="pubmed", id=pmid))
    for linksetdb in record[0]["LinkSetDb"]:
        identidier[linksetdb["LinkName"]] =len(linksetdb["Link"])
    return identidier


def write(abstracts, names):
    for i in range(len(abstracts)):
        save_path_file = 'output/' + names[i] + ".xml"
        with open(save_path_file, "wb") as f:
                f.write(abstracts[i])



def main():
    # call: python assignment1.py 33669327
    pmid = sys.argv[1]
    cpus =  mp1.cpu_count()


    ids = find_PMID(pmid)
    ten_ids = ids[0:10]


    with mp.Pool(cpus) as pool:
        result = pool.map(print_abstract, ten_ids)

    write(result, ten_ids)



if __name__ == "__main__":
    main()
    print('Finish!!!')
