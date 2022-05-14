from Bio import Entrez
import multiprocessing as mp
import sys
import time 

Entrez.api_key = 'c01cb5f0a7a067a2fe5e57a845b52934c708'
Entrez.email  = "stylianosmavrianos@gmail.com"



def write(pmid):
    handle = Entrez.efetch(db="pmc", id=pmid, rettype="XML", retmode="text",
                           api_key='c01cb5f0a7a067a2fe5e57a845b52934c708')


    with open(f'output/{pmid}.xml', 'wb') as file:
        file.write(handle.read())
        print("Downloading file:" , pmid)



def get_ref(pmid):
    results = Entrez.read(Entrez.elink(dbfrom="pubmed",
                                db="pmc", 
                                LinkName = "pubmed_pmc_refs",
                                id = pmid,
                                api_key='c01cb5f0a7a067a2fe5e57a845b52934c708'))

    references = [f'{link["Id"]}' for link in results[0]["LinkSetDb"][0]["Link"]]
    return references




def main():
    # call: python assignment1.py 33669327
    pmid = sys.argv[1]
    cpus =  mp.cpu_count()

    ids = get_ref(pmid)

    start_time = int(round(time.time()) *1000)

    with mp.Pool(cpus) as pool:
        pool.map(write, ids[:10])
    stop_time = int(round(time.time())*1000)

    print('Time consumed:',stop_time- start_time, 'ms')





if __name__ == "__main__":
    main()
    
    print('Finish!!!')
