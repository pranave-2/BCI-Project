import os, sys, urllib.request, concurrent.futures as cf
BASE="https://s3.amazonaws.com/openneuro.org/ds002739/"
def size(url):
    r=urllib.request.Request(url,method="HEAD"); return int(urllib.request.urlopen(r,timeout=60).headers["Content-Length"])
def fetch(url,dest,workers=24,chunk=8_000_000):
    n=size(url)
    with open(dest,"wb") as f: f.truncate(n)
    def part(a):
        b=min(a+chunk,n)-1
        for t in range(8):
            try:
                rq=urllib.request.Request(url,headers={"Range":f"bytes={a}-{b}"})
                d=urllib.request.urlopen(rq,timeout=90).read()
                assert len(d)==b-a+1
                fd=os.open(dest,os.O_WRONLY); os.pwrite(fd,d,a); os.close(fd); return
            except Exception as e: err=e
        raise err
    with cf.ThreadPoolExecutor(workers) as ex: list(ex.map(part,range(0,n,chunk)))
    assert os.path.getsize(dest)==n
def get(rel,dest,**k): fetch(BASE+rel,dest,**k)
if __name__=="__main__": get(sys.argv[1],sys.argv[2])
