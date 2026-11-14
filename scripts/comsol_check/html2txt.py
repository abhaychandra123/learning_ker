import re,html,sys
s=open(sys.argv[1],encoding="utf-8").read()
s=re.sub(r"<(script|style)[^>]*>.*?</\1>","",s,flags=re.S)
s=re.sub(r"<img[^>]*src=\"([^\"]+)\"[^>]*>",r" [IMG \1] ",s)
s=re.sub(r"\s+"," ",s)
s=re.sub(r"<(tr|p|div|h\d|li|br)[^>]*>","\n",s)
s=re.sub(r"<t[dh][^>]*>"," | ",s)
s=re.sub(r"<[^>]+>","",s)
s=html.unescape(s)
lines=[re.sub(r"(\s*\|\s*)+"," | ",l).strip(" |") for l in s.split("\n")]
print("\n".join(l for l in lines if l.strip()))
