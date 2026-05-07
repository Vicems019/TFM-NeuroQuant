import glob
TRANSPARENT = "rgba(0,0,0,0)"
files = glob.glob('C:/Users/vicen/Desktop/crypto_dashboard/pages/*.py')
for f in files:
    with open(f, encoding='utf-8') as fh:
        content = fh.read()
    new = content
    new = new.replace('paper_bgcolor="transparent"', f'paper_bgcolor="{TRANSPARENT}"')
    new = new.replace('plot_bgcolor="transparent"',  f'plot_bgcolor="{TRANSPARENT}"')
    with open(f, 'w', encoding='utf-8') as fh:
        fh.write(new)
    print('Fixed:', f)
print("Done")
