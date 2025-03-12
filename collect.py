import glob

files = sorted(glob.glob("slurm*"))

result = open('result.txt', 'w')

for file in files:
    with open(file, 'r', errors='replace') as f:
        content = f.readlines()
        step = content[0][-2]
        seq_length = str(int(content[1][-2]) - 1)
        backbone = content[2].split()[1]
        freeze = content[3][-2]

        model = backbone + '[' +  step + seq_length + freeze + ']'
        try:
            focal = float(content[-2].split()[1])
            dice = float(content[-6].split()[1])
            iou = float(content[-5].split()[1])
            P = float(content[-4].split()[1])
            R = float(content[-3].split()[1])
            brier = float(content[-7].split()[1])

            result.write(f"{model} & {focal:.6f} & {dice:.5f} & {iou:.5f} & {P:.5f} & {R:.5f} & {brier:.5f} : {file}\n")
        except:
            print(f"fail on {file}")

result.close()