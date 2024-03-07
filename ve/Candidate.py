# candidate
import pandas as pd
def learn(concepts, target):
    s = concepts[0].copy()
    g = [["?" for _ in s] for _ in s]
    for i, h in enumerate(concepts):
        if target[i] == 'yes':
            s = [h[x] if h[x] == s[x] else '?' for x in range(len(s))]
            g = [[g[x][y] if h[x] == s[x] else '?' for y in range(len(g))] for x in range(len(g))]
        elif target[i] == 'no':
            for x in range(len(h)):
                if h[x] != s[x]:
                    g[x][x] = s[x]
                else:
                    g[x][x] = '?'
    g = [row for row in g if row != ["?" for _ in s]]
    return s, g

df = pd.read_csv('EnjoySport-2.csv').drop(['slno'], axis=1)
concepts, target = df.values[:, :-1], df.values[:, -1]
s_final, g_final = learn(concepts, target)
print(f"final s: {s_final}")
print(f"final g: {g_final}")
