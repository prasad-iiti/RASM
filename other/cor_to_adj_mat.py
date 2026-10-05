corr_mtx = df.corr(method="spearman").abs()
avg_corr = corr_mtx.mean(axis = 1)
up = corr_mtx.where(np.triu(np.ones(corr_mtx.shape), k=1).astype(np.bool_))

drop = list()

for row in range(len(up)-1):
    col_idx = row + 1
    for col in range (col_idx, len(up)):
        if(corr_mtx.iloc[row, col] > 0.8):
            if(avg_corr.iloc[row] > avg_corr.iloc[col]): 
                drop.append(row)
            else: 
                drop.append(col)

drop_set = list(set(drop))
dropcols_names = list(df.columns[[item for item in drop_set]])
print(dropcols_names)
print(len(dropcols_names))
