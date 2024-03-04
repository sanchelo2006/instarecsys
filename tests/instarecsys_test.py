from instarecsys import Insta

insta = Insta()
sample = insta.load_sample()

print('loaded dataset')
print(sample.head())

df_final = insta.feature(sample)

my_rec = insta.recommend(df_final, users = [1, 2, 7])

print('recommend:')
print(my_rec.head(10))


