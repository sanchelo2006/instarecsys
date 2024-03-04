# INSTARECSYS

## DESCRIPTION

This package is for make recommendation on instacart dataset for entry users

For recommendation used two features:
 - total quantity of purchase same item
 - add to cart order

Some features of this package:
- loading fast dataset (using engine - 'pyarrow' of pandas library)
- reduce size of dataset used memory by changing type of values
- easy to use. Just load dataset - build features - make recommendation
- recommendation for: all users, set of users, one user
- have build in sample dataset

## INSTALLATION

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install instarecsys.

```bash
pip install instarecsys
```

## USAGE

```python
# import
from instarecsys import Insta

# create Insta instance
my_insta = Insta()

# we load sample dataset which is build in this library
df = my_insta.load_sample()

# make features
df_final = my_insta.feature(df)

# make recommendation for one user
recsys_one_user = my_insta.recommend(df_final, users = [1])

# make recommendation fo set of users
recsys_set_users = my_insta.recommend(df_final, users = [1, 2, 7])

# make recommendation for all users
recsys_all_users = my_insta.recommend(df_final)
```
## LICENSE

[MIT](https://choosealicense.com/licenses/mit/)

## CONTACTS

my e-mail: sanchelo2006@yandex.ru

any suggestions, questions, discussion always welcome.



