# :open_file_folder: The Simpsons Characters Dataset

The original dataset can be found at [__Kaggle - The Simpsons Characters Data__](https://www.kaggle.com/alexattia/the-simpsons-characters-dataset).

Anyway, in this project a prepared version of the dataset has been used, so as to clean the 
directories with a few images and just keep the ones with enough samples to train the model.

## :mechanical_arm: Train Dataset

The train dataset can be downloaded from the following Dropbox URL: https://www.dropbox.com/s/8u2k79tuqmwrwi8/train.zip

## :test_tube: Test Dataset

The test dataset can be downloaded from the following Dropbox URL: https://www.dropbox.com/s/pnipjr7brjz1pm5/test.zip

---

To download and extract the train and test datasets from the terminal just use the following commands:

```
mkdir dataset/
cd dataset/
wget --no-check-certificate https://www.dropbox.com/s/8u2k79tuqmwrwi8/train.zip -O train.zip
unzip -q train.zip
rm train.zip
wget --no-check-certificate https://www.dropbox.com/s/pnipjr7brjz1pm5/test.zip -O test.zip
unzip -q test.zip
rm test.zip
```

---

Additionally, if you are using Google Colab, just use the following steps to include as a code cell in a
Notebook, so as to download and extract the dataset under the `/content/` directory (by default):

1. Make sure that there are no directories with the same name under the same directory (you can 
modify the names if it exists) and if so, just remove them:

```
!rm -r /content/train
!rm -r /content/test
```

2. Then you need to download both the train and test ZIP files using WGET as it follows:

```
!wget --no-check-certificate \
        https://www.dropbox.com/s/8u2k79tuqmwrwi8/train.zip \
       -O /tmp/train.zip

!wget --no-check-certificate \
        https://www.dropbox.com/s/pnipjr7brjz1pm5/test.zip \
       -O /tmp/test.zip
```

3. Finally, you just need to use `ZipFile` to extract the ZIP files into the `/content/` directory for both train and test sets:

```python
import zipfile

with zipfile.ZipFile("//tmp/train.zip", "r") as zip_ref:
    zip_ref.extractall("/content/train")
zip_ref.close()

with zipfile.ZipFile("//tmp/test.zip", "r") as zip_ref:
    zip_ref.extractall("/content/test")
zip_ref.close()
```

__Note__: you can use the `/tmp/` as it is the recommendation, but if you plan to use this dataset 
frequently it's just better to store it under the `/content/` directory so as to keep them, as `/tmp` is temporary.