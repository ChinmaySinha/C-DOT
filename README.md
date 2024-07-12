# C-DOT
POWER_CONSUMPTION PREDICTOR

Link to data set - https://www.kaggle.com/datasets/jehanbhathena/smart-meter-data-mathura-and-bareilly ,
from the above dataset I performed my operations on the file named - "CEEW - Smart meter data Bareilly 2020"

Steps to run the lstm based model
1) Run "pwrmgmtlstm.ipynb" file and make sure to install all the libraries using the pip command
2) Correctly edit the path to the dataset according to your system
3) Run all the cells of the notebook
4) I have defined two models, both have same number of layers and neurons, the only difference is that one has dropout %, to prevent overfitting, I used the one without dropout because there was no overfitting in my case.
5) Run the rest of the cells in the notebook
6) Now in the end, save the model using "Joblib" library

Now Open apptest.py

Before running the apptest.py , make sure following things are done:
The files arrangement should be in this manner--
|-MAIN FOLDER
   |
   template
   |-index.html (html file for our website)
   static
   |-style.css (css file for our website)
   CEEW - Smart meter data Bareilly 2020.csv
   |
   apptest.py
   |
   model.pkl
   |
   scaler_feature.pkl
   |
   scaler_target.pkl
   |
   pwrmgmtlstm.ipynb
|

Also make sure that in terminal you run this code - pip install numpy pandas matplotlib seaborn scikit-learn tensorflow statsmodels flask joblib,
otherwise it will show libraries not found error!

now run the apptest.py! the website should load up, after clicking the local server link!

If you wanna use the prophet model, you should follow the same rule, but with pwrmgmtprophet.ipynb, app_prophet , index1.html, style1.css . Although i dont recommned using it as its result accuracy is very low.

I have used parallel proccessing in the apptest.py, i.e. it the prediction speed depends upon your cpu, more number of cores will result into faster prediction.
(the prediction is being calculated for every min on every core available instead of just one)

