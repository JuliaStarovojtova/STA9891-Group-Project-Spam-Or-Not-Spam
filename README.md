# STA9891-Group-Project-Spam-Or-Not-Spam

This project was made for Machine Learning for Data Mining STA 9891 class

The aim of this study is to detect possible features of spam emails and obtain information that later can be used for developing spam filters. Such filters will protect people not only from their attention theft, but also from other dangerous consequences such as theft of important personal information and money.

We are classifying emails based on certain features whether they are spam or not. For this goal, we are using the 4 following methods: Lasso, Elastic-Net, Ridge and Random Forrest.

The dataset comes from UCI Machine Learning Repository. In the dataset, the collection of spam emails came from individuals who had filled spam, while collection of non-spam emails came from filed work and personal e-mails.

In total there are 4601 observations and 57 continuous all-numeric features except target class variable. Out of all observations, 1813 observations correspond to spam emails, while 2788 correspond to non-spam ones. The dataset contains no missing values.

As for the features, 54 of them indicate whether a particular word or character was frequently occurring in the e-mail. For example, the variable “word_freq_free” shows percentage of words in the e-mail that match word FREE. Similarly, we can say about other word variables that display analogous statistics for words such as “credit”, “money”, “receive” etc. The variable “char_freq_!” (Exclamation mark) shows percentage of characters in the e-mail that match Exclamation mark. Also, there are other 3 continuous variables that stand for the average length of uninterrupted sequences of capital letters, length of longest uninterrupted sequence of capital letters and total number of capital letters in the e-mail.

The dataset and more information about how the data collected can be found here: https://archive.ics.uci.edu/ml/datasets/Spambase.
