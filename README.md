FinalProject7345.ipynb - this file is an unsupervised learner that uses a precreated 
data set to predict new outfits  -outdated

cnn_model_generator = produces cnn model to detect clothes. I put an interface here for the unsuoervised learner.
Fashion_minst.ipynb - this file is a supervised learner that classifies
clothing from the Fashion_Mnist library. It also returns concatenated clothing.
 
UPT.ipynb - This file takes the concatenated outfits from Fashion_minst.ipynb 
and uses them to predict new outfits

Image_Segmentation_KMeans.ipynb - image segmentation optimizer. Can be used to help with prediction accuracy.

concat.ipynb - concat(list, name) takes a  list of images and a name and creates a 200x200 png image that concatinates those images together into generatedOutfit.png. interfacing with .ipynb files is weird so i just printed out the file paths in the cnn model recommender and pasted those here and ran the main.

multi_object_detection.ipynb - still in work. given a model and an image will detect, classify, and crop out  all recognized objects.
